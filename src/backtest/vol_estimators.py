"""Compare des estimateurs de risque (echantillon vs EWMA) pour le ciblage de volatilite.

Critere : rapport volatilite realisee / volatilite cible nominale, avec marge = 1.0.
Valeur ideale apres calibrage de la marge sur la validation : proche de 1 et stable.
"""
import numpy as np
import pandas as pd

from src import config, profiles
from src.backtest.engine import walk_forward
from src.optimization import allocators as al
from src.optimization.risk_models import ewma_estimator
from src.prepare_data import split

START = "2017-01-01"
ESTIMATORS = {
    "Echantillon 60m (actuel)": None,
    "EWMA demi-vie 6 mois": ewma_estimator(6),
    "EWMA demi-vie 12 mois": ewma_estimator(12),
    "EWMA demi-vie 24 mois": ewma_estimator(24),
}


def main() -> None:
    rets = pd.read_csv(config.DATA_PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
    assets = [c for c in rets.columns if c != "RF"]
    n, mw = len(assets), config.MAX_WEIGHT
    train, _, _ = split(rets)
    targets = profiles.target_vols(train[assets])

    val_rows, test_rows = {}, {}
    for est_name, est in ESTIMATORS.items():
        print(f"Estimateur : {est_name}", flush=True)
        vr, tr = {}, {}
        for risk in ["Prudent", "Equilibre"]:
            tv = targets[risk]      # cible nominale, sans marge de securite

            def scaled(fn, tv=tv):
                return lambda mu, cov, rf: profiles.scale_to_target(fn(mu, cov, rf), cov, tv)

            methods = {
                "1/N": scaled(lambda mu, cov, rf: al.equal_weight(n)),
                "Max Sharpe": scaled(lambda mu, cov, rf: al.max_sharpe(mu, cov, rf, mw)),
            }
            res, _, _ = walk_forward(rets, assets, methods, START, config.ESTIMATION_WINDOW,
                                     config.COST_BPS, estimator=est)
            for m in methods:
                key = f"{risk[:4]} {m}"
                vr[key] = res.loc[:config.VAL_END, m].std() * np.sqrt(12) / tv
                tr[key] = res.loc["2021-01-01":, m].std() * np.sqrt(12) / tv
        val_rows[est_name], test_rows[est_name] = vr, tr

    val, test = pd.DataFrame(val_rows).T, pd.DataFrame(test_rows).T
    val["Moyenne"], test["Moyenne"] = val.mean(axis=1), test.mean(axis=1)
    val["Ecart relatif"] = val.iloc[:, :4].std(axis=1) / val["Moyenne"]
    val["Marge implicite"] = 1 / val["Moyenne"]
    test["Ratio avec marge de la validation"] = val["Marge implicite"] * test["Moyenne"]

    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 220
    pd.options.display.max_columns = 20
    print("\nVALIDATION : volatilite realisee / cible nominale (marge 1.0) :")
    print(val)
    print("\nTEST : volatilite realisee / cible nominale (marge 1.0) :")
    print(test)


if __name__ == "__main__":
    main()
