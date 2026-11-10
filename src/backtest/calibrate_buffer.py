"""Calibrage de la marge de securite sur la volatilite cible, sur la VALIDATION uniquement."""
import numpy as np
import pandas as pd

from src import config, profiles
from src.backtest.engine import walk_forward
from src.optimization import allocators as al
from src.prepare_data import split

BUFFERS = [1.0, 0.9, 0.8, 0.7, 0.6]


def main() -> None:
    rets = pd.read_csv(config.DATA_PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
    assets = [c for c in rets.columns if c != "RF"]
    n, mw = len(assets), config.MAX_WEIGHT
    train, _, _ = split(rets)
    target = profiles.target_vols(train[assets])["Prudent"]

    base = {
        "1/N": lambda mu, cov, rf: al.equal_weight(n),
        "Vol inverse": lambda mu, cov, rf: al.inverse_vol(cov),
        "Min variance": lambda mu, cov, rf: al.min_variance(cov, mw),
        "Max Sharpe": lambda mu, cov, rf: al.max_sharpe(mu, cov, rf, mw),
        "HRP": lambda mu, cov, rf: al.hrp(cov),
    }
    methods = {
        f"{b:.1f}|{m}": (lambda mu, cov, rf, fn=fn, b=b: profiles.scale_to_target(fn(mu, cov, rf), cov, target, b))
        for b in BUFFERS for m, fn in base.items()
    }
    res, _, _ = walk_forward(rets, assets, methods, "2017-01-01",
                             config.ESTIMATION_WINDOW, config.COST_BPS)
    val = res.loc[:config.VAL_END]          # validation uniquement
    vol = val.std() * np.sqrt(12)

    table = pd.DataFrame(
        {m: [vol[f"{b:.1f}|{m}"] / target for b in BUFFERS] for m in base},
        index=[f"buffer {b:.1f}" for b in BUFFERS],
    )
    table["Moyenne"] = table.mean(axis=1)
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 200
    print(f"Profil Prudent, cible {target:.1%}. Rapport volatilite realisee / cible "
          f"(validation, {len(val)} mois) ; 1.00 = cible atteinte :")
    print(table)
    best = (table["Moyenne"] - 1).abs().idxmin()
    print(f"\nMeilleure marge sur la validation : {best}")


if __name__ == "__main__":
    main()
