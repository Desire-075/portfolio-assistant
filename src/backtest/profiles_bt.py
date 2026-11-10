"""Backtest par profil : chaque methode est ramenee a la volatilite cible du profil."""
import pandas as pd

from src import config, profiles
from src.backtest.engine import fixed_weights, walk_forward
from src.metrics import portfolio_stats
from src.optimization import allocators as al
from src.prepare_data import split


def main() -> None:
    rets = pd.read_csv(config.DATA_PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
    assets = [c for c in rets.columns if c != "RF"]
    n, mw = len(assets), config.MAX_WEIGHT
    train, _, _ = split(rets)

    targets = profiles.target_vols(train[assets])
    print("Volatilites cibles (melanges de reference, train) :",
          {k: f"{v:.1%}" for k, v in targets.items()})

    base = {
        "1/N": lambda mu, cov, rf: al.equal_weight(n),
        "Vol inverse": lambda mu, cov, rf: al.inverse_vol(cov),
        "Min variance": lambda mu, cov, rf: al.min_variance(cov, mw),
        "Max Sharpe": lambda mu, cov, rf: al.max_sharpe(mu, cov, rf, mw),
        "HRP": lambda mu, cov, rf: al.hrp(cov),
    }
    methods = {}
    for prof, tv in targets.items():
        methods[f"{prof} | Reference"] = (
            lambda mu, cov, rf, p=prof: fixed_weights(assets, profiles.REFERENCE_MIXES[p]))
        for mname, fn in base.items():
            methods[f"{prof} | {mname}"] = (
                lambda mu, cov, rf, fn=fn, tv=tv: profiles.scale_to_target(fn(mu, cov, rf), cov, tv))

    res, turn, fails = walk_forward(rets, assets, methods, "2017-01-01",
                                    config.ESTIMATION_WINDOW, config.COST_BPS)
    if any(fails.values()):
        print("Echecs d'optimisation :", {k: v for k, v in fails.items() if v})

    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200
    pd.options.display.max_columns = 20
    periods = {"VALIDATION 2017-2020": slice(None, config.VAL_END),
               "TEST 2021 -> oct. 2026": slice("2021-01-01", None)}
    for prof, tv in targets.items():
        for label, sl in periods.items():
            r, t = res.loc[sl], turn.loc[sl]
            rows = {}
            for m in [c for c in res.columns if c.startswith(prof + " |")]:
                s = portfolio_stats(r[m], rets["RF"].loc[r.index])
                s["Rotation ann."] = t[m].mean() * 12
                rows[m.split("| ")[1]] = s
            print(f"\n{prof.upper()} (vol. cible {tv:.1%}) - {label} :")
            print(pd.DataFrame(rows).T[["Rendement ann.", "Volatilite ann.", "Sharpe",
                                        "Drawdown max", "Rotation ann."]])


if __name__ == "__main__":
    main()
