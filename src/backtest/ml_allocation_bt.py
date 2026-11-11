"""Allocation Max Sharpe avec rendements attendus du ML (Ridge, HGB) contre la moyenne historique."""
import pandas as pd

from src import config, profiles
from src.backtest.engine import fixed_weights, walk_forward
from src.metrics import portfolio_stats
from src.optimization import allocators as al
from src.prepare_data import split

START = "2017-01-01"


def main() -> None:
    rets = pd.read_csv(config.DATA_PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
    assets = [c for c in rets.columns if c != "RF"]
    n, mw = len(assets), config.MAX_WEIGHT
    train, _, _ = split(rets)
    targets = profiles.target_vols(train[assets])

    fc = pd.read_csv(config.DATA_PROCESSED / "forecasts.csv", parse_dates=["date", "target_date"])
    overrides = {
        "Max Sharpe (Ridge)": fc.pivot(index="date", columns="asset", values="Ridge")[assets],
        "Max Sharpe (HGB)": fc.pivot(index="date", columns="asset", values="HGB")[assets],
    }

    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200
    pd.options.display.max_columns = 20
    periods = {"VALIDATION 2017-2020": slice(None, config.VAL_END),
               "TEST 2021 -> oct. 2026": slice("2021-01-01", None)}

    for risk, base_target in targets.items():
        tv = config.VOL_BUFFER * base_target

        def scaled(fn):
            return lambda mu, cov, rf: profiles.scale_to_target(fn(mu, cov, rf), cov, tv)

        max_sharpe = lambda mu, cov, rf: al.max_sharpe(mu, cov, rf, mw)
        methods = {
            "Reference": lambda mu, cov, rf, r=risk: fixed_weights(assets, profiles.REFERENCE_MIXES[r]),
            "1/N": scaled(lambda mu, cov, rf: al.equal_weight(n)),
            "Max Sharpe (moyenne hist.)": scaled(max_sharpe),
            "Max Sharpe (Ridge)": scaled(max_sharpe),
            "Max Sharpe (HGB)": scaled(max_sharpe),
        }
        res, turn, fails = walk_forward(rets, assets, methods, START, config.ESTIMATION_WINDOW,
                                        config.COST_BPS, mu_overrides=overrides)
        if any(fails.values()):
            print("Echecs d'optimisation :", {k: v for k, v in fails.items() if v})
        for label, sl in periods.items():
            r, t = res.loc[sl], turn.loc[sl]
            rows = {}
            for m in res.columns:
                s = portfolio_stats(r[m], rets["RF"].loc[r.index])
                s["Rotation ann."] = t[m].mean() * 12
                rows[m] = s
            print(f"\n{risk.upper()} - {label} :")
            print(pd.DataFrame(rows).T[["Rendement ann.", "Volatilite ann.", "Sharpe",
                                        "Drawdown max", "Rotation ann."]])


if __name__ == "__main__":
    main()
