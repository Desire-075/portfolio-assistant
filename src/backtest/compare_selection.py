"""Compare la selection v1 (meilleur candidat) et v2 (consensus) de l'assistant."""
import pandas as pd

from src import config, profiles
from src.agents import agents
from src.backtest.assistant_bt import START, backtest_assistant
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

    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200
    pd.options.display.max_columns = 20
    periods = {"VALIDATION 2017-2020": slice(None, config.VAL_END),
               "TEST 2021 -> oct. 2026": slice("2021-01-01", None)}
    v1, v2 = "ASSISTANT v1 (meilleur)", "ASSISTANT v2 (consensus)"

    for risk, base_target in targets.items():
        tv = config.VOL_BUFFER * base_target
        methods = {
            "Reference": lambda mu, cov, rf, r=risk: fixed_weights(assets, profiles.REFERENCE_MIXES[r]),
            "1/N": lambda mu, cov, rf: profiles.scale_to_target(al.equal_weight(n), cov, tv),
            "Max Sharpe": lambda mu, cov, rf: profiles.scale_to_target(al.max_sharpe(mu, cov, rf, mw), cov, tv),
        }
        print(f"\n##### Profil {risk} : calcul en cours...", flush=True)
        res, turn, _ = walk_forward(rets, assets, methods, START, config.ESTIMATION_WINDOW, config.COST_BPS)
        for mode, label in [("best", v1), ("consensus", v2)]:
            agents.SELECTION_MODE = mode
            r_, t_, _ = backtest_assistant(rets, assets, risk, "Fructifier")
            res[label], turn[label] = r_, t_
        cols = [v1, v2, "Reference", "1/N", "Max Sharpe"]
        for label, sl in periods.items():
            r, t = res.loc[sl, cols], turn.loc[sl, cols]
            rows = {}
            for m in cols:
                s = portfolio_stats(r[m], rets["RF"].loc[r.index])
                s["Rotation ann."] = t[m].mean() * 12
                rows[m] = s
            print(f"\n{risk.upper()} - {label} :")
            print(pd.DataFrame(rows).T[["Rendement ann.", "Volatilite ann.", "Sharpe",
                                        "Drawdown max", "Rotation ann."]])


if __name__ == "__main__":
    main()
