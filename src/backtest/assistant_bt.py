"""Backtest de l'assistant complet (tous les agents) : decision mensuelle avec l'information passee."""
import numpy as np
import pandas as pd

from src import config, profiles
from src.agents import agents
from src.backtest.engine import fixed_weights, walk_forward
from src.metrics import portfolio_stats
from src.optimization import allocators as al
from src.prepare_data import split

START = "2017-01-01"


def backtest_assistant(rets: pd.DataFrame, assets: list, risk: str, objective: str):
    """Renvoie (rendements nets, rotation, methode choisie) pour chaque mois."""
    idx = rets.index
    p0 = int(idx.searchsorted(pd.Timestamp(START)))
    profile = agents.InvestorProfile(capital=100_000, risk=risk, objective=objective)
    out, turn, picks = [], [], []
    drifted = None
    for p in range(p0, len(idx)):
        ctx = agents.run_assistant(profile, as_of=idx[p - 1])   # information jusqu'a la fin du mois p-1
        w, cash = ctx.selected.weights, ctx.selected.cash
        r = rets.iloc[p][assets].values.astype(float)
        rf_now = float(rets.iloc[p]["RF"])
        pos = np.r_[w, cash]
        to = 0.0 if drifted is None else float(np.abs(pos - drifted).sum())
        gross = float(w @ r) + cash * rf_now
        out.append(gross - to * config.COST_BPS / 1e4)
        turn.append(to)
        picks.append(ctx.selected.name)
        drifted = np.r_[w * (1 + r), cash * (1 + rf_now)] / (1 + gross)
    dates = idx[p0:]
    return pd.Series(out, index=dates), pd.Series(turn, index=dates), pd.Series(picks, index=dates)


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

    for risk, base_target in targets.items():
        tv = config.VOL_BUFFER * base_target
        base = {
            "1/N": lambda mu, cov, rf: al.equal_weight(n),
            "Vol inverse": lambda mu, cov, rf: al.inverse_vol(cov),
            "Min variance": lambda mu, cov, rf: al.min_variance(cov, mw),
            "Max Sharpe": lambda mu, cov, rf: al.max_sharpe(mu, cov, rf, mw),
            "HRP": lambda mu, cov, rf: al.hrp(cov),
        }
        methods = {"Reference": lambda mu, cov, rf, r=risk: fixed_weights(assets, profiles.REFERENCE_MIXES[r])}
        for name, fn in base.items():
            methods[name] = lambda mu, cov, rf, fn=fn: profiles.scale_to_target(fn(mu, cov, rf), cov, tv)

        print(f"\n##### Profil {risk} (cible effective {tv:.1%}) : calcul en cours...", flush=True)
        res, turn, _ = walk_forward(rets, assets, methods, START, config.ESTIMATION_WINDOW, config.COST_BPS)
        a_ret, a_turn, picks = backtest_assistant(rets, assets, risk, "Fructifier")
        res.insert(0, "ASSISTANT", a_ret)
        turn.insert(0, "ASSISTANT", a_turn)

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
            print("Methodes choisies par l'agent de selection :", picks.loc[sl].value_counts().to_dict())


if __name__ == "__main__":
    main()
