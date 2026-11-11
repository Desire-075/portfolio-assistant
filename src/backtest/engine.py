"""Backtest walk-forward : chaque mois, on re-estime avec les 60 derniers mois puis on investit.

Une methode renvoie des poids sur les actifs ; le poids en liquidites est implicite (1 - somme)
et rapporte le taux sans risque du mois.
"""
import numpy as np
import pandas as pd

from src import config
from src.metrics import portfolio_stats
from src.optimization import allocators as al


def fixed_weights(assets: list, spec: dict) -> np.ndarray:
    w = np.zeros(len(assets))
    for ticker, v in spec.items():
        w[assets.index(ticker)] = v
    return w


def walk_forward(rets: pd.DataFrame, assets: list, methods: dict, start: str,
                 window: int, cost_bps: float, mu_overrides: dict | None = None, estimator=None):
    """Renvoie (rendements nets par methode, rotation par methode, nombre d'echecs d'optimisation)."""
    idx = rets.index
    p0 = int(idx.searchsorted(pd.Timestamp(start)))
    net_returns = {m: [] for m in methods}
    turnover = {m: [] for m in methods}
    fails = {m: 0 for m in methods}
    drifted = {m: None for m in methods}
    last_w = {}

    for p in range(p0, len(idx)):
        hist = rets.iloc[p - window: p]          # information disponible jusqu'a la fin du mois p-1
        mu, cov = (estimator or al.estimate)(hist[assets])
        rf = hist["RF"].mean()
        r = rets.iloc[p][assets].values.astype(float)   # rendement realise du mois p
        rf_now = float(rets.iloc[p]["RF"])              # taux sans risque du mois p

        for name, fn in methods.items():
            try:
                mu_m = mu_overrides[name].loc[idx[p - 1]].values if mu_overrides and name in mu_overrides else mu
                w = fn(mu_m, cov, rf)
            except RuntimeError:
                fails[name] += 1
                w = last_w.get(name, np.full(len(assets), 1.0 / len(assets)))
            last_w[name] = w
            cash = max(0.0, 1.0 - float(w.sum()))
            pos = np.r_[w, cash]
            to = 0.0 if drifted[name] is None else float(np.abs(pos - drifted[name]).sum())
            gross = float(w @ r) + cash * rf_now
            net_returns[name].append(gross - to * cost_bps / 1e4)
            turnover[name].append(to)
            drifted[name] = np.r_[w * (1 + r), cash * (1 + rf_now)] / (1 + gross)

    dates = idx[p0:]
    return pd.DataFrame(net_returns, index=dates), pd.DataFrame(turnover, index=dates), fails


def main() -> None:
    rets = pd.read_csv(config.DATA_PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
    assets = [c for c in rets.columns if c != "RF"]
    n = len(assets)
    mw = config.MAX_WEIGHT

    methods = {
        "1/N": lambda mu, cov, rf: al.equal_weight(n),
        "Vol inverse": lambda mu, cov, rf: al.inverse_vol(cov),
        "Min variance": lambda mu, cov, rf: al.min_variance(cov, mw),
        "Max Sharpe": lambda mu, cov, rf: al.max_sharpe(mu, cov, rf, mw),
        "HRP": lambda mu, cov, rf: al.hrp(cov),
        "SPY seul": lambda mu, cov, rf: fixed_weights(assets, {"SPY": 1.0}),
        "60/40 SPY-AGG": lambda mu, cov, rf: fixed_weights(assets, {"SPY": 0.6, "AGG": 0.4}),
    }

    res, turn, fails = walk_forward(rets, assets, methods, "2017-01-01",
                                    config.ESTIMATION_WINDOW, config.COST_BPS)
    print(f"Backtest : {len(res)} mois ({res.index.min().date()} -> {res.index.max().date()}), "
          f"fenetre {config.ESTIMATION_WINDOW} mois, cout {config.COST_BPS} bps")
    if any(fails.values()):
        print("Echecs d'optimisation :", {k: v for k, v in fails.items() if v})

    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200
    pd.options.display.max_columns = 20
    periods = {
        "VALIDATION 2017-2020": slice(None, config.VAL_END),
        "TEST 2021 -> oct. 2026": slice("2021-01-01", None),
    }
    for label, sl in periods.items():
        r, t = res.loc[sl], turn.loc[sl]
        rows = {}
        for m in res.columns:
            s = portfolio_stats(r[m], rets["RF"].loc[r.index])
            s["Rotation ann."] = t[m].mean() * 12
            rows[m] = s
        print(f"\n{label} ({len(r)} mois) :")
        print(pd.DataFrame(rows).T)

    res.to_csv(config.DATA_PROCESSED / "backtest_returns.csv")
    print("\nFichier ecrit : data/processed/backtest_returns.csv")


if __name__ == "__main__":
    main()


