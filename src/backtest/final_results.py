"""Resultats finaux : assistant v2, references, tests bootstrap du Sharpe, figures et tableau du rapport."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, profiles
from src.agents import agents
from src.backtest.assistant_bt import START, backtest_assistant
from src.backtest.engine import fixed_weights, walk_forward
from src.metrics import portfolio_stats
from src.optimization import allocators as al
from src.prepare_data import split

FIG_DIR = config.ROOT / "reports" / "figures"
ASSISTANT = "Assistant v2"
COLS = ["Rendement ann.", "Volatilite ann.", "Sharpe", "Drawdown max", "Rotation ann."]


def sharpe_arr(x: np.ndarray, rf: np.ndarray) -> float:
    return float((x - rf).mean() / x.std(ddof=1) * np.sqrt(12))


def bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, rf: pd.Series, n_boot: int = 5000,
                          block: int = 6, seed: int = 42):
    """Difference de Sharpe a - b, bootstrap par blocs circulaires (apparie). Renvoie (diff, IC95, p-value)."""
    A, B, R = a.values, b.values, rf.values
    T = len(A)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    obs = sharpe_arr(A, R) - sharpe_arr(B, R)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        diffs[i] = sharpe_arr(A[idx], R[idx]) - sharpe_arr(B[idx], R[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float(np.mean(np.abs(diffs - obs) >= abs(obs)))
    return obs, float(lo), float(hi), p


def plot_performance(res: pd.DataFrame, risk: str) -> None:
    wealth = 100 * (1 + res).cumprod()
    dd = wealth / wealth.cummax() - 1
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for m in res.columns:
        lw = 2.6 if m == ASSISTANT else 1.4
        axes[0].plot(wealth.index, wealth[m], lw=lw, label=m)
        axes[1].plot(dd.index, dd[m] * 100, lw=lw, label=m)
    for ax in axes:
        ax.axvline(pd.Timestamp("2021-01-01"), color="gray", ls="--", lw=1)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"Profil {risk} : croissance et drawdown (trait pointillé : début du test)")
    axes[0].set_ylabel("Valeur (base 100, début 2017)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"05_performance_{risk}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rets = pd.read_csv(config.DATA_PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
    assets = [c for c in rets.columns if c != "RF"]
    n, mw = len(assets), config.MAX_WEIGHT
    train, _, _ = split(rets)
    targets = profiles.target_vols(train[assets])
    agents.SELECTION_MODE = "consensus"

    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200
    pd.options.display.max_columns = 20
    periods = {"VALIDATION 2017-2020": slice(None, config.VAL_END),
               "TEST 2021 -> oct. 2026": slice("2021-01-01", None)}
    all_rows = []

    for risk, base_target in targets.items():
        tv = config.VOL_BUFFER * base_target
        methods = {
            "Reference": lambda mu, cov, rf, r=risk: fixed_weights(assets, profiles.REFERENCE_MIXES[r]),
            "1/N + liquidites": lambda mu, cov, rf: profiles.scale_to_target(al.equal_weight(n), cov, tv),
            "Max Sharpe + liquidites": lambda mu, cov, rf: profiles.scale_to_target(
                al.max_sharpe(mu, cov, rf, mw), cov, tv),
        }
        print(f"\n##### Profil {risk} : calcul en cours...", flush=True)
        res, turn, _ = walk_forward(rets, assets, methods, START, config.ESTIMATION_WINDOW, config.COST_BPS)
        a_ret, a_turn, _ = backtest_assistant(rets, assets, risk, "Fructifier")
        res.insert(0, ASSISTANT, a_ret)
        turn.insert(0, ASSISTANT, a_turn)
        res.to_csv(config.DATA_PROCESSED / f"final_returns_{risk}.csv")
        plot_performance(res, risk)

        for label, sl in periods.items():
            r, t = res.loc[sl], turn.loc[sl]
            rf = rets["RF"].loc[r.index]
            rows = {}
            for m in res.columns:
                s = portfolio_stats(r[m], rf)
                s["Rotation ann."] = t[m].mean() * 12
                rows[m] = s
                all_rows.append({"Profil": risk, "Periode": label, "Methode": m, **s})
            print(f"\n{risk.upper()} - {label} ({len(r)} mois) :")
            print(pd.DataFrame(rows).T[COLS])
            for x, y in [(ASSISTANT, "Reference"), (ASSISTANT, "1/N + liquidites"),
                         ("Max Sharpe + liquidites", "Reference")]:
                obs, lo, hi, p = bootstrap_sharpe_diff(r[x], r[y], rf)
                print(f"  Delta Sharpe {x} - {y} : {obs:+.2f}  IC95% [{lo:+.2f} ; {hi:+.2f}]  p = {p:.3f}")

    out = pd.DataFrame(all_rows)
    out.to_csv(config.ROOT / "reports" / "resultats_finaux.csv", index=False)
    print("\nFichiers ecrits : reports/resultats_finaux.csv et reports/figures/05_performance_*.png")


if __name__ == "__main__":
    main()
