"""Methodes d'allocation : 1/N, inverse volatilite, minimum variance, max Sharpe, HRP."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from src import config
from src.metrics import portfolio_stats
from src.prepare_data import split


# ---------- Estimation ----------
def estimate(returns: pd.DataFrame):
    """Rendements moyens mensuels et covariance (Ledoit-Wolf) a partir d'un historique."""
    mu = returns.mean().values
    cov = LedoitWolf().fit(returns.values).covariance_
    return mu, cov


# ---------- Methodes ----------
def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def inverse_vol(cov: np.ndarray) -> np.ndarray:
    w = 1.0 / np.sqrt(np.diag(cov))
    return w / w.sum()


def _solve(objective, n: int, max_weight: float) -> np.ndarray:
    res = minimize(
        objective,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"Optimisation non convergee : {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def min_variance(cov: np.ndarray, max_weight: float = 0.4) -> np.ndarray:
    return _solve(lambda w: 1e4 * (w @ cov @ w), len(cov), max_weight)


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float, max_weight: float = 0.4) -> np.ndarray:
    return _solve(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), len(mu), max_weight)


def _cluster_var(cov: np.ndarray, idx: list) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp(cov: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado) : arbre de correlation + bisection recursive."""
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method="single")
    order = [int(i) for i in leaves_list(link)]

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters
                    for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for k in range(0, len(clusters), 2):
            c0, c1 = clusters[k], clusters[k + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w.loc[c0] *= alpha
            w.loc[c1] *= 1 - alpha
    return w.sort_index().values


# ---------- Demonstration sur le train, controle sur la validation ----------
def main() -> None:
    rets = pd.read_csv(config.DATA_PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
    train, val, _ = split(rets)
    assets = [c for c in rets.columns if c != "RF"]

    mu, cov = estimate(train[assets])
    rf = train["RF"].mean()

    weights = pd.DataFrame({
        "1/N": equal_weight(len(assets)),
        "Vol inverse": inverse_vol(cov),
        "Min variance": min_variance(cov),
        "Max Sharpe": max_sharpe(mu, cov, rf),
        "HRP": hrp(cov),
    }, index=assets)

    pd.options.display.float_format = "{:.1f}".format
    print("Poids (%) estimes sur le TRAIN :")
    print(weights * 100)
    print("\nSomme des poids :", weights.sum().round(6).to_dict())

    pd.options.display.float_format = "{:.3f}".format
    for name, period in [("TRAIN (dans l'echantillon)", train), ("VALIDATION (hors echantillon, poids fixes)", val)]:
        rows = {m: portfolio_stats(period[assets] @ weights[m], period["RF"]) for m in weights.columns}
        print(f"\n{name} :")
        print(pd.DataFrame(rows).T)


if __name__ == "__main__":
    main()
