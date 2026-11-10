"""Metriques de performance d'un portefeuille a partir de rendements mensuels."""
import numpy as np
import pandas as pd


def portfolio_stats(r: pd.Series, rf: pd.Series) -> dict:
    """r : rendements mensuels du portefeuille ; rf : taux sans risque mensuel (meme index)."""
    excess = r - rf
    wealth = (1 + r).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, wealth.values])[1:]
    return {
        "Rendement ann.": r.mean() * 12,
        "CAGR": wealth.iloc[-1] ** (12 / len(r)) - 1,
        "Volatilite ann.": r.std() * np.sqrt(12),
        "Sharpe": excess.mean() / r.std() * np.sqrt(12),
        "Drawdown max": (wealth.values / peak - 1).min(),
    }
