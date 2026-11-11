"""Estimateurs de risque alternatifs : covariance EWMA avec retrecissement de Ledoit-Wolf."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def ewma_estimator(halflife: float):
    """Renvoie une fonction returns -> (mu, cov). mu = moyenne simple ; cov = EWMA (mois recents plus lourds)."""
    def estimate(returns: pd.DataFrame):
        X = returns.values
        T = len(X)
        w = 0.5 ** (np.arange(T - 1, -1, -1) / halflife)
        w = w / w.sum()
        m = (w[:, None] * X).sum(axis=0)
        Xs = (X - m) * np.sqrt(w * T)[:, None]
        cov = LedoitWolf(assume_centered=True).fit(Xs).covariance_
        return X.mean(axis=0), cov
    return estimate
