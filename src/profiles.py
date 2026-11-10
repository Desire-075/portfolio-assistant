"""Profils d'investisseur : cibles de volatilite et mise a l'echelle vers les liquidites."""
import numpy as np
import pandas as pd

# Melanges de reference SPY/AGG qui ancrent le niveau de risque de chaque profil
REFERENCE_MIXES = {
    "Prudent": {"SPY": 0.2, "AGG": 0.8},
    "Equilibre": {"SPY": 0.5, "AGG": 0.5},
    "Dynamique": {"SPY": 0.8, "AGG": 0.2},
}


def target_vols(train_assets: pd.DataFrame) -> dict:
    """Volatilite annualisee de chaque melange de reference, mesuree sur le TRAIN uniquement."""
    out = {}
    for name, spec in REFERENCE_MIXES.items():
        w = pd.Series(spec).reindex(train_assets.columns).fillna(0.0)
        out[name] = float((train_assets @ w).std() * np.sqrt(12))
    return out


def scale_to_target(w_risky: np.ndarray, cov: np.ndarray, target_vol: float,
                    buffer: float = 1.0) -> np.ndarray:
    """Reduit le portefeuille risque pour viser buffer x cible ; le reste va en liquidites."""
    vol = float(np.sqrt(w_risky @ cov @ w_risky * 12))
    return w_risky * min(1.0, buffer * target_vol / vol)
