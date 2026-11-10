"""Configuration centrale : univers d'actifs, sources, chemins, periodes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# ticker -> (nom, classe d'actifs). ETF cotes aux Etats-Unis.
UNIVERSE = {
    "SPY": ("Actions US large caps", "Actions US"),
    "QQQ": ("Actions US technologie", "Actions US"),
    "IWM": ("Actions US small caps", "Actions US"),
    "EFA": ("Actions pays developpes hors US", "Actions internationales"),
    "EEM": ("Actions pays emergents", "Actions internationales"),
    "AGG": ("Obligations US agregees", "Obligations"),
    "TLT": ("Obligations Tresor long terme", "Obligations"),
    "LQD": ("Obligations d'entreprises", "Obligations"),
    "GLD": ("Or", "Refuges et diversifiants"),
    "VNQ": ("Immobilier cote", "Refuges et diversifiants"),
    "DBC": ("Matieres premieres", "Refuges et diversifiants"),
}
TICKERS = list(UNIVERSE.keys())

# Taux sans risque (liquidites) : taux du Tresor US a 3 mois (FRED), en % annuel
FRED_RISK_FREE_ID = "TB3MS"

# Date de debut du telechargement (large, pour voir les vraies dates de debut de chaque ETF)
DOWNLOAD_START = "2003-01-01"

# Periode d'analyse et decoupage chronologique
ANALYSIS_START = "2007-01-01"   # DBC (le plus recent) existe depuis fevrier 2006
TRAIN_END = "2016-12-31"        # train : 2007-2016
VAL_END = "2020-12-31"          # validation : 2017-2020 ; test : 2021 jusqu'a aujourd'hui

# Backtest et allocation (parametres provisoires)
ESTIMATION_WINDOW = 60   # mois d'historique pour estimer mu et la covariance
MAX_WEIGHT = 0.4         # plafond par actif (min variance, max Sharpe)
COST_BPS = 10            # cout de transaction : 10 points de base par unite de rotation
