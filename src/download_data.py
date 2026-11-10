"""Telecharge les prix des ETF (yfinance) et le taux sans risque (FRED) dans data/raw."""
import datetime as dt

import pandas as pd
import yfinance as yf

from src import config


def download_prices() -> pd.DataFrame:
    """Prix de cloture ajustes (dividendes inclus), frequence journaliere."""
    raw = yf.download(
        config.TICKERS,
        start=config.DOWNLOAD_START,
        auto_adjust=True,
        progress=False,
    )
    prices = raw["Close"].copy()
    prices.index.name = "Date"
    # Garder l'ordre de config et signaler les tickers sans donnees
    missing = [t for t in config.TICKERS if t not in prices.columns or prices[t].isna().all()]
    if missing:
        print(f"ATTENTION : aucune donnee pour {missing}")
    return prices[[t for t in config.TICKERS if t not in missing]]


def download_risk_free() -> pd.DataFrame:
    """Taux du Tresor US a 3 mois, mensuel, en % annuel."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={config.FRED_RISK_FREE_ID}"
    df = pd.read_csv(url)
    col = config.FRED_RISK_FREE_ID
    df.columns = ["Date", col]  # le nom de la 1re colonne varie selon la version de FRED
    df["Date"] = pd.to_datetime(df["Date"])
    df[col] = pd.to_numeric(df[col], errors="coerce")  # FRED note "." les valeurs manquantes
    df = df.set_index("Date")
    return df[df.index >= config.DOWNLOAD_START]


def main() -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    prices = download_prices()
    prices.to_csv(config.DATA_RAW / "prices_etf.csv")
    print(f"\nPrix ETF : {prices.shape[0]} jours x {prices.shape[1]} actifs")
    print(f"{'Ticker':<8}{'Premiere date':<16}{'Derniere date':<16}{'Valeurs manquantes':>20}")
    for t in prices.columns:
        s = prices[t].dropna()
        print(f"{t:<8}{str(s.index.min().date()):<16}{str(s.index.max().date()):<16}{prices[t].isna().sum():>20}")

    rf = download_risk_free()
    rf.to_csv(config.DATA_RAW / "risk_free_tb3ms.csv")
    print(f"\nTaux sans risque ({config.FRED_RISK_FREE_ID}) : {len(rf)} mois, "
          f"de {rf.index.min().date()} a {rf.index.max().date()}")

    sources = (
        f"Date de telechargement : {dt.date.today()}\n"
        "prices_etf.csv : prix de cloture ajustes (dividendes inclus), journaliers, source Yahoo Finance via yfinance\n"
        f"risk_free_tb3ms.csv : taux du Tresor US a 3 mois (% annuel, mensuel), source FRED serie {config.FRED_RISK_FREE_ID}\n"
        f"Actifs : {', '.join(config.TICKERS)}\n"
    )
    (config.DATA_RAW / "SOURCES.txt").write_text(sources, encoding="utf-8")
    print("\nFichiers ecrits dans data/raw : prices_etf.csv, risk_free_tb3ms.csv, SOURCES.txt")


if __name__ == "__main__":
    main()
