"""Prepare les donnees : rendements mensuels, taux sans risque, decoupage chronologique."""
import numpy as np
import pandas as pd

from src import config


def load_raw():
    prices = pd.read_csv(config.DATA_RAW / "prices_etf.csv", index_col="Date", parse_dates=True)
    rf = pd.read_csv(config.DATA_RAW / "risk_free_tb3ms.csv", index_col="Date", parse_dates=True)
    return prices, rf[config.FRED_RISK_FREE_ID]


def check_daily_moves(prices: pd.DataFrame) -> None:
    """Controle qualite : plus fortes variations journalieres (pour reperer des erreurs de donnees)."""
    daily = prices.pct_change(fill_method=None)
    top = daily.stack().abs().sort_values(ascending=False).head(5)
    print("5 plus fortes variations journalieres (en valeur absolue) :")
    for (date, ticker), v in top.items():
        print(f"  {date.date()}  {ticker:<5} {daily.loc[date, ticker]:+.1%}")


def to_monthly_returns(prices: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Rendements mensuels simples des ETF + colonne RF (taux sans risque mensuel)."""
    last_date = prices.index.max()
    # Si le jour ouvre suivant est dans le meme mois, le dernier mois est incomplet : on l'exclut.
    incomplete = (last_date + pd.offsets.BDay(1)).month == last_date.month

    # On garde decembre 2006 pour pouvoir calculer le rendement de janvier 2007.
    monthly = prices.loc["2006-12-01":].resample("ME").last()
    if incomplete:
        monthly = monthly.iloc[:-1]
        print(f"Mois en cours exclu (donnees jusqu'au {last_date.date()}).")

    rets = monthly.pct_change().loc[config.ANALYSIS_START:]

    # FRED : valeur datee du 1er du mois -> on la rattache a la fin du mois ; % annuel -> mensuel
    rf_m = rf.copy()
    rf_m.index = rf_m.index + pd.offsets.MonthEnd(0)
    rets["RF"] = (rf_m / 100 / 12).reindex(rets.index)

    assert rets.isna().sum().sum() == 0, "Valeurs manquantes apres preparation"
    return rets


def split(rets: pd.DataFrame):
    train = rets.loc[: config.TRAIN_END]
    val = rets.loc["2017-01-01": config.VAL_END]
    test = rets.loc["2021-01-01":]
    return train, val, test


def summarize(r: pd.DataFrame) -> pd.DataFrame:
    """Statistiques annualisees par actif (a appeler sur le train uniquement)."""
    assets = r.drop(columns="RF")
    excess = assets.sub(r["RF"], axis=0)
    wealth = (1 + assets).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return pd.DataFrame({
        "Rendement ann.": assets.mean() * 12,
        "Volatilite ann.": assets.std() * np.sqrt(12),
        "Sharpe": excess.mean() / assets.std() * np.sqrt(12),
        "Drawdown max": drawdown.min(),
    })


def main() -> None:
    prices, rf = load_raw()
    check_daily_moves(prices)

    rets = to_monthly_returns(prices, rf)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    rets.to_csv(config.DATA_PROCESSED / "returns_monthly.csv")

    train, val, test = split(rets)
    print(f"\nRendements mensuels : {len(rets)} mois ({rets.index.min().date()} -> {rets.index.max().date()})")
    print(f"  Train      : {len(train):>3} mois ({train.index.min().date()} -> {train.index.max().date()})")
    print(f"  Validation : {len(val):>3} mois ({val.index.min().date()} -> {val.index.max().date()})")
    print(f"  Test       : {len(test):>3} mois ({test.index.min().date()} -> {test.index.max().date()})")

    pd.options.display.float_format = "{:.3f}".format
    print("\nStatistiques annualisees sur le TRAIN uniquement :")
    print(summarize(train))
    print(f"\nTaux sans risque moyen (train) : {train['RF'].mean() * 12:.2%} par an")
    print("\nFichier ecrit : data/processed/returns_monthly.csv")


if __name__ == "__main__":
    main()
