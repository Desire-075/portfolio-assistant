"""Prevision des rendements mensuels : Ridge et boosting contre la moyenne historique (walk-forward)."""
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config

FEATURES = ["r1", "mom3", "mom6", "mom12", "vol6", "vol12", "dd12", "rf", "drf3"]


def build_dataset(rets: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (mois t, ETF). Variables calculees avec le passe jusqu'a t ; cible y = rendement de t+1."""
    assets = [c for c in rets.columns if c != "RF"]
    rf = rets["RF"] * 12
    frames = []
    for t in assets:
        r = rets[t]
        lr = np.log1p(r)
        lw = lr.cumsum()
        frames.append(pd.DataFrame({
            "asset": t,
            "r1": r,
            "mom3": lr.rolling(3).sum(),
            "mom6": lr.rolling(6).sum(),
            "mom12": lr.rolling(12).sum(),
            "vol6": r.rolling(6).std(),
            "vol12": r.rolling(12).std(),
            "dd12": lw - lw.rolling(12).max(),
            "rf": rf,
            "drf3": rf - rf.shift(3),
            "y": r.shift(-1),
        }))
    data = pd.concat(frames).sort_index(kind="stable")
    data.index.name = "date"
    return data


def make_models() -> dict:
    ridge = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": [1, 10, 100, 1000]},
        cv=TimeSeriesSplit(5),
        scoring="neg_mean_squared_error",
    )
    hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                        l2_regularization=1.0, random_state=42)
    return {"Ridge": ridge, "HGB": hgb}


def walk_forward_forecast(rets: pd.DataFrame, start: str = "2016-12-31", window: int = 60) -> pd.DataFrame:
    """A chaque date de decision d : entrainement sur les lignes dont la cible est connue a d, prevision de d+1."""
    assets = [c for c in rets.columns if c != "RF"]
    data = build_dataset(rets).dropna(subset=FEATURES)
    dates = list(rets.index)
    decision_dates = [d for d in dates[:-1] if d >= pd.Timestamp(start)]
    rows = []
    for k, d in enumerate(decision_dates, 1):
        train = data[data.index < d].dropna(subset=["y"])    # cibles realisees avant d
        test = data[data.index == d]
        preds = {}
        for name, model in make_models().items():
            model.fit(train[FEATURES], train["y"])
            preds[name] = model.predict(test[FEATURES])
        hist_mean = rets[assets].loc[:d].tail(window).mean()
        rows.append(pd.DataFrame({
            "date": d,
            "target_date": dates[dates.index(d) + 1],
            "asset": test["asset"].values,
            "y": test["y"].values,
            "HistMean": hist_mean.reindex(test["asset"]).values,
            **preds,
        }))
        if k % 20 == 0:
            print(f"  {k}/{len(decision_dates)} dates traitees", flush=True)
    return pd.concat(rows, ignore_index=True)


def dm_test(d: pd.Series):
    """Diebold-Mariano (horizon 1) : H0 E[d_t] = 0, avec d_t = perte(baseline) - perte(modele)."""
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))


def evaluate(pred: pd.DataFrame) -> None:
    val_end = pd.Timestamp(config.VAL_END)
    periods = {"VALIDATION (cibles 2017-2020)": pred["target_date"] <= val_end,
               "TEST (cibles 2021 -> oct. 2026)": pred["target_date"] > val_end}
    for label, mask in periods.items():
        p = pred[mask]
        sse_base = ((p["y"] - p["HistMean"]) ** 2).sum()
        rows = {}
        for model in ["HistMean", "Ridge", "HGB"]:
            err = p["y"] - p[model]
            ics = [spearmanr(g[model], g["y"])[0] for _, g in p.groupby("target_date")]
            row = {"RMSE (%)": np.sqrt((err ** 2).mean()) * 100,
                   "R2 vs moyenne": 1 - (err ** 2).sum() / sse_base,
                   "IC moyen": np.nanmean(ics),
                   "% mois IC>0": np.mean(np.array(ics) > 0) * 100}
            if model != "HistMean":
                d = ((p["y"] - p["HistMean"]) ** 2 - err ** 2).groupby(p["target_date"]).mean()
                row["DM stat"], row["DM p-value"] = dm_test(d)
            rows[model] = row
        print(f"\n{label} : {p['target_date'].nunique()} mois, {len(p)} previsions")
        print(pd.DataFrame(rows).T)


def main() -> None:
    rets = pd.read_csv(config.DATA_PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
    print("Prevision walk-forward en cours (quelques minutes)...", flush=True)
    pred = walk_forward_forecast(rets)
    pred.to_csv(config.DATA_PROCESSED / "forecasts.csv", index=False)
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200
    pd.options.display.max_columns = 20
    evaluate(pred)
    print("\nFichier ecrit : data/processed/forecasts.csv")


if __name__ == "__main__":
    main()
