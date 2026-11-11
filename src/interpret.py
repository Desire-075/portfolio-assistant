"""Interpretabilite du modele Ridge : coefficients standardises et importance par permutation."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config
from src.forecasting import FEATURES, build_dataset, make_models


def main() -> None:
    rets = pd.read_csv(config.DATA_PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
    data = build_dataset(rets).dropna(subset=FEATURES)
    val_end = pd.Timestamp(config.VAL_END)
    train = data[data.index < val_end].dropna(subset=["y"])      # cibles connues jusqu'a fin 2020
    test = data[data.index >= val_end].dropna(subset=["y"])      # cibles 2021 -> oct. 2026

    model = make_models()["Ridge"]
    model.fit(train[FEATURES], train["y"])
    best = model.best_estimator_
    print(f"Ridge : alpha retenu = {model.best_params_['ridge__alpha']} ; "
          f"{len(train)} lignes d'entrainement, {len(test)} de test")

    coefs = pd.Series(best.named_steps["ridge"].coef_ * 100, index=FEATURES)
    print("\nCoefficients standardises (points de % de rendement mensuel par ecart-type de la variable) :")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).round(3))

    base_mse = float(np.mean((test["y"] - best.predict(test[FEATURES])) ** 2))
    pi = permutation_importance(best, test[FEATURES], test["y"],
                                scoring="neg_mean_squared_error", n_repeats=30, random_state=42)
    imp = pd.DataFrame({
        "Hausse du MSE (%)": pi.importances_mean / base_mse * 100,
        "Ecart-type": pi.importances_std / base_mse * 100,
    }, index=FEATURES).sort_values("Hausse du MSE (%)", ascending=False)
    pd.options.display.float_format = "{:.3f}".format
    print("\nImportance par permutation sur le TEST (hausse relative de l'erreur quand la variable est melangee) :")
    print(imp)


if __name__ == "__main__":
    main()
