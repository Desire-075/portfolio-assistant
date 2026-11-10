"""Analyse exploratoire sur le TRAIN uniquement : distributions, correlations, ACP, clustering."""
import matplotlib

matplotlib.use("Agg")  # sauvegarde des figures sans fenetre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import jarque_bera, kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.prepare_data import split

FIG_DIR = config.ROOT / "reports" / "figures"


def load_train() -> pd.DataFrame:
    rets = pd.read_csv(config.DATA_PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
    train, _, _ = split(rets)
    return train.drop(columns="RF")


def distribution_stats(a: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Asymetrie": skew(a.values, axis=0),
        "Kurtosis exces": kurtosis(a.values, axis=0),
        "JB p-value": [jarque_bera(a[c]).pvalue for c in a.columns],
    }, index=a.columns)


def plot_growth(a: pd.DataFrame) -> None:
    growth = 100 * (1 + a).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    growth.plot(ax=ax, lw=1.5)
    ax.set_title("Croissance de 100 investis dans chaque ETF (train, 2007-2016)")
    ax.set_ylabel("Valeur (base 100)")
    ax.set_xlabel("")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_croissance_train.png", dpi=150)
    plt.close(fig)


def plot_corr(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Corrélations des rendements mensuels (train)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_correlations_train.png", dpi=150)
    plt.close(fig)


def run_pca(a: pd.DataFrame) -> None:
    z = StandardScaler().fit_transform(a)
    pca = PCA().fit(z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    eig = pca.explained_variance_

    print("\nACP (variables centrees-reduites) :")
    for k in range(5):
        print(f"  PC{k + 1}: valeur propre {eig[k]:.2f} | variance {evr[k]:.1%} | cumulee {cum[k]:.1%}")
    print(f"  Composantes avec valeur propre >= 1 : {int((eig >= 1).sum())}")
    print(f"  Composantes pour atteindre 80 % de variance : {int(np.argmax(cum >= 0.8) + 1)}")

    # Correlations variables-facteurs (loadings * sqrt(valeur propre))
    load = pd.DataFrame(
        pca.components_[:3].T * np.sqrt(eig[:3]),
        index=a.columns, columns=["F1", "F2", "F3"],
    )
    print("\nCorrelations variables-facteurs :")
    print(load.round(2))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(evr) + 1)
    ax.bar(x, evr * 100, color="#4C72B0", label="Variance expliquée")
    ax.plot(x, cum * 100, "o-", color="#C44E52", label="Variance cumulée")
    ax.axhline(80, ls="--", color="gray", lw=1)
    ax.set_xlabel("Composante principale")
    ax.set_ylabel("%")
    ax.set_title("ACP : variance expliquée (train)")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_acp_variance.png", dpi=150)
    plt.close(fig)


def run_clustering(a: pd.DataFrame, corr: pd.DataFrame) -> None:
    dist = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method="average")

    print("\nClustering hierarchique (distance de correlation, liaison moyenne) :")
    best_k, best_s = None, -1
    for k in range(2, 6):
        labels = fcluster(link, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(dist, labels, metric="precomputed")
        print(f"  k={k} : silhouette = {s:.3f}")
        if s > best_s:
            best_k, best_s = k, s
    labels = fcluster(link, t=best_k, criterion="maxclust")
    print(f"\nMeilleur k : {best_k}")
    for c in sorted(set(labels)):
        print(f"  Groupe {c}: {', '.join(a.columns[labels == c])}")

    fig, ax = plt.subplots(figsize=(9, 5))
    dendrogram(link, labels=list(a.columns), ax=ax)
    ax.set_title("Dendrogramme des actifs (train)")
    ax.set_ylabel("Distance de corrélation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_dendrogramme_train.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a = load_train()
    print(f"Train : {len(a)} mois, {a.shape[1]} actifs ({a.index.min().date()} -> {a.index.max().date()})")

    pd.options.display.float_format = "{:.3f}".format
    print("\nDistribution des rendements mensuels :")
    print(distribution_stats(a))

    corr = a.corr()
    print("\nCorrelations les plus fortes (hors diagonale) :")
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().dropna().sort_values()
    print("  Les plus negatives :")
    for (i, j), v in pairs.head(3).items():
        print(f"    {i}-{j}: {v:+.2f}")
    print("  Les plus positives :")
    for (i, j), v in pairs.tail(3).items():
        print(f"    {i}-{j}: {v:+.2f}")

    plot_growth(a)
    plot_corr(corr)
    run_pca(a)
    run_clustering(a, corr)
    print(f"\nFigures ecrites dans {FIG_DIR.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()

