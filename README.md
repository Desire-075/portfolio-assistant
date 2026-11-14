# Assistant multi-agent de sélection de portefeuilles financiers

**Application en ligne** : https://portfolio-ecc.streamlit.app/

Projet de Machine Learning, École Centrale Casablanca (sujet 4), novembre 2026.

**Équipe** : Mouhamadou Bachir NGOM, Alain KISSOU, Mory Désiré OUATTARA, Mahamoudou OUEDRAOGO.
**Encadrant** : Pr. Youssef LAMRANI ALAOUI.

## Présentation

Placer son épargne suppose de choisir parmi de nombreux produits, de comprendre leurs risques et de décider quelle part de son capital confier à chacun. Ce projet construit un assistant qui aide à cette décision, et surtout qui mesure honnêtement ce que valent ses propositions.

L'investisseur indique son capital, la durée de son placement, son objectif et son profil de risque (Prudent, Équilibré ou Dynamique). Sept agents logiciels spécialisés étudient alors les marchés, comparent plusieurs répartitions possibles, contrôlent le risque de chacune et proposent une allocation expliquée en langage courant, avec la part gardée en liquidités.

Les sept agents, qui partagent un contexte commun et laissent une trace de chacune de leurs décisions :

| Agent | Rôle |
|---|---|
| Données | Prépare les 60 derniers mois de rendements, sans jamais utiliser une donnée postérieure à la date de décision |
| Profil | Traduit les réponses du formulaire en niveau de risque visé |
| Analyse | Décrit l'univers : statistiques par fonds, corrélations, familles d'actifs |
| Optimisation | Construit cinq répartitions candidates et les ramène vers les liquidités pour atteindre le risque visé |
| Risque | Accepte ou refuse chaque candidat selon des règles explicites (volatilité, concentration, diversification) |
| Sélection | Retient le portefeuille final à partir des candidats acceptés |
| Synthèse | Rédige l'explication destinée à l'investisseur |

## Résultats principaux

L'univers comprend 11 ETF cotés aux États-Unis (actions, obligations, or, immobilier, matières premières) et le taux sans risque. Les données sont mensuelles, de janvier 2007 à octobre 2026 (238 mois). Le découpage est chronologique : entraînement de 2007 à 2016, validation de 2017 à 2020, test de 2021 à octobre 2026.

| Question | Résultat |
|---|---|
| Les modèles de prévision (Ridge, gradient boosting) battent-ils la moyenne historique ? | Non. Ridge est indiscernable de la référence (test de Diebold-Mariano : p = 0,53 en validation, 0,32 en test). Le boosting est significativement moins bon (p = 0,003 et 0,015). |
| Une méthode d'allocation domine-t-elle ? | Non. Sur 18 comparaisons de ratios de Sharpe par bootstrap, 2 sont significatives à 5 %, ce qui est proche de ce que le hasard produit. |
| Qu'apporte l'assistant ? | Le ciblage du risque par profil avec des liquidités. Profil Prudent sur le test : volatilité de 3,9 % et perte maximale de -9,0 %, contre 7,2 % et -16,3 % pour un mélange classique de 20 % d'actions et 80 % d'obligations. |
| Peut-on mieux estimer le risque ? | Non. Les estimateurs à pondération exponentielle (EWMA) n'améliorent pas l'estimation sur 60 mois. La marge de sécurité de 0,8 appliquée à la volatilité cible reste stable. |

Nous rapportons aussi les essais qui n'ont pas abouti : une première règle de sélection qui s'est effondrée sur le test, des prévisions par apprentissage automatique sans gain démontré, et des estimateurs de risque sans effet. Tout est détaillé dans le rapport.

## Structure du dépôt

```
app/streamlit_app.py        application web
src/config.py               univers d'actifs, dates, paramètres
src/download_data.py        téléchargement (yfinance, FRED)
src/prepare_data.py         rendements mensuels, découpage chronologique
src/eda.py                  analyse exploratoire (entraînement uniquement)
src/metrics.py              métriques de performance
src/optimization/           méthodes d'allocation, estimateurs de risque
src/profiles.py             profils d'investisseur, ciblage de volatilité
src/agents/agents.py        les sept agents et l'orchestrateur
src/forecasting.py          prévision des rendements (Ridge, boosting)
src/interpret.py            interprétabilité (importance par permutation)
src/backtest/               moteur walk-forward, calibrages, résultats finaux
data/raw/                   données brutes téléchargées (versionnées)
reports/                    figures et tableau des résultats finaux
```

## Installation

Python 3.10 ou plus récent est recommandé.

```
python -m venv .venv
.venv\Scripts\Activate.ps1        (Windows)
source .venv/bin/activate         (Mac et Linux)
pip install -r requirements.txt
```

## Reproduire les résultats

Les données brutes sont déjà dans `data/raw`. Pour les retélécharger, lancer la première commande : les prix ajustés de Yahoo Finance peuvent légèrement évoluer avec le temps.

```
python -m src.download_data                  téléchargement
python -m src.prepare_data                   rendements mensuels et découpage
python -m src.eda                            analyse exploratoire, figures 01 à 04
python -m src.optimization.allocators        poids des méthodes d'allocation
python -m src.backtest.engine                backtest walk-forward des méthodes
python -m src.backtest.profiles_bt           backtest par profil
python -m src.backtest.calibrate_buffer      calibrage de la marge (validation)
python -m src.agents.agents                  démonstration des agents
python -m src.backtest.assistant_bt          backtest de l'assistant complet
python -m src.backtest.compare_selection     versions 1 et 2 de la sélection
python -m src.forecasting                    prévision Ridge et boosting, Diebold-Mariano
python -m src.interpret                      importance par permutation
python -m src.backtest.ml_allocation_bt      allocation avec prévisions de ML
python -m src.backtest.vol_estimators        comparaison des estimateurs de risque
python -m src.backtest.final_results         tests bootstrap, figures 05, tableau final
```

## Lancer l'application

```
streamlit run app/streamlit_app.py
```

Si le dossier `data/processed` n'existe pas, l'application le recrée à partir des données brutes.

## Données

- Prix ajustés (dividendes inclus) de 11 ETF : Yahoo Finance, via la bibliothèque `yfinance`.
- Taux sans risque : taux du Trésor américain à 3 mois, série `TB3MS` de FRED.
- Traitements : prix de fin de mois, rendements simples, exclusion du mois en cours, alignement du taux sans risque.

Les détails figurent dans `data/raw/SOURCES.txt` et dans le rapport.

## Méthodologie en bref

- **Aucune fuite de données.** Le découpage chronologique est fixé avant tout calcul. Le backtest est glissant (walk-forward) : à chaque date de décision, seules les 60 dernières observations disponibles servent à estimer.
- **Allocation.** Répartition égale (1/N), inverse de la volatilité, variance minimale, Sharpe maximal (plafond de 40 % par fonds) et HRP, avec covariance de Ledoit-Wolf et coûts de transaction de 10 points de base.
- **Profils.** La volatilité cible vient de mélanges actions et obligations de référence mesurés sur l'entraînement. Le portefeuille est ramené vers les liquidités pour l'atteindre, avec une marge de sécurité de 0,8 calibrée sur la validation.
- **Prévision.** Ridge (validation croisée temporelle) et gradient boosting comparés à la moyenne historique, avec test de Diebold-Mariano et importance par permutation.
- **Statistique.** Les différences de ratio de Sharpe sont testées par bootstrap par blocs de 6 mois (5 000 tirages).

## Limites

- Univers américain en dollars : le marché marocain et le risque de change ne sont pas modélisés.
- Les résultats dépendent du régime de marché : ce qui gagne sur 2017-2020 ne gagne pas toujours sur 2021-2026.
- Le jeu de test a été consulté plusieurs fois pendant le projet. La version 2 de la sélection et l'étude des estimateurs de risque ont été décidées après avoir vu les résultats du test, et doivent être lues comme telles.
- Avec 48 et 70 mois, un ratio de Sharpe n'est connu qu'à environ plus ou moins 0,5 près.

## Avertissement

Ce projet est un travail universitaire. Il ne constitue pas un conseil en investissement : les performances passées ne préjugent pas des performances futures.