# Assistant multi-agent de sélection de portefeuilles

Projet de Machine Learning, ECC (sujet 4). Équipe : à compléter.

Un investisseur renseigne son capital, sa durée, son objectif et son profil de risque. Sept agents spécialisés
(données, profil, analyse, optimisation, risque, sélection, synthèse) étudient les marchés, comparent plusieurs
répartitions, contrôlent le risque et proposent une allocation expliquée, avec la part gardée en liquidités.

## Résultats principaux

Univers : 11 ETF cotés aux États-Unis (actions, obligations, or, immobilier, matières premières) et taux sans
risque. Données mensuelles de janvier 2007 à octobre 2026 (238 mois). Découpage chronologique : entraînement
2007-2016, validation 2017-2020, test 2021 à octobre 2026.

| Question | Résultat |
|---|---|
| Les modèles de prévision (Ridge, boosting) battent-ils la moyenne historique ? | Non. Ridge est indiscernable de la baseline (Diebold-Mariano p = 0,53 en validation, 0,32 en test). Le boosting est significativement moins bon (p = 0,003 et 0,015). |
| Une méthode d'allocation domine-t-elle ? | Non. Sur 18 comparaisons de Sharpe par bootstrap, 2 sont significatives à 5 %, ce qui est proche de ce que le hasard produit. |
| Que apporte l'assistant ? | Le ciblage du risque par profil avec liquidités. Profil Prudent sur le test : volatilité 3,9 % et drawdown -9,0 %, contre 7,2 % et -16,3 % pour un mélange classique 20/80 actions-obligations. |
| Peut-on mieux estimer le risque ? | Non. Les estimateurs EWMA n'améliorent pas l'estimateur sur 60 mois. La marge de sécurité de 0,8 sur la volatilité cible est stable d'un estimateur à l'autre. |

Ces résultats, y compris négatifs, sont détaillés dans le rapport.

## Structure du dépôt

~~~
app/streamlit_app.py        application web
src/config.py               univers d'actifs, dates, paramètres
src/download_data.py        téléchargement (yfinance, FRED)
src/prepare_data.py         rendements mensuels, découpage chronologique
src/eda.py                  analyse exploratoire (train uniquement)
src/metrics.py              métriques de performance
src/optimization/           méthodes d'allocation, estimateurs de risque
src/profiles.py             profils d'investisseur, ciblage de volatilité
src/agents/agents.py        les sept agents et l'orchestrateur
src/forecasting.py          prévision des rendements (Ridge, boosting)
src/interpret.py            interprétabilité (permutation)
src/backtest/               moteur walk-forward, calibrages, résultats finaux
data/raw/                   données brutes téléchargées (versionnées)
reports/                    figures et tableau des résultats finaux
~~~

## Installation

~~~
python -m venv .venv
.venv\Scripts\Activate.ps1        (Windows)   |   source .venv/bin/activate   (Mac et Linux)
pip install -r requirements.txt
~~~

## Reproduire les résultats

Les données brutes sont déjà dans `data/raw`. Pour les retélécharger, lancer la première commande
(les prix ajustés de Yahoo Finance peuvent légèrement évoluer avec le temps).

~~~
python -m src.download_data                  téléchargement
python -m src.prepare_data                   rendements mensuels et découpage
python -m src.eda                            analyse exploratoire, figures 01 à 04
python -m src.optimization.allocators        poids des méthodes d'allocation
python -m src.backtest.engine                backtest walk-forward des méthodes
python -m src.backtest.profiles_bt           backtest par profil
python -m src.backtest.calibrate_buffer      calibrage de la marge (validation)
python -m src.agents.agents                  démonstration des agents
python -m src.backtest.assistant_bt          backtest de l'assistant complet
python -m src.backtest.compare_selection     versions v1 et v2 de la sélection
python -m src.forecasting                    prévision Ridge et boosting, Diebold-Mariano
python -m src.interpret                      importance par permutation
python -m src.backtest.ml_allocation_bt      allocation avec prévisions ML
python -m src.backtest.vol_estimators        comparaison des estimateurs de risque
python -m src.backtest.final_results         tests bootstrap, figures 05, tableau final
~~~

## Lancer l'application

~~~
streamlit run app/streamlit_app.py
~~~

Si `data/processed` n'existe pas, l'application le recrée à partir des données brutes.

## Données

- Prix ajustés (dividendes inclus) de 11 ETF : Yahoo Finance via la bibliothèque yfinance.
- Taux sans risque : taux du Trésor américain à 3 mois, série TB3MS de FRED.
- Traitements : fin de mois, rendements simples, exclusion du mois en cours, alignement du taux sans risque.
Détails dans `data/raw/SOURCES.txt` et dans le rapport.

## Méthodologie en bref

- Aucune fuite de données : découpage chronologique avant tout calcul, backtest walk-forward, estimation sur
  les 60 derniers mois disponibles à chaque date de décision.
- Allocation : 1/N, inverse de la volatilité, minimum de variance, Markowitz (max Sharpe, plafond de 40 %), HRP,
  avec covariance de Ledoit-Wolf et coûts de transaction de 10 points de base.
- Profils : volatilité cible issue de mélanges de référence mesurés sur le train, réduction vers les liquidités,
  marge de sécurité de 0,8 calibrée sur la validation.
- Prévision : Ridge (validation croisée temporelle) et boosting contre la moyenne historique, test de
  Diebold-Mariano, importance par permutation.

## Limites

- Univers américain en dollars : le marché marocain et l'effet de change ne sont pas modélisés.
- Les résultats dépendent du régime de marché : ce qui gagne sur 2017-2020 ne gagne pas toujours sur 2021-2026.
- Le test a été consulté plusieurs fois pendant le projet. La version 2 de la sélection et l'étude des estimateurs
  de risque ont été décidées après avoir vu les résultats du test et doivent être lues comme telles.
- Avec 48 et 70 mois, un Sharpe est connu à environ plus ou moins 0,5.
