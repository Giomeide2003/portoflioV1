# Thematic Signals Engine

Projet de recherche quantitative consacré à la construction de signaux thématiques multi-facteurs et à leur évaluation dans un cadre de portefeuille.

## Objectif

Construire progressivement un pipeline Python permettant de :

- charger et normaliser des données de marché ;
- construire des métriques thématiques et multi-facteurs ;
- calculer des signaux de momentum, d'exposition sectorielle et de valorisation relative ;
- tester les signaux avec un backtest vectorisé ;
- mesurer rendement, volatilité, Sharpe ratio, drawdown et persistance des signaux ;
- préparer des jeux de données exploitables avec SQL.

Le projet est développé par étapes afin de documenter les hypothèses, les choix de modélisation et les limites méthodologiques.

## Architecture

```text
thematic-signals-engine/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── sql/
│   └── universe_screen.sql
├── src/
│   ├── data/
│   │   └── market_data.py
│   ├── signals/
│   │   └── thematic_signals.py
│   └── backtest/
│       └── engine.py
└── tests/
    ├── test_market_data.py
    ├── test_signals.py
    └── test_backtest.py
```

## Signaux initiaux

### Momentum thématique

Le momentum mesure la performance passée d'un actif ou d'un panier sur une fenêtre donnée.

### Exposition sectorielle

L'exposition sectorielle permet d'agréger les positions ou les poids par secteur afin d'identifier la concentration d'un portefeuille.

### Valorisation relative

Le module est prévu pour comparer des métriques de valorisation entre titres d'un même univers, par exemple un ratio de valorisation normalisé par rapport à son secteur.

## Analyse sectorielle

Le module `src/portfolio/sector_exposure.py` complète la construction de portefeuille avec une analyse d'exposition sectorielle. Pour chaque date, il calcule :

- le poids de chaque secteur dans les positions sélectionnées ;
- le poids du secteur dominant ;
- le **Herfindahl-Hirschman Index (HHI)** comme mesure simple de concentration.

Les poids sont calculés en équipondération des titres sélectionnés. Cette convention est volontairement simple et sert à isoler l'effet de la sélection des titres avant d'introduire une optimisation de portefeuille plus avancée.

## Backtesting

Le moteur applique les signaux avec un décalage temporel afin d'éviter d'utiliser une information future pour générer un rendement passé.

Les métriques prévues sont :

- rendement cumulé ;
- volatilité annualisée ;
- Sharpe ratio ;
- maximum drawdown ;
- persistance du signal.

## Données

Aucune donnée propriétaire n'est incluse dans le dépôt. Les tests utilisent de petits jeux de données synthétiques afin de rendre le code reproductible.

Le module `src/data/market_data.py` fournit maintenant une première brique d'ingestion depuis un fichier CSV. Il valide le schéma minimal (`date`, `ticker`, `close`), normalise les tickers, convertit les dates et rejette les observations dupliquées ou incohérentes.

Une source de données de marché réelle pourra être connectée ultérieurement.

## SQL

Le dossier `sql/` contient des exemples de requêtes pour filtrer un univers de titres et préparer des séries temporelles de prix.

## Statut

Projet en cours de développement.
