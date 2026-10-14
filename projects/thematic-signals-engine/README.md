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
│       ├── engine.py
│       ├── walk_forward.py
│       └── momentum_walk_forward.py
└── tests/
    ├── test_market_data.py
    ├── test_signals.py
    ├── test_backtest.py
    ├── test_walk_forward.py
    └── test_momentum_walk_forward.py
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

### Contrôle des limites sectorielles

Le module `src/portfolio/sector_risk.py` compare le poids de chaque secteur à une limite configurable (35 % par défaut) et signale les dépassements par date. Il s'agit d'un diagnostic de risque, pas d'un mécanisme automatique de rebalancement. Les poids sont équipondérés et doivent être interprétés comme une hypothèse de recherche simplifiée.

## Backtesting

Le moteur applique les signaux avec un décalage temporel afin d'éviter d'utiliser une information future pour générer un rendement passé.

Les métriques prévues sont :

- rendement cumulé ;
- volatilité annualisée ;
- Sharpe ratio ;
- maximum drawdown ;
- persistance du signal.

### Turnover et coûts de transaction

Le module `src/backtest/trading_costs.py` calcule le turnover unidirectionnel à partir des variations de poids entre dates, puis estime les rendements nets en déduisant un coût paramétrable en points de base. Par exemple, avec 10 bps de coût et 50 % de turnover, le coût estimé est de 5 bps sur la période. Il s'agit d'un modèle simplifié : il n'intègre ni spread bid-ask, ni impact de marché, ni commissions spécifiques à un courtier.

### Analyse brute vs nette

Le backtest expose maintenant `compare_gross_net_performance`, qui applique les coûts de transaction au rendement brut à partir du turnover fourni et compare le rendement cumulé, le Sharpe ratio et le maximum drawdown avant/après coûts. Le différentiel de rendement cumulé est présenté comme un **coût de performance estimé**. Cette analyse reste indicative tant que le turnover n'est pas construit à partir des poids réellement exécutés et que les coûts ne sont pas calibrés sur le marché ciblé.

### Évaluation walk-forward

Le module `src/backtest/walk_forward.py` fournit des fenêtres chronologiques expanding-window : les dates d'entraînement précèdent strictement les dates de test, sans mélange aléatoire. Les tailles sont exprimées en dates uniques, pas en lignes. La fonction `run_walk_forward` appelle un évaluateur fourni par l'utilisateur pour chaque fenêtre et concatène les rendements de test hors échantillon. L'évaluateur doit calibrer les paramètres uniquement sur l'entraînement, puis retourner une série de rendements indexée par les dates du test. Le module vérifie que les dates retournées appartiennent bien à la fenêtre de test et refuse les fenêtres de test qui se chevauchent dans le résultat final. Cette brique ne calibre pas automatiquement les paramètres : la logique de calibration doit être implémentée dans l'évaluateur. Elle ne garantit pas à elle seule l'absence de biais.


#### Calibration walk-forward du momentum

Le module `src/backtest/momentum_walk_forward.py` relie maintenant la validation temporelle à une stratégie long-only simple : il compare plusieurs fenêtres de momentum sur la période d'entraînement selon le Sharpe ratio, retient la meilleure, puis calcule les rendements sur la période de test avec une sélection décalée d'une période. `run_momentum_walk_forward` répète ce processus sur les fenêtres successives et renvoie les rendements hors échantillon concaténés.

Cette implémentation est un prototype pédagogique, pas une stratégie prête à être tradée. Le Sharpe d'entraînement peut être instable sur de petits échantillons ; il faut comparer à un benchmark, intégrer le turnover et les coûts de transaction dans la boucle de sélection, et tester sur des données de marché fiables. Les tests ont été ajoutés mais ne sont pas exécutés automatiquement par cette modification.

## Exemple reproductible

Un exemple de bout en bout génère des prix synthétiques déterministes, calibre une fenêtre de momentum sur des périodes d'entraînement successives, calcule les rendements hors échantillon et les compare à un benchmark équipondéré de l'univers :

```bash
python -m examples.run_synthetic_walk_forward
```

Le script affiche le rendement cumulé hors échantillon, la volatilité annualisée, le Sharpe ratio (taux sans risque nul), le maximum drawdown et la performance excédentaire face au benchmark. Il n'utilise aucune donnée de marché réelle et **n'intègre pas les coûts de transaction** : ses chiffres servent à vérifier le parcours de recherche, pas à revendiquer une performance exploitable.

## Intégration continue

Le workflow GitHub Actions `.github/workflows/thematic-signals-engine-tests.yml` installe les dépendances du projet et exécute `pytest` avec Python 3.12 lors des push et pull requests qui touchent ce projet. Il peut aussi être lancé manuellement depuis l'onglet **Actions**.

Le statut du workflow doit être vérifié dans GitHub Actions : l'ajout de cette configuration ne signifie pas, à lui seul, que la suite de tests a déjà réussi.

## Données

Aucune donnée propriétaire n'est incluse dans le dépôt. Les tests utilisent de petits jeux de données synthétiques afin de rendre le code reproductible.

Le module `src/data/market_data.py` fournit maintenant une première brique d'ingestion depuis un fichier CSV. Il valide le schéma minimal (`date`, `ticker`, `close`), normalise les tickers, convertit les dates et rejette les observations dupliquées ou incohérentes.

Une source de données de marché réelle pourra être connectée ultérieurement.

## SQL

Le dossier `sql/` contient des exemples de requêtes pour filtrer un univers de titres et préparer des séries temporelles de prix.

## Statut

Projet en cours de développement.
