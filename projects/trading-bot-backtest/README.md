# Trading Bot et Backtesting

Projet personnel consacré à l'évaluation d'une stratégie de trading systématique sur données historiques.

## Objectif

Construire un petit moteur de backtesting permettant de comparer une stratégie à une approche passive et de mesurer son comportement en termes de rendement et de risque.

Le projet sert surtout à travailler sur la chaîne complète :

collecte des données  
préparation des séries temporelles  
génération des signaux  
simulation des positions  
prise en compte des coûts de transaction  
calcul des métriques  
analyse des résultats

## Stratégie

La première version utilise une stratégie de croisement de moyennes mobiles.

Une position longue est ouverte lorsque la moyenne mobile courte passe au-dessus de la moyenne mobile longue. Elle est fermée lorsque le signal s'inverse.

Les paramètres sont volontairement simples afin de garder un modèle facile à analyser et à améliorer.

## Métriques

Le backtest calcule notamment :

Rendement cumulé  
Volatilité annualisée  
Ratio de Sharpe  
Maximum drawdown  
Taux de réussite  
Nombre de transactions

Les frais de transaction sont intégrés dans la simulation afin d'éviter une estimation trop optimiste des résultats.

## Structure

```text
trading-bot-backtest/
README.md
requirements.txt
src/
  backtest.py
tests/
  test_backtest.py
```

## Utilisation

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer le backtest :

```bash
python src/backtest.py
```

Lancer les tests :

```bash
python -m unittest discover tests
```

## Limites

Ce projet est un outil pédagogique. Les résultats historiques ne constituent pas une garantie de performance future.

La stratégie ne prend pas encore en compte l'ensemble des contraintes rencontrées sur un desk : slippage variable, liquidité, impact de marché, contraintes de position ou données intraday.

## Suite

Les prochaines versions pourront intégrer des données réelles, plusieurs stratégies, une gestion du risque plus complète et une analyse plus détaillée des coûts d'exécution.
