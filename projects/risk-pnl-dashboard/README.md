# P&L et Risk Dashboard

Projet quantitatif consacré au suivi du P&L et à l'analyse du risque d'un portefeuille multi-actifs.

## Objectif

Construire un moteur simple permettant de :

calculer le P&L d'un portefeuille  
agréger les expositions par actif  
estimer une VaR historique  
calculer une VaR paramétrique  
effectuer des stress tests sur les facteurs de risque  
mesurer l'impact de scénarios de marché

Le projet privilégie un moteur de calcul simple et testable avant l'ajout d'une interface graphique.

## Méthodologie

Le portefeuille est représenté par des positions et des prix.

La VaR historique utilise la distribution empirique des rendements historiques du portefeuille.

La VaR paramétrique suppose une distribution normale des rendements et utilise la volatilité historique du portefeuille.

Les stress tests appliquent des chocs définis sur les prix des actifs afin d'étudier la sensibilité du portefeuille à différents scénarios.

Les données de marché utilisées dans l'exemple sont synthétiques.

## Métriques

Le projet calcule notamment :

P&L total  
P&L par position  
VaR historique à 95 %  
VaR paramétrique à 95 %  
volatilité annualisée  
stress P&L

## Structure

```text
risk-pnl-dashboard/
README.md
requirements.txt
src/
  risk_engine.py
tests/
  test_risk_engine.py
```

## Utilisation

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer l'exemple :

```bash
python src/risk_engine.py
```

Lancer les tests :

```bash
python -m unittest discover tests
```

## Limites

Ce projet est un outil pédagogique. Les méthodes de VaR dépendent fortement des données, de l'horizon et des hypothèses statistiques retenues.

La VaR ne mesure pas directement les pertes au-delà du quantile choisi et ne constitue pas une mesure exhaustive du risque.

Une utilisation sur un desk nécessiterait notamment des données de marché fiables, des facteurs de risque plus complets, des scénarios historiques pertinents et des contrôles indépendants.

## Suite

Une extension naturelle consiste à ajouter une interface Streamlit, des séries historiques réelles, des stress tests multi-facteurs et une décomposition plus détaillée du P&L.
