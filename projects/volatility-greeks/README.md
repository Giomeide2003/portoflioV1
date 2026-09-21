# Volatilité Implicite et Greeks

Projet quantitatif consacré à l'analyse de la volatilité implicite et des principaux Greeks d'une option.

## Objectif

Construire un petit moteur permettant de :

calculer le prix Black-Scholes d'une option  
inverser le modèle pour estimer la volatilité implicite  
calculer Delta, Gamma, Vega et Theta  
étudier la sensibilité du prix aux paramètres du modèle  
comparer l'effet de différentes volatilités implicites

Les données de marché utilisées dans l'exemple sont synthétiques afin de rendre les résultats reproductibles.

## Méthodologie

Le prix théorique d'une option européenne est calculé avec Black-Scholes.

La volatilité implicite est ensuite obtenue par résolution numérique : on recherche la volatilité qui permet au modèle de retrouver le prix observé.

La méthode de Newton-Raphson est utilisée lorsque la Vega permet une mise à jour suffisamment stable, avec une borne de sécurité sur la volatilité. Une recherche par dichotomie fournit une solution robuste lorsque nécessaire.

## Greeks

Le projet calcule :

Delta : sensibilité du prix au spot  
Gamma : sensibilité du Delta au spot  
Vega : sensibilité à la volatilité  
Theta : sensibilité à l'écoulement du temps

## Structure

```text
volatility-greeks/
README.md
requirements.txt
src/
  volatility_greeks.py
tests/
  test_volatility_greeks.py
```

## Utilisation

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer l'exemple :

```bash
python src/volatility_greeks.py
```

Lancer les tests :

```bash
python -m unittest discover tests
```

## Limites

Black-Scholes repose sur des hypothèses simplificatrices : volatilité constante, taux constant, absence de coûts de transaction et exercice européen.

La volatilité implicite calculée dépend également de la qualité du prix d'option utilisé comme entrée.

## Suite

Une extension naturelle consiste à construire une surface de volatilité implicite par strike et maturité, puis à analyser le smile et la term structure.
