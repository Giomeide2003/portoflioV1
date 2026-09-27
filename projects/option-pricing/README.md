# Option Pricing — Black-Scholes

Projet personnel d’apprentissage consacré au pricing d’options européennes avec le modèle de Black-Scholes.

## Objectif

Construire progressivement un petit pricer de dérivés afin de comprendre les mécanismes utilisés en pricing et en gestion du risque :

- prix d’un Call et d’un Put ;
- calcul des Greeks ;
- volatilité implicite ;
- tests numériques ;
- organisation d’un code Python simple et réutilisable.

## Modèle

Pour une option européenne sans dividende, le modèle de Black-Scholes utilise notamment :

- `S` : prix du sous-jacent ;
- `K` : strike ;
- `T` : temps restant jusqu’à maturité, en années ;
- `r` : taux sans risque ;
- `σ` : volatilité.

Les termes principaux sont :

`d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)`

`d2 = d1 - σ√T`

Pour un Call :

`C = S N(d1) - K exp(-rT) N(d2)`

Pour un Put :

`P = K exp(-rT) N(-d2) - S N(-d1)`

## Greeks

Le projet calcule actuellement :

- **Delta** : sensibilité du prix à une variation du sous-jacent ;
- **Gamma** : sensibilité du Delta au sous-jacent ;
- **Vega** : sensibilité du prix à la volatilité ;
- **Theta** : sensibilité du prix au passage du temps.

Les conventions utilisées dans le code sont documentées directement dans les fonctions.

## Volatilité implicite

La volatilité implicite est obtenue par résolution numérique : le solveur cherche la valeur de `σ` qui permet au prix Black-Scholes de reproduire un prix de marché donné.

La méthode actuelle utilise **Newton-Raphson**, avec Vega comme dérivée du prix par rapport à la volatilité.

## Exemple

Un exemple minimal est disponible dans `examples/basic_pricer.py`.

Pour le cas :

- `S = 100`
- `K = 100`
- `T = 1 an`
- `r = 5%`
- `σ = 20%`

le modèle donne approximativement :

- Call : `10.4506`
- Delta Call : `0.6368`
- Gamma : `0.0188`
- Vega : `37.5240`

Le solveur de volatilité implicite retrouve ensuite une volatilité proche de `20%` à partir du prix du Call.

## Tests

Les tests couvrent actuellement :

- prix de Call et Put sur des valeurs de référence ;
- put-call parity ;
- validation des paramètres ;
- Greeks ;
- convergence de la volatilité implicite.

## Structure

```text
option-pricing/
├── README.md
├── examples/
│   └── basic_pricer.py
├── src/
│   ├── black_scholes.py
│   ├── greeks.py
│   └── implied_volatility.py
└── tests/
    ├── test_black_scholes.py
    ├── test_greeks.py
    └── test_implied_volatility.py
```

## Statut

Projet en cours de développement.
