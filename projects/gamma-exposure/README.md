# Gamma Exposure et Gamma Flip

Projet quantitatif consacré à l'analyse de l'exposition gamma d'un portefeuille d'options.

## Objectif

Construire un outil permettant de calculer la gamma d'options individuelles, d'agréger l'exposition par strike et d'identifier un niveau de Gamma Flip dans une chaîne d'options synthétique.

Le projet travaille sur :

modèle de Black-Scholes  
calcul des Greeks  
structure d'une chaîne d'options  
agrégation de l'Open Interest  
Gamma Exposure par strike  
recherche d'un changement de signe de l'exposition gamma

## Méthodologie

La gamma de chaque option est calculée avec le modèle de Black-Scholes.

L'exposition agrégée est ensuite calculée à partir de la gamma, de l'Open Interest et d'un facteur de taille de contrat.

Le Gamma Flip correspond ici au niveau de prix où l'exposition gamma agrégée change de signe.

Les données utilisées dans l'exemple sont synthétiques. Le projet ne prétend donc pas mesurer le positionnement réel des dealers sur un marché donné.

## Structure

```text
gamma-exposure/
README.md
requirements.txt
src/
  gamma_exposure.py
tests/
  test_gamma_exposure.py
```

## Utilisation

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer l'exemple :

```bash
python src/gamma_exposure.py
```

Lancer les tests :

```bash
python -m unittest discover tests
```

## Limites

Le modèle repose sur Black-Scholes et sur une chaîne d'options synthétique.

Une application de marché réelle nécessiterait notamment des données fiables sur les strikes, maturités, volatilités implicites, Open Interest, multiplicateurs de contrats et conventions de marché.

Le signe de l'exposition dealer ne peut pas être déduit de l'Open Interest seul sans hypothèses supplémentaires sur la répartition des positions.

## Suite

Les prochaines versions pourront intégrer des données de marché réelles, plusieurs maturités, une visualisation de l'exposition par strike et une analyse plus détaillée de la sensibilité au spot.
