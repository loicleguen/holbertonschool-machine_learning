#!/usr/bin/env python3
"""
Ce module contient la fonction likelihood qui calcule la vraisemblance
d'obtenir des données observées selon une distribution binomiale.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calcule la vraisemblance d'obtenir les données x et n pour chaque
    probabilité hypothétique contenue dans P.

    Paramètres :
    ------------
    x : int
        Le nombre de patients développant des effets secondaires graves.
    n : int
        Le nombre total de patients observés.
    P : np.ndarray
        Un tableau 1D contenant les différentes probabilités hypothétiques.

    Retourne :
    ----------
    np.ndarray
        Un tableau 1D contenant la vraisemblance pour chaque probabilité de P.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, np.integer)) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calcul du coefficient binomial : n! / (x! * (n - x)!)
    fact = np.math.factorial
    comp = fact(n) / (fact(x) * fact(n - x))

    # Calcul de la vraisemblance binomiale pour chaque p dans P
    return comp * (P ** x) * ((1 - P) ** (n - x))


def intersection(x, n, P, Pr):
    """
    Calcule l'intersection d'obtenir les données x et n avec chaque
    probabilité de P selon les croyances a priori Pr.

    Paramètres :
    ------------
    x : int
        Le nombre de patients développant des effets secondaires graves.
    n : int
        Le nombre total de patients observés.
    P : np.ndarray
        Un tableau 1D contenant les différentes probabilités hypothétiques.
    Pr : np.ndarray
        Un tableau 1D contenant les probabilités a priori de P.

    Retourne :
    ----------
    np.ndarray
        Un tableau 1D contenant l'intersection pour chaque probabilité.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, np.integer)) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Réutilisation directe de la fonction likelihood définie plus haut
    l_vals = likelihood(x, n, P)

    # L'intersection est le produit élément par
    # élément de la vraisemblance et du prior
    return l_vals * Pr
