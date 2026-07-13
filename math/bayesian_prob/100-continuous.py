#!/usr/bin/env python3
"""
Ce module contient la fonction posterior pour un espace de probabilité continu,
en utilisant la distribution Beta conjuguée à une loi binomiale.
"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calcule la probabilité a posteriori que la probabilité de développer
    des effets secondaires graves se situe dans l'intervalle [p1, p2].

    Paramètres :
    ------------
    x : int
        Le nombre de patients développant des effets secondaires graves.
    n : int
        Le nombre total de patients observés.
    p1 : float
        La borne inférieure de l'intervalle.
    p2 : float
        La borne supérieure de l'intervalle.

    Retourne :
    ----------
    float
        La probabilité a posteriori que p soit dans [p1, p2].
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")

    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Paramètres de la loi Beta a posteriori (Prior Uniforme = Beta(1,1))
    a = x + 1
    b = n - x + 1

    # Calcul des CDFs en utilisant la fonction beta incomplète régularisée
    cdf_p2 = special.betainc(a, b, p2)
    cdf_p1 = special.betainc(a, b, p1)

    return cdf_p2 - cdf_p1
