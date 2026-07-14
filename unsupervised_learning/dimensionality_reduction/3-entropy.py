#!/usr/bin/env python3
"""
Module pour calculer l'entropie de Shannon et les affinités P
relatives à un point de données dans t-SNE.
"""
import numpy as np


def HP(Di, beta):
    """
    Calcule l'entropie de Shannon et les affinités P pour un point donné.

    Args:
        Di: np.ndarray de forme (n - 1,) contenant les distances au carré
            entre un point et tous les autres (sauf lui-même).
        beta: np.ndarray de forme (1,) contenant la valeur beta de la
              distribution gaussienne.

    Returns:
        (Hi, Pi)
        Hi: l'entropie de Shannon des points.
        Pi: np.ndarray de forme (n - 1,) contenant les affinités P.
    """
    # 1. Calcul des numérateurs (loi normale avec précision beta)
    numerators = np.exp(-Di * beta)

    # 2. Somme des numérateurs pour la normalisation
    sum_numerators = np.sum(numerators)

    # 3. Calcul des affinités Pi
    Pi = numerators / sum_numerators

    # 4. Calcul de l'entropie de Shannon Hi
    # Utilisation de la définition directe avec un seuil minimal (1e-12)
    # pour empêcher de calculer log2(0) tout en préservant une fidélité totale
    Hi = -np.sum(Pi * np.log2(np.maximum(Pi, 1e-12)))

    return Hi, Pi
