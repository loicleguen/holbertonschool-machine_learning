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
    # 1. Calcul des numérateurs (exposants de la Gaussienne)
    # On multiplie les distances par -beta
    numerators = np.exp(-Di * beta)

    # 2. Somme des numérateurs pour la normalisation
    sum_numerators = np.sum(numerators)

    # 3. Calcul des affinités Pi (normalisation)
    Pi = numerators / sum_numerators

    # 4. Calcul de l'entropie de Shannon Hi
    # Formule directe simplifiée et robuste pour éviter log2(0):
    # H = -sum(P * log2(P))
    # Ce qui équivaut algébriquement à :
    #   H = (beta * sum(D * P) + log(sum_numerators)) / log(2)
    # (cf. page 4 de l'article t-SNE)
    Hi = (beta * np.sum(Di * Pi) + np.log(sum_numerators)) / np.log(2)

    # Si tu as besoin d'une implémentation plus classique mais protégée :
    # Hi = -np.sum(Pi * np.log2(np.maximum(Pi, 1e-12)))

    return Hi, Pi
