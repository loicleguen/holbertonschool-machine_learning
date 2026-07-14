#!/usr/bin/env python3
"""
Module pour calculer les affinités Q dans l'espace de faible dimension (t-SNE)
"""
import numpy as np


def Q_affinities(Y):
    """
    Calcule les affinités Q et leurs numérateurs pour un jeu de données.

    Args:
        Y: np.ndarray de forme (n, ndim) contenant la transformation
           en basse dimension de X.

    Returns:
        Q: np.ndarray de forme (n, n) contenant les affinités Q.
        num: np.ndarray de forme (n, n) contenant
        le numérateur des affinités Q.
    """
    # 1. Calcul des distances euclidiennes au carré par paires dans l'espace Y
    sum_Y = np.sum(np.square(Y), axis=1, keepdims=True)
    D = sum_Y - 2 * np.dot(Y, Y.T) + sum_Y.T

    # 2. Calcul du numérateur de la distribution t-Student (1 / (1 + d^2))
    num = 1.0 / (1.0 + D)

    # La diagonale doit être explicitement fixée à 0 (i == j)
    np.fill_diagonal(num, 0)

    # 3. Normalisation pour obtenir la matrice d'affinités Q
    Q = num / np.sum(num)

    return Q, num
