#!/usr/bin/env python3
"""
Module pour calculer la moyenne et la matrice de covariance d'un dataset.
"""
import numpy as np


def mean_cov(X):
    """
    Calcule la moyenne et la matrice de covariance d'un ensemble de données.

    Parameters:
    X (numpy.ndarray) : Un tableau de dimensions (n, d) contenant le dataset.
        - n est le nombre de points de données.
        - d est le nombre de dimensions de chaque point.

    Returns:
    mean, cov :
        - mean : numpy.ndarray de dimension (1, d) contenant les moyennes.
        - cov : numpy.ndarray de dimension (d, d) contenant la matrice de cov.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calcul de la moyenne le long de l'axe des lignes (axis=0)
    # keepdims=True permet de garder la forme (1, d) au lieu de (d,)
    mean = np.mean(X, axis=0, keepdims=True)

    # Centrage des données (X - moyenne)
    X_centered = X - mean

    # Formule mathématique de la covariance
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
