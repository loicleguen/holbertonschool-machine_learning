#!/usr/bin/env python3
"""
Module pour calculer la matrice de corrélation
à partir d'une matrice de covariance.
"""
import numpy as np


def correlation(C):
    """
    Calcule la matrice de corrélation à partir d'une matrice de covariance.

    Parameters:
    C (numpy.ndarray) : Une matrice carrée de dimensions (d, d)
                        contenant la matrice de covariance.

    Returns:
    numpy.ndarray : Une matrice de dimensions (d, d) contenant
                    la matrice de corrélation.
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # On extrait les variances situées sur la diagonale principale (C_ii)
    diag = np.diag(C)

    # On calcule les écarts-types (racine carrée des variances)
    # On utilise np.sqrt en s'assurant de ne pas avoir de valeurs négatives
    std_dev = np.sqrt(diag)

    # Pour faire l'opération de manière vectorielle efficace :
    # std_dev[None, :] donne une ligne (1, d)
    # std_dev[:, None] donne une colonne (d, 1)
    # Le produit extérieur donne une matrice(d, d)
    # contenant tous les (σ_i * σ_j)
    outer_std = std_dev[:, None] * std_dev[None, :]

    # La matrice de corrélation est obtenue en
    # divisant C par ce produit extérieur
    corr = C / outer_std

    return corr
