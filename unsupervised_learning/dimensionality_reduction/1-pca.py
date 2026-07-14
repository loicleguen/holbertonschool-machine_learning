#!/usr/bin/env python3
"""
Module pour projeter un jeu de données sur un nombre défini de dimensions (PCA)
"""
import numpy as np


def pca(X, ndim):
    """
    Effectue la PCA sur un jeu de données
    pour le projeter dans un nouvel espace.

    Args:
        X: np.ndarray de forme (n, d) à transformer.
            n est le nombre de points de données.
            d est le nombre de dimensions d'origine.
        ndim: la nouvelle dimensionnalité du jeu de données transformé.

    Returns:
        T: np.ndarray de forme (n, ndim) contenant les données projetées.
    """
    # 1. Centrer les données en soustrayant la moyenne de chaque colonne
    X_mean = X - np.mean(X, axis=0)

    # 2. Calculer la SVD sur les données centrées
    _, _, vh = np.linalg.svd(X_mean)

    # 3. Sélectionner les 'ndim' premières composantes principales
    # vh contient les vecteurs propres en lignes, on prend les 'ndim' premières
    # et on transpose pour obtenir une matrice de poids W de forme (d, ndim)
    W = vh[:ndim].T

    # 4. Projeter les données centrées sur le nouvel espace
    T = np.matmul(X_mean, W)

    return T
