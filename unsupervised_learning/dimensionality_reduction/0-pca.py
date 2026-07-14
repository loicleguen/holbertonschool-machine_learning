#!/usr/bin/env python3
"""
Module pour effectuer la réduction de dimensionnalité via la PCA
"""
import numpy as np


def pca(X, var=0.95):
    """
    Effectue la PCA sur un jeu de données.

    Args:
        X: np.ndarray de forme (n, d) contenant les données centrées.
            n est le nombre de points de données.
            d est le nombre de dimensions d'origine.
        var: la fraction de variance que la transformation doit maintenir.

    Returns:
        W: np.ndarray de forme (d, nd) contenant la matrice de poids,
           où nd est la nouvelle dimensionnalité.
    """
    # Application de la SVD sur la matrice X
    # X = U * S * Vh
    # vh contient les vecteurs propres de la matrice de covariance (lignes)
    _, s, vh = np.linalg.svd(X, full_matrices=False)

    # Calcul de la variance expliquée par chaque composante
    squared_singular_values = s ** 2
    explained_variance = (
        squared_singular_values / np.sum(squared_singular_values))

    # Somme cumulée pour atteindre le seuil de variance demandé
    cumulative_variance = np.cumsum(explained_variance)

    # Recherche du nombre de dimensions minimal requis
    # np.argmax renvoie le premier index qui valide la condition
    nd = np.argmax(cumulative_variance >= var) + 1

    # Sélection des 'nd' premières composantes principales (les colonnes de W)
    # On transpose vh pour obtenir les vecteurs en colonnes
    W = vh[:nd].T

    return W
