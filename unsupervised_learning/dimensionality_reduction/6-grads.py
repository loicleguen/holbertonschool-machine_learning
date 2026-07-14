#!/usr/bin/env python3
"""
Module pour calculer les gradients de Y dans l'algorithme t-SNE
"""
import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calcule les gradients de Y et les affinités Q.

    Args:
        Y: np.ndarray de forme (n, ndim) contenant la transformation
           en basse dimension de X.
        P: np.ndarray de forme (n, n) contenant les affinités symétriques P.

    Returns:
        dY: np.ndarray de forme (n, ndim) contenant les gradients de Y.
        Q: np.ndarray de forme (n, n) contenant les affinités Q de Y.
    """
    n, ndim = Y.shape

    # 1. Calcul des affinités Q et de leur numérateur
    Q, num = Q_affinities(Y)

    # 2. Initialisation de la matrice de gradients dY
    dY = np.zeros((n, ndim))

    # 3. Vectorisation et calcul du gradient
    # Le produit scalaire pondéré (P - Q) * num
    PQ_diff = (P - Q) * num

    for i in range(n):
        # Différence des coordonnées spatiales : (y_i - y_j)
        # Y[i] - Y crée un broadcast de forme (n, ndim)
        y_diff = Y[i] - Y

        # Multiplication de la différence par le
        #       coefficient (P_ij - Q_ij) * num_ij
        # On ajoute une dimension à PQ_diff[i] pour
        #       faire un broadcast sur les dimensions (ndim)
        # Somme sur toutes les cibles j (axe 0)
        dY[i] = np.sum(PQ_diff[i][:, np.newaxis] * y_diff, axis=0)

    return dY, Q
