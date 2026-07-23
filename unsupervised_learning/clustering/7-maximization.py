#!/usr/bin/env python3
"""
Module providing a function to calculate the maximization step in EM for GMM.
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d).
    - g (numpy.ndarray): Posterior probabilities of shape (k, n).

    Returns:
    - pi (numpy.ndarray): Updated priors of shape (k,).
    - m (numpy.ndarray): Updated centroid means of shape (k, d).
    - S (numpy.ndarray): Updated covariance matrices of shape (k, d, d).
    - Or (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    # Verification que la somme des probabilites
    #   a posteriori pour chaque point est 1
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    # Somme des probabilites a posteriori pour
    #   chaque cluster : N_i de forme (k,)
    N_i = np.sum(g, axis=1)

    # 1. Nouveaux priors (pi)
    pi = N_i / n

    # 2. Nouvelles moyennes (m)
    m = np.matmul(g, X) / N_i[:, None]

    # 3. Nouvelles matrices de covariance (S)
    S = np.zeros((k, d, d))

    # Au maximum 1 boucle autorisee pour les k clusters
    for i in range(k):
        diff = X - m[i]  # shape (n, d)
        # Multiplication ponderee : (d, n) x (n, d) -> (d, d)
        S[i] = np.matmul(g[i] * diff.T, diff) / N_i[i]

    return pi, m, S
