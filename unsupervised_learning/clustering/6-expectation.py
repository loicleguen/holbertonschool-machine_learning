#!/usr/bin/env python3
"""
Module providing a function to calculate the expectation step in EM for GMM.
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d).
    - pi (numpy.ndarray): Priors for each cluster of shape (k,).
    - m (numpy.ndarray): Centroid means for each cluster of shape (k, d).
    - S (numpy.ndarray): Covariance matrices
        for each cluster of shape (k, d, d).

    Returns:
    - g (numpy.ndarray): Posterior probabilities of shape (k, n).
    - l (float): Total log likelihood.
    - Or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    # Matrice pour stocker la densite ponderee par les a priori: shape (k, n)
    weighted_pdf = np.zeros((k, n))

    # Au maximum 1 boucle autorisee pour parcourir les k clusters
    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        weighted_pdf[i] = pi[i] * P

    # Somme sur tous les clusters pour chaque point n: shape (n,)
    marginal_pdf = np.sum(weighted_pdf, axis=0)

    # Calcul des probabilites a posteriori g: shape (k, n)
    g = weighted_pdf / marginal_pdf

    # Vraisemblance logarithmique totale
    #   (somme des log des vraisemblances marginales)
    log_like = np.sum(np.log(marginal_pdf))

    return g, log_like
