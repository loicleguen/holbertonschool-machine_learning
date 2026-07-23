#!/usr/bin/env python3
"""
Module providing a function to calculate the PDF of a Gaussian distribution.
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Parameters:
    - X (numpy.ndarray): Data points of shape (n, d).
    - m (numpy.ndarray): Mean of the distribution of shape (d,).
    - S (numpy.ndarray): Covariance matrix of the distribution of shape (d, d).

    Returns:
    - P (numpy.ndarray): PDF values for each data point of shape (n,),
      or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    det = np.linalg.det(S)
    if det <= 0:
        return None

    inv = np.linalg.inv(S)

    # Difference entre chaque point et la moyenne (n, d)
    diff = X - m

    # Forme quadratique: (x - m) @ inv(S) * (x - m) sommer sur les colonnes
    quad = np.sum((diff @ inv) * diff, axis=1)

    # Normalisation
    norm = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)

    P = norm * np.exp(-0.5 * quad)

    # Valeur minimale 1e-300
    return np.maximum(P, 1e-300)
