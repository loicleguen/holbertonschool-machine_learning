#!/usr/bin/env python3
"""
Module providing a function to initialize variables for a GMM.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d).
    - k (int): Number of clusters (positive integer).

    Returns:
    - pi (numpy.ndarray): Priors for each cluster of shape (k,).
    - m (numpy.ndarray): Centroid means of shape (k, d).
    - S (numpy.ndarray): Covariance matrices of shape (k, d, d).
    - Or (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    pi = np.full((k,), 1 / k)
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
