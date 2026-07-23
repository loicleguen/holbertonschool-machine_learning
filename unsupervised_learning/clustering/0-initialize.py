#!/usr/bin/env python3
"""
Module providing a function to initialize cluster centroids for K-means.
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d).
    - k (int): Number of clusters (positive integer).

    Returns:
    - numpy.ndarray: Initialized centroids of shape (k, d), or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    return np.random.uniform(low, high, size=(k, X.shape[1]))
