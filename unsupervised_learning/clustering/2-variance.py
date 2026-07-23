#!/usr/bin/env python3
"""
Module providing a function to calculate the total intra-cluster variance.
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d).
    - C (numpy.ndarray): Centroid means for each cluster of shape (k, d).

    Returns:
    - total_variance (float): The total intra-cluster variance,
      or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Distances au carré entre chaque point et chaque centroïde : shape (n, k)
    distances_squared = np.sum((X[:, None, :] - C) ** 2, axis=-1)

    # Prendre la distance au carré minimale pour chaque point
    min_distances_squared = np.min(distances_squared, axis=1)

    # Somme globale de toutes les distances au carré
    return np.sum(min_distances_squared)
