#!/usr/bin/env python3
"""
Module providing a function to perform K-means clustering.
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d).
    - k (int): Number of clusters (positive integer).
    - iterations (int): Maximum number of iterations (positive integer).

    Returns:
    - C (numpy.ndarray): Centroid means of shape (k, d).
    - clss (numpy.ndarray): Index of cluster for each point of shape (n,).
    - Or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    # 1er appel a np.random.uniform
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        C_prev = np.copy(C)

        # Calcul des distances et attribution
        #   des points au cluster le plus proche
        distances = np.linalg.norm(X[:, None, :] - C, axis=-1)
        clss = np.argmin(distances, axis=1)

        # Mise a jour des centroides (2eme boucle autorisee)
        for i in range(k):
            points = X[clss == i]
            if len(points) == 0:
                # 2eme appel a np.random.uniform si un cluster est vide
                C[i] = np.random.uniform(low, high, size=(1, d))
            else:
                C[i] = np.mean(points, axis=0)

        # Recalcul de l'attribution avec les centroides mis a jour
        distances = np.linalg.norm(X[:, None, :] - C, axis=-1)
        clss = np.argmin(distances, axis=1)

        # Condition de convergence
        if np.array_equal(C_prev, C):
            break

    return C, clss
