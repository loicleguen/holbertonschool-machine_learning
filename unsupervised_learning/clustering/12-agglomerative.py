#!/usr/bin/env python3
"""
Module providing a function to perform agglomerative clustering on a dataset.
"""
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset with Ward linkage.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d).
    - dist (float/int): Maximum cophenetic distance for all clusters.

    Returns:
    - clss (numpy.ndarray): Cluster indices for each data point of shape (n,).
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    return clss
