#!/usr/bin/env python3
"""
Module providing a function to perform K-means using scikit-learn.
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset using scikit-learn.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d).
    - k (int): Number of clusters.

    Returns:
    - C (numpy.ndarray): Centroid means for each cluster of shape (k, d).
    - clss (numpy.ndarray): Index of the cluster each data point belongs to.
    """
    km = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = km.cluster_centers_
    clss = km.labels_

    return C, clss
