#!/usr/bin/env python3
"""
Module providing a function to calculate a GMM using scikit-learn.
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset using scikit-learn.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d).
    - k (int): Number of clusters.

    Returns:
    - pi (numpy.ndarray): Cluster priors of shape (k,).
    - m (numpy.ndarray): Centroid means of shape (k, d).
    - S (numpy.ndarray): Covariance matrices of shape (k, d, d).
    - clss (numpy.ndarray): Cluster indices for each data point of shape (n,).
    - bic (float): BIC value for the model.
    """
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
