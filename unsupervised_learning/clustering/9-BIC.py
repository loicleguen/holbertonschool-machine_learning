#!/usr/bin/env python3
"""
Module providing a function to find the best number of clusters for a GMM
using the Bayesian Information Criterion (BIC).
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d).
    - kmin (int): Minimum number of clusters to check (inclusive).
    - kmax (int): Maximum number of clusters to check (inclusive).
    - iterations (int): Maximum number of iterations for EM.
    - tol (float): Tolerance for EM.
    - verbose (bool): Verbosity flag for EM.

    Returns:
    - best_k (int): Best value for k based on BIC.
    - best_result (tuple): (pi, m, S) for the best k.
    - log_likes (numpy.ndarray): Log likelihood for each k.
    - bics (numpy.ndarray): BIC value for each k.
    - Or (None, None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0 or kmax < kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    num_k = kmax - kmin + 1
    log_likes = np.zeros(num_k)
    bics = np.zeros(num_k)
    results = []

    for idx, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_like = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None or m is None or S is None or log_like is None:
            return None, None, None, None

        results.append((pi, m, S))
        log_likes[idx] = log_like

        # Calculation of parameters p
        p = k - 1 + k * d + k * d * (d + 1) / 2
        bics[idx] = p * np.log(n) - 2 * log_like

    best_idx = np.argmin(bics)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, log_likes, bics
