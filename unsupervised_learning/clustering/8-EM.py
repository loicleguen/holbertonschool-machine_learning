#!/usr/bin/env python3
"""
Module providing a function to perform Expectation Maximization for a GMM.
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs Expectation Maximization for a Gaussian Mixture Model (GMM).

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d).
    - k (int): Positive integer containing the number of clusters.
    - iterations (int): Maximum number of iterations.
    - tol (float): Non-negative tolerance of the log likelihood for stopping.
    - verbose (bool): If True, prints information about the algorithm.

    Returns:
    - pi, m, S, g, log_like or (None, None, None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, log_like = expectation(X, pi, m, S)
    if g is None or log_like is None:
        return None, None, None, None, None

    i = 0
    while i < iterations:
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {:.5f}".format(
                i, log_like))

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, new_log_like = expectation(X, pi, m, S)
        if g is None or new_log_like is None:
            return None, None, None, None, None

        if abs(new_log_like - log_like) <= tol:
            i += 1
            log_like = new_log_like
            break

        log_like = new_log_like
        i += 1

    if verbose:
        print("Log Likelihood after {} iterations: {:.5f}".format(
            i, log_like))

    return pi, m, S, g, log_like
