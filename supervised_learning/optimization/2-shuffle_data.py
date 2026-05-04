#!/usr/bin/env python3
"""Shuffle the data points in two matrices the same way."""
import numpy as np


def shuffle_data(X, Y):
    """Shuffle the data points in two matrices the same way."""
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
