#!/usr/bin/env python3
import numpy as np


def normalization_constants(X):
    """calculates the normalization constants
    of a matrix: mean and standard deviation"""
    return np.mean(X, axis=0), np.std(X, axis=0)
