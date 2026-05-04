#!/usr/bin/env python3
"""Creates mini-batches from the training data"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Creates mini-batches from the training data"""
    X_shuff, Y_shuff = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []
    for i in range(0, m, batch_size):
        X_mini = X_shuff[i:i + batch_size]
        Y_mini = Y_shuff[i:i + batch_size]
        mini_batches.append((X_mini, Y_mini))
    return mini_batches
