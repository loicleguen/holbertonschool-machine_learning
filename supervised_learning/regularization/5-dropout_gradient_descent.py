#!/usr/bin/env python3
"""5-dropout_gradient_descent.py"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights and biases of a
    neural network using gradient descent"""
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            dA = np.matmul(weights['W' + str(i)].T, dZ)
            dA *= cache['D' + str(i - 1)] / keep_prob
            dZ = dA * (1 - np.power(cache['A' + str(i - 1)], 2))
