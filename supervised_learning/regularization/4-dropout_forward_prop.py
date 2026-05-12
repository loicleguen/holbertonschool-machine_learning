#!/usr/bin/env python3
"""Module that performs forward propagation with dropout regularization"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Function that performs forward propagation
    with dropout regularization"""
    cache = {'A0': X}
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        Z = np.matmul(W, A_prev) + b

        if i == L:
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            cache['A' + str(i)] = expZ / np.sum(expZ, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            cache['D' + str(i)] = D
            cache['A' + str(i)] = A * D / keep_prob

    return cache
