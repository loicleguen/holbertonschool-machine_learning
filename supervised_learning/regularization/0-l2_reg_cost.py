#!/usr/bin/env python3
"""Calculates the cost of a neural network with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization"""
    sum_sq = 0.0
    for i in range(1, L + 1):
        W = weights.get('W' + str(i))
        if W is not None:
            sum_sq += np.sum(np.square(W))
    reg = (lambtha / (2.0 * m)) * sum_sq
    return cost + reg
