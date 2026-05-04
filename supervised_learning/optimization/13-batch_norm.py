#!/usr/bin/env python3
"""performs batch normalization on an unactivated output of a neural network"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a neural network using batch"""
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)
    Z_std = (Z - mean) / np.sqrt(var + epsilon)
    Z_norm = gamma * Z_std + beta
    return Z_norm
