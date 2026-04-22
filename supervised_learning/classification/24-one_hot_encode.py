#!/usr/bin/env python3
"""Converts a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    try:
        if not isinstance(Y, np.ndarray) or Y.ndim != 1:
            return None
        if not isinstance(classes, int) or classes <= 0:
            return None
        if np.max(Y) >= classes or np.min(Y) < 0:
            return None

        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
