#!/usr/bin/env python3
"""Converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    try:
        if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
            return None

        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
