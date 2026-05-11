#!/usr/bin/env python3
"""Calculates sensitivity for each class."""
import numpy as np


def sensitivity(confusion):
    """calculates sensitivity for each class."""
    return np.diag(confusion) / np.sum(confusion, axis=1)
