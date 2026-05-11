#!/usr/bin/env python3
"""Calculates precision for each class."""
import numpy as np


def precision(confusion):
    """Calculate precision for each class."""
    return np.diag(confusion) / np.sum(confusion, axis=0)
