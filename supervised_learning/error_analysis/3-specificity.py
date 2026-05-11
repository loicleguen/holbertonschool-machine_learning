#!/usr/bin/env python3
"""Calculates specificity for each class."""
import numpy as np


def specificity(confusion):
    """Calculate the specificity for each class."""
    total = np.sum(confusion)
    row_sum = np.sum(confusion, axis=1)
    col_sum = np.sum(confusion, axis=0)
    tp = np.diag(confusion)
    tn = total - row_sum - col_sum + tp
    fp = col_sum - tp

    return tn / (tn + fp)
