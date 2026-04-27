#!/usr/bin/env python3
"""Converts a label vector into a one-hot matrix"""
import numpy as np


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix"""
    if classes is None:
        classes = np.max(labels) + 1
    return np.eye(classes)[labels]
