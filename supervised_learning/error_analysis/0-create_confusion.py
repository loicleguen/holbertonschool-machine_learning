#!/usr/bin/env python3
"""Creates a confusion matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Create a confusion matrix."""
    if (not isinstance(labels, np.ndarray)
            or not isinstance(logits, np.ndarray)):
        return None
    if labels.shape != logits.shape:
        return None

    _, classes = labels.shape
    confusion = np.zeros((classes, classes), dtype=int)
    true_idx = np.argmax(labels, axis=1)
    pred_idx = np.argmax(logits, axis=1)

    for t, p in zip(true_idx, pred_idx):
        confusion[t, p] += 1

    return confusion
