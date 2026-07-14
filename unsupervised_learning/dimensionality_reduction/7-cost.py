#!/usr/bin/env python3
"""
Module pour calculer le coût de la transformation t-SNE (KL Divergence)
"""
import numpy as np


def cost(P, Q):
    """
    Calcule le coût de la transformation t-SNE en utilisant la divergence KL.

    Args:
        P: np.ndarray de forme (n, n) contenant les affinités P.
        Q: np.ndarray de forme (n, n) contenant les affinités Q.

    Returns:
        C: le coût (divergence de Kullback-Leibler) de la transformation.
    """
    # Évite les erreurs de division par 0 ou log(0)
    P_safe = np.maximum(P, 1e-12)
    Q_safe = np.maximum(Q, 1e-12)

    # Calcul de la divergence de Kullback-Leibler
    # C = sum(P * log(P / Q))
    C = np.sum(P_safe * np.log(P_safe / Q_safe))

    return C
