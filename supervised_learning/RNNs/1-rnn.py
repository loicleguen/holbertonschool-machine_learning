#!/usr/bin/env python3
"""Module pour la propagation avant d'un RNN simple."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Effectue la propagation avant pour un réseau de neurones récurrent.

    Args:
        rnn_cell (RNNCell): Instance de la classe RNNCell.
        X (numpy.ndarray): Données d'entrée de forme (t, m, i).
        h_0 (numpy.ndarray): État caché initial de forme (m, h).

    Returns:
        H (numpy.ndarray): Tous les états cachés de forme (t + 1, m, h).
        Y (numpy.ndarray): Toutes les sorties de forme (t, m, o).
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    # Initialisation du tableau H pour contenir tous les états cachés (t + 1)
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Initialisation du tableau Y pour les sorties
    Y = np.zeros((t, m, o))

    # Boucle sur chaque pas de temps
    h_prev = h_0
    for step in range(t):
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y[step] = y
        h_prev = h_next

    return H, Y
