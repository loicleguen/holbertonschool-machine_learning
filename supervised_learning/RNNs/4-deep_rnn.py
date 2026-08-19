#!/usr/bin/env python3
"""Module pour la propagation avant d'un Deep RNN."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Effectue la propagation avant pour un
        réseau de neurones récurrent profond.

    Args:
        rnn_cells (list): Liste d'instances de RNNCell de longueur l.
        X (numpy.ndarray): Données d'entrée de forme (t, m, i).
        h_0 (numpy.ndarray): États cachés initiaux de forme (l, m, h).

    Returns:
        H (numpy.ndarray): Tous les états cachés de forme (t + 1, l, m, h).
        Y (numpy.ndarray): Toutes les sorties de forme (t, m, o).
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    # Initialisation de H (t + 1 pas, l couches, m exemples, h dimensions)
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    # Initialisation de Y (t pas, m exemples, o sorties)
    Y = np.zeros((t, m, o))

    # On conserve les derniers états cachés pour chaque couche
    current_h = h_0.copy()

    for step in range(t):
        # L'entrée du premier niveau à l'instant t est X[step]
        x_t = X[step]

        for layer in range(l):
            cell = rnn_cells[layer]
            h_prev = current_h[layer]

            # Forward dans la cellule courante
            h_next, y = cell.forward(h_prev, x_t)

            # Mise à jour des structures
            current_h[layer] = h_next
            H[step + 1, layer] = h_next

            # La sortie de cette couche devient l'entrée de la suivante
            x_t = h_next

        # Seule la sortie de la toute dernière couche est conservée dans Y
        Y[step] = y

    return H, Y
