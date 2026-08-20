#!/usr/bin/env python3
"""
Module containing the bi_rnn function.
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Parameters:
        bi_cell (BidirectionalCell): Instance of BidirectionalCell used for
                                     the forward propagation.
        X (numpy.ndarray): Shape (t, m, i) containing the data to be used.
        h_0 (numpy.ndarray): Shape (m, h) containing the initial hidden state
                             in the forward direction.
        h_t (numpy.ndarray): Shape (m, h) containing the initial hidden state
                             in the backward direction.

    Returns:
        H (numpy.ndarray): Shape (t, m, 2 * h) containing all concatenated
                           hidden states.
        Y (numpy.ndarray): Shape (t, m, o) containing all outputs.
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    # Initialisation des tableaux pour stocker les états cachés
    H_f = np.zeros((t, m, h))
    H_b = np.zeros((t, m, h))

    # Propagation dans le sens chronologique (Forward)
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        H_f[step] = h_prev

    # Propagation dans le sens rétrograde (Backward)
    h_next = h_t
    for step in range(t - 1, -1, -1):
        h_next = bi_cell.backward(h_next, X[step])
        H_b[step] = h_next

    # Concaténation sur l'axe des dimensions cachées (axis=2)
    H = np.concatenate((H_f, H_b), axis=2)

    # Calcul de toutes les sorties via la méthode output de la cellule
    Y = bi_cell.output(H)

    return H, Y
