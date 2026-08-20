#!/usr/bin/env python3
"""
Module containing the BidirectionalCell class
        with forward and backward methods.
"""
import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN.
    """

    def __init__(self, i, h, o):
        """
        Class constructor.

        Parameters:
            i (int): Dimensionality of the data inputs.
            h (int): Dimensionality of the hidden states.
            o (int): Dimensionality of the outputs.
        """
        # Poids pour le passage avant (forward)
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        # Poids pour le passage arrière (backward)
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        # Poids pour la sortie (output)
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.

        Parameters:
            h_prev (numpy.ndarray): Shape (m, h) containing the previous
                                   hidden state.
            x_t (numpy.ndarray): Shape (m, i) containing the data input for
                                 the cell.

        Returns:
            h_next (numpy.ndarray): The next hidden state.
        """
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(x_concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the
            backward direction for one time step.

        Parameters:
            h_next (numpy.ndarray): Shape (m, h) containing the next
                                    hidden state.
            x_t (numpy.ndarray): Shape (m, i) containing the data input for
                                 the cell.

        Returns:
            h_prev (numpy.ndarray): The previous
                hidden state (backward direction).
        """
        # Concaténation de l'état suivant (h_next) et de l'entrée x_t
        x_concat = np.concatenate((h_next, x_t), axis=1)

        # Calcul de l'état caché précédent avec
        #       les poids rétrogrades Whb et bhb
        h_prev = np.tanh(np.dot(x_concat, self.Whb) + self.bhb)

        return h_prev
