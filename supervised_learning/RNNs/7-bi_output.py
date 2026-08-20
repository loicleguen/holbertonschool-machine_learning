#!/usr/bin/env python3
"""
Module containing the BidirectionalCell class with forward, backward,
and output methods.
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
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.
        """
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(x_concat, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the
            backward direction for one time step.
        """
        x_concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(x_concat, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN.

        Parameters:
            H (numpy.ndarray): Shape (t, m, 2 * h) containing the concatenated
                              hidden states from both directions.

        Returns:
            Y (numpy.ndarray): Outputs of shape (t, m, o).
        """
        t, m, _ = H.shape
        o = self.Wy.shape[1]

        Y = np.zeros((t, m, o))

        for step in range(t):
            # Produit matriciel entre les états cachés concaténés et Wy + bias
            logits = np.dot(H[step], self.Wy) + self.by

            # Activation Softmax axe par axe
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            Y[step] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return Y
