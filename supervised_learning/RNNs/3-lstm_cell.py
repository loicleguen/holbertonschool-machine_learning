#!/usr/bin/env python3
"""Module contenant la classe LSTMCell."""

import numpy as np


class LSTMCell:
    """Représente une cellule d'un réseau LSTM."""

    def __init__(self, i, h, o):
        """Initialise les attributs de la cellule LSTM.

        Args:
            i (int): Dimension des données d'entrée.
            h (int): Dimension de l'état caché.
            o (int): Dimension des sorties.
        """
        # Matrice de poids pour les portes (forget, update, candidate, output)
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))

        # Matrice de poids pour la sortie finale (de dimension h vers o)
        self.Wy = np.random.normal(size=(h, o))

        # Biais initialisés à zéro
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Effectue la propagation avant pour un pas de temps.

        Args:
            h_prev (numpy.ndarray): État caché précédent de forme (m, h)
            c_prev (numpy.ndarray): État de cellule précédent de forme (m, h)
            x_t (numpy.ndarray): Entrée de données de forme (m, i)

        Returns:
            h_next (numpy.ndarray): Prochain état caché
            c_next (numpy.ndarray): Prochain état de cellule
            y (numpy.ndarray): Sortie de la cellule
        """
        # Concaténation de h_prev et x_t le long des colonnes
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = 1 / (1 + np.exp(-(np.dot(concat, self.Wf) + self.bf)))

        # Update gate
        u_t = 1 / (1 + np.exp(-(np.dot(concat, self.Wu) + self.bu)))

        # Intermediate cell state (candidate)
        c_tilde = np.tanh(np.dot(concat, self.Wc) + self.bc)

        # Nouveau cell state
        c_next = f_t * c_prev + u_t * c_tilde

        # Output gate
        o_t = 1 / (1 + np.exp(-(np.dot(concat, self.Wo) + self.bo)))

        # Nouveau hidden state
        h_next = o_t * np.tanh(c_next)

        # Calcul de la sortie softmax
        y_linear = np.dot(h_next, self.Wy) + self.by
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, c_next, y
