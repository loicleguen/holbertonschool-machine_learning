#!/usr/bin/env python3
"""Module contenant la classe RNNCell."""

import numpy as np


class RNNCell:
    """Représente une cellule d'un RNN simple."""

    def __init__(self, i, h, o):
        """Initialise les attributs de la cellule RNN.

        Args:
            i (int): Dimension des données d'entrée.
            h (int): Dimension de l'état caché.
            o (int): Dimension des sorties.
        """
        # Matrice de poids concaténée pour (h_prev, x_t) -> (h + i, h)
        self.Wh = np.random.normal(size=(h + i, h))
        # Matrice de poids pour la sortie -> (h, o)
        self.Wy = np.random.normal(size=(h, o))
        # Biais initialisés à zéro
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Effectue la propagation avant pour un pas de temps.

        Args:
            h_prev (numpy.ndarray): État caché précédent de forme (m, h)
            x_t (numpy.ndarray): Entrée de données de forme (m, i)

        Returns:
            h_next (numpy.ndarray): Prochain état caché
            y (numpy.ndarray): Sortie de la cellule
        """
        # Concaténation de h_prev et x_t le long des colonnes (axe 1)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Calcul du nouvel état caché avec la fonction d'activation tanh
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)

        # Calcul de la sortie non normalisée (logits)
        y_linear = np.dot(h_next, self.Wy) + self.by

        # Application de la fonction d'activation Softmax
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
