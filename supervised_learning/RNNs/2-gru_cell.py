#!/usr/bin/env python3
"""Module pour la création d'une cellule GRU."""

import numpy as np


class GRUCell:
    """Représente une cellule GRU (Gated Recurrent Unit)."""

    def __init__(self, i, h, o):
        """Constructeur de la cellule GRU.

        Args:
            i (int): Dimension des données d'entrée.
            h (int): Dimension de l'état caché.
            o (int): Dimension des sorties.
        """
        # Initialisation des poids (distribution normale)
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        # Initialisation des biais à zéro
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Effectue la propagation avant pour un pas de temps.

        Args:
            h_prev (numpy.ndarray): État caché précédent de forme (m, h).
            x_t (numpy.ndarray): Données d'entrée de forme (m, i).

        Returns:
            h_next (numpy.ndarray): État caché suivant.
            y (numpy.ndarray): Sortie de la cellule.
        """
        # Concaténation de h_prev et x_t (m, h + i)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # 1. Update Gate (z_t)
        z_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))

        # 2. Reset Gate (r_t)
        r_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))

        # 3. Intermediate Hidden State (h_tilde)
        # On applique la porte de réinitialisation à h_prev
        concat_r = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)

        # 4. Next Hidden State (h_next)
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # 5. Output Prediction (y)
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y
