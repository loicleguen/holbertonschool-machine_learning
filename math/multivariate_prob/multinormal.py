#!/usr/bin/env python3
"""
Module définissant la classe MultiNormal pour une distribution
gaussienne multivariée.
"""
import numpy as np


class MultiNormal:
    """
    Représente une distribution normale multivariée.
    """

    def __init__(self, data):
        """
        Initialise la distribution à partir d'un ensemble de données.

        Parameters:
        data (numpy.ndarray) : Tableau de
                               dimensions (d, n) contenant le dataset.
            - d est le nombre de dimensions de chaque point.
            - n est le nombre de points de données.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calcul de la moyenne le long de l'axe des colonnes (axis=1)
        # keepdims=True permet de garder la forme (d, 1) attendue
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Centrage des données (data - moyenne)
        data_centered = data - self.mean

        # Formule de la covariance adaptée à la forme (d, n) :
        # Σ = (1 / (n - 1)) * (data_centered @ data_centered.T)
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)
