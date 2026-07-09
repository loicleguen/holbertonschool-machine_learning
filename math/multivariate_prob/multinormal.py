#!/usr/bin/env python3
"""
Module définissant la classe MultiNormal pour une distribution
gaussienne multivariée et le calcul de sa PDF.
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

    def pdf(self, x):
        """
        Calcule la valeur de la PDF (Probability Density Function)
        en un point de données donné.

        Parameters:
        x (numpy.ndarray) : Tableau de dimensions (d, 1) contenant le point.

        Returns:
        float : La valeur de la PDF au point x.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if len(x.shape) != 2 or x.shape[0] != d or x.shape[1] != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Extraction du déterminant et de l'inverse de la matrice de covariance
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # Calcul du coefficient de normalisation : 1 / sqrt((2 * pi)^d * det)
        norm_factor = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)

        # Calcul de la distance de Mahalanobis au carré :
        # (x - mu)^T * Σ^-1 * (x - mu)
        x_centered = x - self.mean
        exponent = -0.5 * np.dot(np.dot(x_centered.T, inv), x_centered)

        # L'exposant renvoie une matrice 1x1 (ex: [[valeur]]),
        # on extrait le scalaire avec [0][0]
        pdf_value = norm_factor * np.exp(exponent[0][0])

        return pdf_value
