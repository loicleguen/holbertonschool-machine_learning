#!/usr/bin/env python3
"""
Module traitant de l'initialisation d'un processus gaussien 1D sans bruit.
"""
import numpy as np


class GaussianProcess:
    """
    Représente un processus gaussien 1D sans bruit.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Constructeur de la classe GaussianProcess.

        Parameters:
        -----------
        X_init : numpy.ndarray de forme (t, 1)
            Représente les entrées déjà échantillonnées.
        Y_init : numpy.ndarray de forme (t, 1)
            Représente les sorties de la fonction boîte noire pour chaque X.
        l : float
            Paramètre d'échelle de longueur (length parameter) du noyau.
        sigma_f : float
            Écart-type donné à la sortie de la fonction boîte noire.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calcule la matrice de covariance (noyau RBF) entre deux matrices.

        Parameters:
        -----------
        X1 : numpy.ndarray de forme (m, 1)
            Première matrice d'entrées.
        X2 : numpy.ndarray de forme (n, 1)
            Deuxième matrice d'entrées.

        Returns:
        --------
        numpy.ndarray de forme (m, n)
            La matrice de covariance.
        """
        # Distance au carré entre chaque paire de points X1 et X2
        # (X1 - X2.T)^2 tire parti du broadcasting NumPy
        #       pour générer une forme (m, n)
        sqdist = (X1 - X2.T) ** 2

        # Noyau Radial Basis Function (RBF) / Exponentielle Carrée
        K = (self.sigma_f ** 2) * np.exp(-0.5 * sqdist / (self.l ** 2))
        return K
