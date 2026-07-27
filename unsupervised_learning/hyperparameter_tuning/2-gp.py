#!/usr/bin/env python3
"""
Module traitant de la mise à jour d'un processus gaussien 1D sans bruit
lors de l'ajout d'un nouveau point d'échantillonnage.
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

    def predict(self, X_s):
        """
        Prédit la moyenne et la variance des points dans un processus gaussien.

        Parameters:
        -----------
        X_s : numpy.ndarray de forme (s, 1)
            Contient tous les points dont la moyenne et la variance
            doivent être calculées.

        Returns:
        --------
        mu : numpy.ndarray de forme (s,)
            Contient la moyenne pour chaque point dans X_s.
        sigma : numpy.ndarray de forme (s,)
            Contient la variance pour chaque point dans X_s.
        """
        # Covariance entre les points d'entraînement (self.X) et les points X_s
        K_s = self.kernel(self.X, X_s)

        # Covariance des points X_s entre eux
        K_ss = self.kernel(X_s, X_s)

        # Inverse de la matrice de covariance d'entraînement K
        K_inv = np.linalg.inv(self.K)

        # Calcul de la moyenne prédite (mu) : K_s^T * K_inv * Y
        mu = K_s.T.dot(K_inv).dot(self.Y)
        # Redimensionnement en vecteur 1D de forme (s,)
        mu = mu.reshape(-1)

        # Calcul de la matrice de covariance des prédictions
        sigma_matrix = K_ss - K_s.T.dot(K_inv).dot(K_s)

        # Extraction de la diagonale représentant la variance de chaque point
        sigma = np.diag(sigma_matrix)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Met à jour un processus gaussien avec un
        nouveau point d'échantillonnage.

        Parameters:
        -----------
        X_new : numpy.ndarray de forme (1,)
            Représente le nouveau point d'échantillonnage.
        Y_new : numpy.ndarray de forme (1,)
            Représente la valeur de la fonction au nouveau point.
        """
        # Redimensionnement des entrées de (1,) à (1, 1) pour l'empilement
        X_new_reshaped = X_new.reshape(-1, 1)
        Y_new_reshaped = Y_new.reshape(-1, 1)

        # Ajout du nouveau point aux attributs d'instance X et Y
        self.X = np.vstack((self.X, X_new_reshaped))
        self.Y = np.vstack((self.Y, Y_new_reshaped))

        # Recalcul de la matrice de covariance K avec l'ensemble des points X
        self.K = self.kernel(self.X, self.X)
