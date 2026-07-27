#!/usr/bin/env python3
"""
Module pour l'initialisation de l'optimisation bayésienne
sur un processus gaussien 1D sans bruit.
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Effectue une optimisation bayésienne sur
    un processus gaussien 1D sans bruit.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Constructeur de la classe BayesianOptimization.

        Parameters:
        -----------
        f : function
            La fonction boîte noire à optimiser.
        X_init : numpy.ndarray de forme (t, 1)
            Représente les entrées déjà échantillonnées.
        Y_init : numpy.ndarray de forme (t, 1)
            Représente les sorties de la fonction boîte noire pour chaque X.
        bounds : tuple de forme (min, max)
            Limites de l'espace dans lequel chercher le point optimal.
        ac_samples : int
            Nombre de points à analyser lors de la fonction d'acquisition.
        l : float
            Paramètre d'échelle de longueur (length parameter) du noyau.
        sigma_f : float
            Écart-type donné à la sortie de la fonction boîte noire.
        xsi : float
            Facteur d'exploration-exploitation pour l'acquisition.
        minimize : bool
            True si l'optimisation vise la minimisation,
            False pour la maximisation.
        """
        # Attributs publics
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

        # Génération des points d'échantillonnage régulièrement
        #       espacés entre min et max
        X_s = np.linspace(bounds[0], bounds[1], ac_samples)
        self.X_s = X_s.reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize
