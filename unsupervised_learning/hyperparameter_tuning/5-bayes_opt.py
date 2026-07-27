#!/usr/bin/env python3
"""
Module pour l'exécution complète de l'optimisation bayésienne
sur un processus gaussien 1D sans bruit.
"""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """
        Calcule la prochaine meilleure position d'échantillonnage
        en utilisant la fonction d'acquisition Expected Improvement (EI).

        Returns:
        --------
        X_next : numpy.ndarray de forme (1,)
            Représente le prochain meilleur point à échantillonner.
        EI : numpy.ndarray de forme (ac_samples,)
            Contient l'amélioration espérée pour
            chaque point candidat dans X_s.
        """
        # Obtenir les prédictions (moyenne et variance)
        #       pour tous les points X_s
        mu, sigma = self.gp.predict(self.X_s)

        # Déterminer la meilleure valeur observée à ce jour (Y_opt)
        if self.minimize:
            y_opt = np.min(self.gp.Y)
            mu_diff = y_opt - mu - self.xsi
        else:
            y_opt = np.max(self.gp.Y)
            mu_diff = mu - y_opt - self.xsi

        # Traitement des cas où la variance est nulle (ou extrêmement
        #       proche de 0) pour éviter les divisions par zéro
        with np.errstate(divide='warn'):
            Z = np.where(sigma > 0, mu_diff / sigma, 0.0)
            EI = np.where(sigma > 0,
                          mu_diff * norm.cdf(Z) + sigma * norm.pdf(Z),
                          0.0)

        # Le point candidat ayant le plus grand EI
        #       est le prochain meilleur point
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimise la fonction boîte noire sur un nombre maximal d'itérations.

        Parameters:
        -----------
        iterations : int
            Le nombre maximal d'itérations à effectuer.

        Returns:
        --------
        X_opt : numpy.ndarray de forme (1,)
            Représente le point optimal trouvé.
        Y_opt : numpy.ndarray de forme (1,)
            Représente la valeur optimale de la fonction.
        """
        for _ in range(iterations):
            # Calcul du prochain point prometteur
            X_next, _ = self.acquisition()

            # Vérification de l'arrêt précoce si le point
            #       a déjà été échantillonné
            if np.isin(X_next, self.gp.X):
                break

            # Évaluation de la fonction réelle au point X_next
            Y_next = self.f(X_next)

            # Mise à jour du modèle avec la nouvelle observation
            self.gp.update(X_next, Y_next)

        # Recherche du meilleur point parmi tous ceux observés
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
