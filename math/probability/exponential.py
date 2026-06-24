#!/usr/bin/env python3
"""
Ce module fournit une classe Exponential pour modéliser
et manipuler une distribution exponentielle.
"""


class Exponential:
    """
    Représente une distribution exponentielle.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialise la distribution exponentielle.

        Paramètres:
            data (list): Liste de données pour estimer la distribution.
            lambtha (float): Le taux attendu d'occurrences.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """
        Calcule la valeur de la PDF pour une période de temps donnée.

        Paramètres:
            x (int/float): La période de temps à évaluer.

        Retourne:
            float: La valeur de la PDF pour x, ou 0 si x est hors limites.
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calcule la valeur de la CDF pour une période de temps donnée.

        Paramètres:
            x (int/float): La période de temps à évaluer.

        Retourne:
            float: La valeur de la CDF pour x, ou 0 si x est hors limites.
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return 1 - (e ** (-self.lambtha * x))
