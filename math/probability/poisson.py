#!/usr/bin/env python3
"""
Ce module fournit une classe Poisson pour modéliser
et manipuler une distribution de Poisson.
"""


class Poisson:
    """
    Représente une distribution de Poisson.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialise la distribution de Poisson.

        Paramètres:
            data (list): Liste de données pour estimer la distribution.
            lambtha (float): Le nombre attendu d'occurrences.
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

            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calcule la valeur de la PMF pour un nombre donné de "succès".

        Paramètres:
            k (int/float): Le nombre de succès à évaluer.

        Retourne:
            float: La valeur de la PMF pour k, ou 0 si k est hors limites.
        """
        k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        fact = 1
        for i in range(1, k + 1):
            fact *= i
        pmf = (e ** -self.lambtha) * (self.lambtha ** k) / fact
        return pmf

    def cdf(self, k):
        """
        Calcule la valeur de la CDF pour un nombre donné de "succès".

        Paramètres:
            k (int/float): Le nombre de succès à évaluer.

        Retourne:
            float: La valeur de la CDF pour k, ou 0 si k est hors limites.
        """
        k = int(k)
        if k < 0:
            return 0
        totproba = 0.0
        for i in range(k + 1):
            totproba += self.pmf(i)
        return totproba
