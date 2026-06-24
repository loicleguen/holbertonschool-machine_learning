#!/usr/bin/env python3
"""
Ce module fournit une classe Binomial pour modéliser
et manipuler une distribution binomiale.
"""


class Binomial:
    """
    Représente une distribution binomiale.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialise la distribution binomiale.

        Paramètres:
            data (list): Liste de données pour estimer la distribution.
            n (int): Le nombre d'essais de Bernoulli.
            p (float): La probabilité de succès.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p_initial = 1 - (variance / mean)

            n_calculated = mean / p_initial
            self.n = int(round(n_calculated))

            self.p = float(mean / self.n)
