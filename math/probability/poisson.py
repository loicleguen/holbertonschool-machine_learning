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
