#!/usr/bin/env python3
"""
Ce module fournit une classe Normal pour modéliser
et manipuler une distribution normale.
"""


class Normal:
    """
    Représente une distribution normale.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialise la distribution normale.

        Paramètres:
            data (list): Liste de données pour estimer la distribution.
            mean (float): La moyenne de la distribution.
            stddev (float): L'écart-type de la distribution.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)

            variance_sum = sum((x - self.mean) ** 2 for x in data)
            variance = variance_sum / len(data)

            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calcule le z-score d'une valeur x donnée.

        Paramètres:
            x (int/float): La valeur à convertir.

        Retourne:
            float: Le z-score de x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calcule la valeur x d'un z-score donné.

        Paramètres:
            z (int/float): Le z-score à convertir.

        Retourne:
            float: La valeur x de z.
        """
        return self.mean + (z * self.stddev)
