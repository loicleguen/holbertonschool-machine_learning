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

    def pmf(self, k):
        """
        Calcule la valeur de la PMF pour un nombre donné de "succès".

        Paramètres:
            k (int/float): Le nombre de succès à évaluer.

        Retourne:
            float: La valeur de la PMF pour k, ou 0 si k est hors limites.
        """
        # 1. Conversion en entier comme demandé
        k = int(k)

        # 2. Vérification de la plage de valeurs valides
        if k < 0 or k > self.n:
            return 0

        # 3. Fonction locale pour calculer la factorielle
        def factorial(num):
            fact = 1
            for i in range(1, num + 1):
                fact *= i
            return fact

        # 4. Calcul du coefficient binomial : n! / (k! * (n - k)!)
        comb = factorial(self.n) / (factorial(k) * factorial(self.n - k))

        # 5. Application de la formule globale
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

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

        if k >= self.n:
            return 1.0

        # Somme des PMF de 0 à k inclus
        total_probability = 0.0
        for i in range(k + 1):
            total_probability += self.pmf(i)

        return total_probability
