#!/usr/bin/env python3
"""
Module pour initialiser les variables de calcul des affinités P dans t-SNE
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initialise toutes les variables requises pour calculer les affinités P.

    Args:
        X: np.ndarray de forme (n, d) contenant le jeu de données.
            n est le nombre de points de données.
            d est le nombre de dimensions de chaque point.
        perplexity: la perplexité souhaitée pour toutes les distributions.

    Returns:
        (D, P, betas, H)
        D: np.ndarray de forme (n, n) des distances euclidiennes au carré.
        P: np.ndarray de forme (n, n) rempli de 0.
        betas: np.ndarray de forme (n, 1) rempli de 1 (les précisions beta).
        H: l'entropie de Shannon pour la perplexité donnée (en base 2).
    """
    n, _ = X.shape

    # 1. Calcul de la distance euclidienne au carré par paires (D)
    # Formule mathématique optimisée : x^2 - 2xy + y^2
    sum_X = np.sum(np.square(X), axis=1, keepdims=True)
    D = sum_X - 2 * np.dot(X, X.T) + sum_X.T

    # On force la diagonale à zéro pour corriger les imprécisions de calcul
    np.fill_diagonal(D, 0)

    # 2. Initialisation de la matrice des affinités P à zéro
    P = np.zeros((n, n))

    # 3. Initialisation de beta (précisions des gaussiennes) à 1.0
    betas = np.ones((n, 1))

    # 4. Calcul de l'entropie de Shannon (base 2) associée à la perplexité
    H = np.log2(perplexity)

    return D, P, betas, H
