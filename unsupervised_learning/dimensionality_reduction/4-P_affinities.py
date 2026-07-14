#!/usr/bin/env python3
"""
Module pour calculer les affinités symétriques P d'un jeu de données
"""
import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calcule les affinités symétriques P d'un jeu de données.

    Args:
        X: np.ndarray de forme (n, d) contenant le jeu de données.
        tol: la tolérance maximale autorisée pour l'entropie de Shannon.
        perplexity: la perplexité cible pour toutes les distributions.

    Returns:
        P: np.ndarray de forme (n, n) contenant les affinités symétriques.
    """
    n, _ = X.shape
    # 1. Initialisation des variables
    D, P, betas, H = P_init(X, perplexity)

    # 2. Recherche de la valeur optimale de beta pour chaque point
    for i in range(n):
        # On extrait les distances du point i à
        # tous les autres points (sauf lui-même)
        Di = np.delete(D[i], i)

        # Bornes pour la recherche dichotomique
        beta_min = None
        beta_max = None
        beta = betas[i, 0]

        # Calcul initial de l'entropie et des affinités Pi
        Hi, Pi = HP(Di, beta)
        h_diff = Hi - H

        # Boucle de recherche dichotomique (limite
        # arbitraire pour éviter une boucle infinie)
        # Mais le critère d'arrêt principal est |Hi - H| <= tol
        for _ in range(50):
            if np.abs(h_diff) <= tol:
                break

            # Si l'entropie calculée est supérieure à l'entropie cible
            # On doit augmenter beta (rétrécir la gaussienne)
            if h_diff > 0:
                beta_min = beta
                if beta_max is None:
                    beta = beta * 2.0
                else:
                    beta = (beta + beta_max) / 2.0

            # Si l'entropie calculée est inférieure à l'entropie cible
            # On doit diminuer beta (élargir la gaussienne)
            else:
                beta_max = beta
                if beta_min is None:
                    beta = beta / 2.0
                else:
                    beta = (beta + beta_min) / 2.0

            # Recalculer l'entropie avec le nouveau beta
            Hi, Pi = HP(Di, beta)
            h_diff = Hi - H

        # Enregistrement du beta trouvé et insertion des affinités Pi dans P
        betas[i, 0] = beta
        # On replace Pi dans la ligne P[i] en sautant
        # l'élément diagonal i (qui reste à 0)
        P[i, np.arange(n) != i] = Pi

    # 3. Symétrisation des affinités jointes P
    # P_sym = (P + P^T) / (2 * n)
    P_symmetric = (P + P.T) / (2 * n)

    return P_symmetric
