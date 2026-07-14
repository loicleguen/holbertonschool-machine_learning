#!/usr/bin/env python3
"""
Module pour exécuter l'algorithme complet t-SNE
"""
import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Effectue une transformation t-SNE complète sur un jeu de données.

    Args:
        X: np.ndarray de forme (n, d) contenant le jeu de données d'origine.
        ndims: nouvelle dimensionnalité de l'espace de sortie.
        idims: dimension intermédiaire de réduction par PCA avant t-SNE.
        perplexity: perplexité pour le calcul des affinités.
        iterations: nombre total de boucles d'optimisation.
        lr: taux d'apprentissage pour la descente de gradient.

    Returns:
        Y: np.ndarray de forme (n, ndims) contenant la projection optimisée.
    """
    # 1. Étape de pré-réduction de dimensionnalité par PCA
    X_reduced = pca(X, idims)
    n = X_reduced.shape[0]

    # 2. Initialisation de Y avec une Gaussienne (moyenne 0, std 1e-4)
    Y = np.random.randn(n, ndims) * 1e-4

    # 3. Initialisation de la vitesse pour le momentum
    i_Y = np.zeros((n, ndims))

    # 4. Calcul des affinités P
    P = P_affinities(X_reduced, perplexity=perplexity)

    # Application de l'exagération précoce (early exaggeration)
    P = P * 4.0

    # 5. Boucle principale de descente de gradient
    for it in range(1, iterations + 1):
        # Calcul des gradients dY et des affinités Q
        dY, Q = grads(Y, P)

        # Détermination du momentum a(t)
        if it <= 20:
            momentum = 0.5
        else:
            momentum = 0.8

        # Mise à jour de la vitesse
        i_Y = momentum * i_Y - lr * dY

        # Mise à jour des coordonnées Y
        Y = Y + i_Y

        # Centrage systématique de Y
        Y = Y - np.mean(Y, axis=0)

        # AFFICHER d'abord le coût (avant de modifier P)
        if it % 100 == 0:
            if it <= 100:
                # P est toujours exagéré à ce stade de la boucle,
                # on le divise temporairement par 4 pour le calcul du coût réel
                current_cost = cost(P / 4.0, Q)
            else:
                current_cost = cost(P, Q)
            print("Cost at iteration {}: {}".format(it, current_cost))

        # APPLIQUER le retrait de l'exagération précoce
        #       après l'affichage du coût à 100
        if it == 100:
            P = P / 4.0

    return Y
