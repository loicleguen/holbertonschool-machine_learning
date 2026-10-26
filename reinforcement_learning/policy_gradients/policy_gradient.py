#!/usr/bin/env python3
"""
Module qui contient des fonctions pour calculer les gradients de politique.
"""
import numpy as np


def policy(matrix, weight):
    """
    Calcule la politique (distribution de probabilité des actions)
    en utilisant une fonction softmax sur le produit matriciel
    de la matrice d'état et des poids.

    Args:
        matrix (np.ndarray): La matrice représentant l'état (shape: 1 x n).
        weight (np.ndarray): La matrice des poids (shape: n x a).

    Returns:
        np.ndarray: La distribution de probabilités pour chaque action.
    """
    # Calcul des logits (produit matriciel des états et des poids)
    z = np.dot(matrix, weight)

    # Calcul de la fonction softmax pour obtenir des probabilités
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
