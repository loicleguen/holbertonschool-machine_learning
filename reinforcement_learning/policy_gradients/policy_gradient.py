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
    z = np.dot(matrix, weight)
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Calcule l'action à sélectionner et le gradient de politique Monte-Carlo.

    Args:
        state (np.ndarray): La matrice ou le vecteur représentant l'état.
        weight (np.ndarray): La matrice de poids.

    Returns:
        tuple: (action, gradient)
            - action (int): L'action choisie selon la distribution.
            - gradient (np.ndarray): Le gradient de log-probabilité.
    """
    # S'assurer que state est au format (1, n)
    if state.ndim == 1:
        state = state.reshape(1, -1)

    # Calcul de la distribution de probabilité avec la fonction policy
    probs = policy(state, weight)

    # Sélection aléatoire d'une action basée sur les probabilités calculées
    action = np.random.choice(probs.shape[1], p=probs[0])

    # Création du vecteur one-hot pour l'action choisie
    d_softmax = probs.copy()
    d_softmax[0, action] -= 1

    # Calcul du gradient : s^T * (probs - one_hot) = -s^T * d_log_pi
    grad = -np.dot(state.T, d_softmax)

    return action, grad
