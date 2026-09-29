#!/usr/bin/env python3
"""Module pour implémenter l'algorithme Epsilon-Greedy."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Détermine la prochaine action en utilisant la méthode epsilon-greedy.

    Args:
        Q (numpy.ndarray): La Q-table.
        state (int): L'état actuel.
        epsilon (float): Le taux d'exploration epsilon.

    Returns:
        int: L'index de la prochaine action.
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        # Exploration : choix d'une action aléatoire
        action = np.random.randint(0, Q.shape[1])
    else:
        # Exploitation : choix de la meilleure action
        action = np.argmax(Q[state])

    return action
