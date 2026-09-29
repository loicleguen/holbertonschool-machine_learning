#!/usr/bin/env python3
"""Module pour initialiser la Q-table."""
import numpy as np


def q_init(env):
    """Initialise la Q-table remplie de zéros.

    Args:
        env: L'instance de l'environnement FrozenLakeEnv.

    Returns:
        numpy.ndarray: La Q-table de zéros avec la forme
        (nombre d'états, nombre d'actions).
    """
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    q_table = np.zeros((state_space_size, action_space_size))

    return q_table
