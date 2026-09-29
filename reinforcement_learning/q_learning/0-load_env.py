#!/usr/bin/env python3
"""Module pour charger l'environnement FrozenLake de Gymnasium."""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Charge l'environnement pré-fait FrozenLakeEnv de Gymnasium.

    Args:
        desc (list, optional): Description personnalisée de la carte.
            Defaults to None.
        map_name (str, optional): Nom d'une carte pré-faite ('4x4', '8x8').
            Defaults to None.
        is_slippery (bool, optional): Indique si la glace est glissante.
            Defaults to False.

    Returns:
        gym.Env: L'environnement FrozenLake initialisé.
    """
    if desc is None and map_name is None:
        map_name = "8x8"

    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"
    )
    return env
