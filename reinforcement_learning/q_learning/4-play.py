#!/usr/bin/env python3
"""Module pour faire jouer l'agent entraîné."""
import numpy as np


def play(env, Q, max_steps=100):
    """Fait jouer l'agent dans l'environnement en utilisant la Q-table.

    Args:
        env: L'instance de l'environnement FrozenLakeEnv.
        Q (numpy.ndarray): La Q-table entraînée.
        max_steps (int): Le nombre maximum d'étapes dans l'épisode.

    Returns:
        tuple: (total_reward, rendered_outputs)
            total_reward: La récompense totale de l'épisode.
            rendered_outputs: La liste des représentations visuelles
            du plateau à chaque étape.
    """
    rendered_outputs = []
    state, _ = env.reset()

    # Capture de l'état initial
    rendered_outputs.append(env.render())

    total_reward = 0

    for step in range(max_steps):
        # Exploitation pure (meilleure action d'après la Q-table)
        action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Capture de l'affichage après l'action
        rendered_outputs.append(env.render())

        total_reward += reward
        state = next_state

        if terminated or truncated:
            break

    return total_reward, rendered_outputs
