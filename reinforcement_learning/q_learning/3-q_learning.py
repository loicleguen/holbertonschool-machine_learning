#!/usr/bin/env python3
"""Module pour l'entraînement d'un agent avec Q-Learning."""
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Entraîne un agent avec l'algorithme Q-learning.

    Args:
        env: L'instance de l'environnement FrozenLakeEnv.
        Q (numpy.ndarray): La Q-table initiale.
        episodes (int): Nombre total d'épisodes d'entraînement.
        max_steps (int): Nombre maximum d'étapes par épisode.
        alpha (float): Taux d'apprentissage.
        gamma (float): Facteur de réduction (discount factor).
        epsilon (float): Taux d'exploration initial.
        min_epsilon (float): Valeur minimale de epsilon.
        epsilon_decay (float): Taux de décroissance de epsilon.

    Returns:
        tuple: (Q, total_rewards) où Q est la Q-table mise à jour et
        total_rewards est la liste des récompenses par épisode.
    """
    total_rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Si l'agent tombe dans un trou
            if terminated and reward == 0:
                reward = -1

            # Équation de Bellman pour la mise à jour de Q
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        # Gestion de la récompense totale reçue dans l'épisode
        if total_reward == -1:
            total_reward = 0

        total_rewards.append(total_reward)

        # Décroissance d'epsilon
        epsilon = min_epsilon + (
            initial_epsilon - min_epsilon
        ) * np.exp(-epsilon_decay * episode)

    return Q, total_rewards
