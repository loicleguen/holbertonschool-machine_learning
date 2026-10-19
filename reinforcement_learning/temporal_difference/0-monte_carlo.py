#!/usr/bin/env python3
"""
Module pour l'algorithme de Monte Carlo.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Exécute la méthode de Monte Carlo pour estimer la valeur
    d'un état dans un environnement.

    Paramètres :
    - env : L'instance de l'environnement Gym/Gymnasium
    - V : Un tableau numpy.ndarray de forme (s,) contenant la valeur estimée
    - policy : La fonction de politique qui prend un état et renvoie une action
    - episodes : Le nombre total d'épisodes pour l'entraînement
    - max_steps : Le nombre maximum d'étapes par épisode
    - alpha : Le taux d'apprentissage (learning rate)
    - gamma : Le facteur de remise (discount rate)

    Retourne :
    - V : Le tableau numpy.ndarray contenant l'estimation de valeur mise à jour
    """
    for episode in range(episodes):
        state = env.reset()[0]
        episode_data = []

        for step in range(max_steps):
            action = policy(state)

            next_state, reward, terminated, truncated, _ = env.step(
                action)

            episode_data.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        G = 0
        episode_data = np.array(episode_data, dtype=int)

        for state, reward in reversed(episode_data):
            G = reward + gamma * G

            if state not in episode_data[:episode, 0]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
