#!/usr/bin/env python3
"""
Module pour l'algorithme TD(lambda).
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Exécute l'algorithme TD(lambda) pour estimer la valeur
    d'un état dans un environnement.

    Paramètres :
    - env : L'instance de l'environnement Gym/Gymnasium
    - V : Un tableau numpy.ndarray de forme (s,) contenant l'estimation
    - policy : La fonction de politique qui prend un état et renvoie une action
    - lambtha : Le facteur de trace d'éligibilité
    - episodes : Le nombre total d'épisodes pour l'entraînement
    - max_steps : Le nombre maximum d'étapes par épisode
    - alpha : Le taux d'apprentissage (learning rate)
    - gamma : Le facteur de remise (discount rate)

    Retourne :
    - V : L'estimation de la valeur mise à jour
    """
    n_states = V.shape[0]

    for episode in range(episodes):
        state = env.reset()[0]
        # Initialisation de la trace d'éligibilité pour chaque épisode
        E = np.zeros(n_states)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Calcul de l'erreur TD (delta)
            delta = reward + gamma * V[next_state] - V[state]

            # Accumulation de la trace pour l'état courant
            E[state] += 1

            # Mise à jour de la valeur de tous les états pondérée par la trace
            V = V + alpha * delta * E

            # Atténuation des traces d'éligibilité
            E = E * gamma * lambtha

            if terminated or truncated:
                break

            state = next_state

    return V
