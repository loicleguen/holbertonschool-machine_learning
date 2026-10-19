#!/usr/bin/env python3
"""
Module pour l'algorithme SARSA(lambda).
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Sélectionne une action en utilisant la politique epsilon-greedy.

    Paramètres :
    - Q : La table Q de forme (s, a)
    - state : L'état courant
    - epsilon : Le seuil d'exploration

    Retourne :
    - action : L'action choisie
    """
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Exécute l'algorithme SARSA(lambda) pour mettre à jour la table Q.

    Paramètres :
    - env : L'instance de l'environnement Gym/Gymnasium
    - Q : Un tableau numpy.ndarray de forme (s, a) contenant la table Q
    - lambtha : Le facteur de trace d'éligibilité
    - episodes : Le nombre total d'épisodes pour l'entraînement
    - max_steps : Le nombre maximum d'étapes par épisode
    - alpha : Le taux d'apprentissage (learning rate)
    - gamma : Le facteur de remise (discount rate)
    - epsilon : Le seuil initial pour la politique epsilon-greedy
    - min_epsilon : La valeur minimale vers laquelle epsilon doit décroître
    - epsilon_decay : Le taux de décroissance d'epsilon entre les épisodes

    Retourne :
    - Q : La table Q mise à jour
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()[0]
        # Initialisation de la trace d'éligibilité pour chaque épisode
        E = np.zeros_like(Q)

        # Sélection de la première action avec la politique epsilon-greedy
        action = epsilon_greedy(Q, state, epsilon)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Sélection de la prochaine action A' (SARSA est On-Policy)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            # Calcul de l'erreur TD (delta)
            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])

            # Accumulation de la trace pour le couple (état, action) courant
            E[state, action] += 1

            # Mise à jour de toute la table Q
            Q += alpha * delta * E

            # Décroissance des traces d'éligibilité
            E *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        # Décroissance d'epsilon après chaque épisode
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )

    return Q
