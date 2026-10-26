#!/usr/bin/env python3
"""
Module pour entraîner un agent avec l'algorithme Monte-Carlo Policy Gradient.
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Entraîne un agent en utilisant la méthode Monte-Carlo Policy Gradient.

    Args:
        env: L'environnement Gymnasium à entraîner.
        nb_episodes (int): Le nombre d'épisodes pour l'entraînement.
        alpha (float): Le taux d'apprentissage (learning rate).
        gamma (float): Le facteur de réduction (discount factor).

    Returns:
        list: La liste contenant le score (somme des récompenses)
              pour chaque épisode.
    """
    # Initialisation des poids de manière aléatoire
    # Taille d'état x Nombre d'actions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    weight = np.random.rand(state_dim, action_dim)

    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        score = 0
        done = False

        # Phase de génération de la trajectoire
        while not done:
            action, grad = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state

        scores.append(score)

        # Calcul des retours actualisés (discounted returns G_t) et mise à jour
        for t in range(len(rewards)):
            # G_t = sum_{k=0}^{T - t - 1} gamma^k * R_{t + k + 1}
            G_t = sum(
                [r * (gamma ** k) for k, r in enumerate(rewards[t:])]
            )
            weight += alpha * grads[t] * G_t

        # Affichage du score de l'épisode
        print(f"Episode: {episode} Score: {score}")

    return scores
