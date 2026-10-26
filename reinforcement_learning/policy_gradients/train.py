#!/usr/bin/env python3
"""
Module pour entraîner un agent avec l'algorithme Monte-Carlo Policy Gradient.
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Entraîne un agent en utilisant la méthode Monte-Carlo Policy Gradient.

    Args:
        env: L'environnement Gymnasium à entraîner.
        nb_episodes (int): Le nombre d'épisodes pour l'entraînement.
        alpha (float): Le taux d'apprentissage (learning rate).
        gamma (float): Le facteur de réduction (discount factor).
        show_result (bool): Si True, affiche le rendu graphique de
                            l'environnement tous les 1000 épisodes.

    Returns:
        list: La liste contenant le score (somme des récompenses)
              pour chaque épisode.
    """
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

        # Vérifie si cet épisode doit être affiché graphiquement
        render = show_result and (episode % 1000 == 0)

        while not done:
            if render:
                env.render()

            action, grad = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state

        scores.append(score)

        # Calcul des retours actualisés G_t et mise à jour des poids
        for t in range(len(rewards)):
            G_t = sum(
                [r * (gamma ** k) for k, r in enumerate(rewards[t:])]
            )
            weight += alpha * grads[t] * G_t

        print(f"Episode: {episode} Score: {score}")

    return scores
