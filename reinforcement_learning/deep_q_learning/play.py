#!/usr/bin/env python3
"""
Affiche des parties d'Atari Breakout jouées par l'agent DQN entraîné,
en utilisant les poids sauvegardés dans policy.h5.
"""
from train import KerasRLGymWrapper, build_agent, make_env
from rl.policy import GreedyQPolicy


class PlayWrapper(KerasRLGymWrapper):
    """Relance la balle (FIRE) au début et après chaque vie perdue."""

    FIRE = 1

    def reset(self, **kwargs):
        """Réinitialise la partie et lance la balle."""
        super().reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        obs, _, _, _ = super().step(self.FIRE)
        return obs

    def step(self, action):
        """Exécute l'action ; si une vie est perdue, relance la balle."""
        obs, reward, done, info = super().step(action)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and not done:
            obs, extra, done, info = super().step(self.FIRE)
            reward += extra
        self.lives = lives
        return obs, reward, done, info


def play():
    """Charge policy.h5 et joue avec une politique gloutonne."""
    env = make_env(render=True, wrapper=PlayWrapper)
    dqn = build_agent(env, GreedyQPolicy(), GreedyQPolicy(),
                      memory_limit=1000)
    dqn.load_weights('policy.h5')
    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()


if __name__ == '__main__':
    play()
