#!/usr/bin/env python3
"""
Entraîne un agent DQN sur l'environnement Atari Breakout de Gymnasium.
"""
import os
import shutil
import sys
import gymnasium as gym
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras

# Patch de compatibilité pour keras-rl2 avec TensorFlow 2.15+
setattr(tf.keras, '__version__', tf.__version__)
setattr(keras, '__version__', tf.__version__)
sys.modules['tensorflow.keras'].__version__ = tf.__version__

from keras.layers import Conv2D, Dense, Flatten, Permute  # noqa: E402
from keras.models import Sequential  # noqa: E402
from rl.agents.dqn import DQNAgent  # noqa: E402
from rl.callbacks import ModelIntervalCheckpoint  # noqa: E402
from rl.core import Processor  # noqa: E402
from rl.memory import SequentialMemory  # noqa: E402
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy  # noqa: E402

# Utilisation de l'optimiseur legacy compatible avec keras-rl2
Adam = tf.keras.optimizers.legacy.Adam


class KerasRLGymWrapper(gym.Wrapper):
    """Wrapper Gymnasium pour rendre l'environnement
       compatible avec keras-rl2."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        """Réinitialise l'environnement et renvoie uniquement l'obs."""
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """Exécute l'action et renvoie 4 valeurs
           selon la signature classique Gym."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, mode='human'):
        """Rend la méthode render() compatible avec keras-rl2."""
        return self.env.render()


def make_env(render=False, wrapper=KerasRLGymWrapper):
    """Crée l'environnement Breakout prêt pour keras-rl2."""
    env = gym.make('BreakoutNoFrameskip-v4', frameskip=4,
                   render_mode='human' if render else None)
    return wrapper(env)


class AtariProcessor(Processor):
    """Processeur de pré-traitement des images d'Atari."""

    def process_observation(self, observation):
        """Redimensionne et convertit en niveaux de gris l'image Atari."""
        img = Image.fromarray(observation)
        img = img.convert('L').resize((84, 84))
        return np.array(img, dtype=np.uint8)

    def process_state_batch(self, batch):
        """Normalise les valeurs de pixels entre 0 et 1."""
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Limite les récompenses entre -1 et 1 (reward clipping)."""
        return np.clip(reward, -1., 1.)


class StepTrackerCheckpoint(ModelIntervalCheckpoint):
    """Callback personnalisé étendant ModelIntervalCheckpoint
       pour sauvegarder dynamiquement le nombre de steps effectués."""

    def __init__(self, filepath, interval, step_file, initial_steps=0,
                 verbose=0):
        super().__init__(filepath, interval=interval, verbose=verbose)
        self.step_file = step_file
        self.initial_steps = initial_steps
        os.makedirs(os.path.dirname(step_file), exist_ok=True)

    def on_step_end(self, step, logs={}):
        """Sauvegarde les poids à chaque intervalle, puis note le
           nombre total de steps réalisés dans le fichier de suivi."""
        super().on_step_end(step, logs)
        if self.total_steps % self.interval == 0:
            with open(self.step_file, 'w') as f:
                f.write(str(self.initial_steps + self.total_steps))


def create_model(input_shape, nb_actions):
    """Crée l'architecture CNN (style DeepMind) pour le DQN."""
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def build_agent(env, policy, test_policy=None, memory_limit=200000):
    """Construit et compile l'agent DQN (partagé par train et play)."""
    window_length = 4
    nb_actions = env.action_space.n
    model = create_model((window_length, 84, 84), nb_actions)
    memory = SequentialMemory(limit=memory_limit,
                              window_length=window_length)
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        test_policy=test_policy,
        memory=memory,
        processor=AtariProcessor(),
        nb_steps_warmup=5000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    return dqn


def train():
    """Entraîne l'agent DQN sur Breakout."""
    total_steps = 1000000
    checkpoint_dir = 'checkpoint'
    checkpoint_path = os.path.join(checkpoint_dir,
                                   'dqn_checkpoint_weights.h5f')
    step_file = os.path.join(checkpoint_dir, 'step_count.txt')
    checkpoint_interval = 100000

    # Lecture dynamique du nombre de steps sauvegardés
    steps_already_done = 0
    if os.path.exists(step_file):
        try:
            with open(step_file, 'r') as f:
                steps_already_done = int(f.read().strip())
            print(f"--> Checkpoint trouvé ! Reprise dynamique depuis "
                  f"{steps_already_done} steps.")
        except ValueError:
            steps_already_done = 0
    else:
        print("--> Aucun fichier de suivi trouvé. Démarrage à 0 step.")

    resume = len(sys.argv) > 1 and sys.argv[1] == 'resume'
    env = make_env()
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps',
        value_max=0.3 if resume else 1.0, value_min=0.1,
        value_test=0.05, nb_steps=total_steps // 2)
    dqn = build_agent(env, policy)
    if resume and steps_already_done == 0:
        dqn.load_weights('policy.h5')
        print("--> Reprise depuis policy.h5")

    # Application du checkpoint si présent
    if steps_already_done > 0 and (os.path.exists(checkpoint_path) or
                                   os.path.exists(checkpoint_path + '.index')):
        dqn.load_weights(checkpoint_path)
        print("--> Poids du checkpoint chargés avec succès !")

    # --- Entraînement ---
    remaining_steps = total_steps - steps_already_done

    if remaining_steps > 0:
        print(f"=== Entraînement ({remaining_steps} steps restants) ===")
        checkpoint_callback = StepTrackerCheckpoint(
            checkpoint_path,
            interval=checkpoint_interval,
            step_file=step_file,
            initial_steps=steps_already_done
        )
        dqn.fit(
            env,
            nb_steps=remaining_steps,
            callbacks=[checkpoint_callback],
            visualize=False,
            verbose=2)
    env.close()

    # Sauvegarde finale du modèle
    dqn.save_weights('policy.h5', overwrite=True)

    # Nettoyage du dossier checkpoint
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print("--> Dossier checkpoint nettoyé avec succès.")

    print("=== Entraînement terminé ! Poids sauvegardés dans policy.h5 ===")


if __name__ == '__main__':
    train()
