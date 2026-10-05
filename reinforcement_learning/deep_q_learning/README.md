<div align="center"><img src="https://github.com/ksyv/holbertonschool-web_front_end/blob/main/baniere_holberton.png?raw=true"></div>

# Deep Q-learning

Entraînement d'un agent DQN (Deep Q-Network) capable de jouer à Atari Breakout, avec `keras`, `keras-rl2` et `gymnasium`.

## Table of Contents

- [Learning Objectives](#learning-objectives)
- [Requirements](#requirements)
- [Installation](#installation)
- [0. Breakout](#0-breakout)
- [Résultat](#résultat)
- [Authors](#authors)

## Learning Objectives

À la fin de ce projet, on doit pouvoir expliquer sans aide :

* Qu'est-ce que le Deep Q-learning ?
* Qu'est-ce que le réseau de politique (policy network) ?
* Qu'est-ce que la mémoire de replay (replay memory) ?
* Qu'est-ce que le réseau cible (target network) ?
* Pourquoi utiliser deux réseaux séparés pendant l'entraînement ?
* Qu'est-ce que `keras-rl` et comment l'utiliser ?

## Requirements

* Éditeurs autorisés : `vi`, `vim`, `emacs`
* Ubuntu 20.04 LTS, `python3` (3.9)
* `numpy` (1.25.2), `gymnasium` (0.29.1), `keras` (2.15.0), `keras-rl2` (1.0.4)
* Tous les fichiers se terminent par une nouvelle ligne et commencent par `#!/usr/bin/env python3`
* Style de code `pycodestyle` (2.11.1)
* Modules, classes et fonctions documentés
* Tous les fichiers sont exécutables

## Installation

```bash
pip install --user keras-rl2==1.0.4
pip install --user gymnasium[atari]==0.29.1
pip install --user tensorflow==2.15.0
pip install --user keras==2.15.0
pip install --user numpy==1.25.2
pip install --user Pillow==10.3.0
pip install --user h5py==3.11.0
pip install autorom[accept-rom-license]
```

## 0. Breakout

### Fichiers

| Fichier | Rôle |
|---------|------|
| `train.py` | Entraîne l'agent et sauvegarde le réseau dans `policy.h5` |
| `play.py` | Charge `policy.h5` et affiche des parties jouées par l'agent |
| `policy.h5` | Poids du réseau entraîné |

### Utilisation

```bash
./train.py          # entraînement de zéro (1 000 000 de steps)
./train.py resume   # reprise à partir de policy.h5
./play.py           # affiche 5 parties de l'agent entraîné
```

Attention : l'entraînement écrase `policy.h5` à la fin (ou si on l'interrompt avec `Ctrl+C`). Pensez à en faire une copie avant de relancer.

### Choix techniques

**Environnement**
* `BreakoutNoFrameskip-v4` avec `frameskip=4` : l'agent choisit une action toutes les 4 images.
* Un wrapper `gymnasium` adapte `reset`, `step` et `render` à l'API attendue par `keras-rl2`.

**Prétraitement** (`AtariProcessor`)
* Niveaux de gris, redimensionnement en 84x84.
* Fenêtre de 4 images empilées pour percevoir le mouvement de la balle.
* Normalisation des pixels entre 0 et 1.
* Reward clipping entre -1 et 1.

**Réseau** (`create_model`)
* CNN de type DeepMind : 3 couches de convolution (32, 64, 64 filtres), une couche dense de 512 neurones, puis une sortie linéaire (une valeur Q par action).

**Agent** (`build_agent`)
* `DQNAgent` avec `SequentialMemory` (200 000 expériences), `gamma=0.99`, `train_interval=4`, `target_model_update=10000`, 5 000 steps de warmup.
* Politique d'entraînement : `EpsGreedyQPolicy` enveloppée dans un `LinearAnnealedPolicy` (epsilon de 1.0 à 0.1 sur la première moitié de l'entraînement).
* Politique de jeu : `GreedyQPolicy`.

**Checkpoints**
* Les poids sont sauvegardés tous les 100 000 steps dans le dossier `checkpoint`, avec le nombre de steps déjà effectués. Si l'entraînement est interrompu, relancer `./train.py` reprend automatiquement depuis ce checkpoint. Le dossier est supprimé à la fin d'un entraînement réussi.
* La mémoire de replay et l'état de l'optimiseur ne sont pas sauvegardés : la reprise est donc approximative.

**Jeu** (`play.py`)
* Dans Breakout, la balle ne part que si l'agent appuie sur FIRE. Avec une politique gloutonne, l'agent peut rester bloqué. `PlayWrapper` appuie donc sur FIRE au début de la partie et après chaque vie perdue.
* La création de l'environnement et de l'agent réutilise les fonctions de `train.py`.

## Résultat

Le `policy.h5` fourni provient d'un entraînement de **1 000 000 de steps** lancé avec `./train.py`.

Avec `./play.py` (politique gloutonne) : **40 briques cassées en 1 229 steps** par partie. Les parties sont identiques d'un épisode à l'autre car la politique gloutonne est déterministe.

Un entraînement supplémentaire de 1 000 000 de steps (`./train.py resume`) a donné un agent moins bon en politique gloutonne (19 briques en 750 steps), malgré de meilleurs scores pendant l'entraînement. C'est pourquoi le modèle à 1 M de steps a été conservé.

## Authors

loicleguen - [GitHub Profile](https://github.com/loicleguen)