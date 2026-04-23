#!/usr/bin/env python3

import numpy as np

Deep = __import__('27-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode

lib = np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)

# Crée un nouveau modèle (poids aléatoires)
deep = Deep(784, [128, 64, 10])

# Affiche zéros placeholder
print(np.zeros((10, X_train.shape[1])))

A, cost = deep.evaluate(X_train, Y_train_one_hot)
print(cost)
print(deep.L)
for i in range(deep.L + 1):
    if i == 0:
        print("A{}".format(i), deep.cache.get("A{}".format(i)))
    else:
        print("W{}".format(i), deep.weights["W{}".format(i)])
        print("A{}".format(i), deep.cache.get("A{}".format(i)))
