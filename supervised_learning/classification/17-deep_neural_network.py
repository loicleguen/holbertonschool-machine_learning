#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor method for DeepNeuralNetwork class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                prev_nodes = nx
            else:
                prev_nodes = layers[i - 1]

            current_nodes = layers[i]

            w_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)

            he_factor = np.sqrt(2 / prev_nodes)
            self.__weights[w_key] = np.random.randn(
                current_nodes, prev_nodes) * he_factor
            self.__weights[b_key] = np.zeros((current_nodes, 1))

    @property
    def L(self):
        """Getter method for L attribute"""
        return self.__L

    @property
    def cache(self):
        """Getter method for cache attribute"""
        return self.__cache

    @property
    def weights(self):
        """Getter method for weights attribute"""
        return self.__weights
