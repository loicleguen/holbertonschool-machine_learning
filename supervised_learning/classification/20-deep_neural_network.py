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

        for i in range(self.__L):
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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            w_key = "W{}".format(i)
            b_key = "b{}".format(i)
            a_prev_key = "A{}".format(i - 1)
            a_key = "A{}".format(i)
            z = np.matmul(self.__weights[w_key],
                          self.__cache[a_prev_key]) + self.__weights[b_key]
            self.__cache[a_key] = 1 / (1 + np.exp(-z))
        return self.__cache[a_key], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost
