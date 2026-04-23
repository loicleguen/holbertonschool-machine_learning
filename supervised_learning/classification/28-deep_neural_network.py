#!/usr/bin/env python3
"""Defines a deep neural network performing multiclass classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Deep Neural Network performing multiclass classification"""

    def __init__(self, nx, layers, activation='sig'):
        """Constructor method for DeepNeuralNetwork class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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

    @property
    def activation(self):
        """Getter method for activation attribute"""
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self._DeepNeuralNetwork__cache["A0"] = X

        for i in range(1, self._DeepNeuralNetwork__L + 1):
            w_key = "W{}".format(i)
            b_key = "b{}".format(i)
            a_prev_key = "A{}".format(i - 1)
            a_key = "A{}".format(i)
            z = np.matmul(self._DeepNeuralNetwork__weights[w_key],
                          self._DeepNeuralNetwork__cache[a_prev_key]) + \
                self._DeepNeuralNetwork__weights[b_key]
            if i == self._DeepNeuralNetwork__L:
                # Softmax activation for the output layer
                e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
                self._DeepNeuralNetwork__cache[a_key] = (
                    e_z / np.sum(e_z, axis=0, keepdims=True))
            else:
                # Sigmoid activation for hidden layers
                if self.__activation == 'sig':
                    self._DeepNeuralNetwork__cache[a_key] = 1 / (1 + np.exp(-z))
                else:
                    self._DeepNeuralNetwork__cache[a_key] = np.tanh(z)
        return (self._DeepNeuralNetwork__cache[a_key],
                self._DeepNeuralNetwork__cache)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A, _ = self.forward_prop(X)
        predictions = np.zeros_like(A)
        predictions[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            w_key = "W{}".format(i)
            b_key = "b{}".format(i)
            A_prev_key = "A{}".format(i - 1)

            dW = np.matmul(dZ, cache[A_prev_key].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 1:
                dA = np.matmul(self.__weights[w_key].T, dZ)
                if self.__activation == 'sig':
                    dZ = dA * (cache[A_prev_key] * (1 - cache[A_prev_key]))
                else:
                    dZ = dA * (1 - cache[A_prev_key] ** 2)

            self.__weights[w_key] -= alpha * dW
            self.__weights[b_key] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int) or isinstance(step, bool):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps_list = []

        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        if verbose:
            print("Cost after 0 iterations: {}".format(cost))
        if graph:
            costs.append(cost)
            steps_list.append(0)

        for i in range(1, iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if i % step == 0:
                A, cache = self.forward_prop(X)
                cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    costs.append(cost)
                    steps_list.append(i)

        if iterations % step != 0:
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            if verbose:
                print("Cost after {} iterations: {}".format(
                    iterations - 1, cost))
            if graph:
                costs.append(cost)
                steps_list.append(iterations)

        if graph:
            plt.figure()
            plt.plot(steps_list, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
