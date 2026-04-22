#!/usr/bin/env python3
"""Defines a neural network with one hidden
layer performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """Class that defines a neural network with one hidden"""
    def __init__(self, nx, nodes):
        """Constructor method for the NeuralNetwork class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method for W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter method for b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter method for A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter method for W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter method for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter method for A2"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        self.forward_prop(X)
        A2 = self.__A2
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""
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
        step_list = []

        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        if verbose:
            print("Cost after 0 iterations: {}".format(cost))
        if graph:
            costs.append(cost)
            step_list.append(0)

        for i in range(1, iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

            if i % step == 0:
                A1, A2 = self.forward_prop(X)
                cost = self.cost(Y, A2)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    costs.append(cost)
                    step_list.append(i)

        if iterations % step != 0:
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)
            if verbose:
                print("Cost after {} iterations: {}".format(iterations, cost))
            if graph:
                costs.append(cost)
                step_list.append(iterations)

        if graph:
            plt.figure()
            plt.plot(step_list, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
