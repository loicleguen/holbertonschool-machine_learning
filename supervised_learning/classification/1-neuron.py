#!/usr/bin/env python3
"""Defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """Class Neuron that defines a single
    neuron performing binary classification"""
    def __init__(self, nx):
        """Constructor method for the Neuron class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter method for the weights vector W"""
        return self.__W

    @property
    def b(self):
        """Getter method for the bias b"""
        return self.__b

    @property
    def A(self):
        """Getter method for the activated output A"""
        return self.__A
