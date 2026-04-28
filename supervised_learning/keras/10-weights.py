#!/usr/bin/env python3
"""Save and load a Keras model's weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Saves a Keras model's weights"""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """Loads a Keras model's weights"""
    network.load_weights(filename)
