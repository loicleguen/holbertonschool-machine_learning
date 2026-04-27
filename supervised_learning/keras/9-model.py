#!/usr/bin/env python3
"""Save and load a Keras model"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves a Keras model"""
    network.save(filename)


def load_model(filename):
    """Loads a Keras model"""
    return K.models.load_model(filename)
