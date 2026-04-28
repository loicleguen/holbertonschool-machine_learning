#!/usr/bin/env python3
"""Save and load a model's configuration in JSON format"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration in JSON format"""
    json_config = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_config)


def load_config(filename):
    """Loads a model's configuration in JSON format"""
    with open(filename, 'r') as f:
        json_config = f.read()
    return K.models.model_from_json(json_config)
