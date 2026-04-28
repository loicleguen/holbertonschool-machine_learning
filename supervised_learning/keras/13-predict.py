#!/usr/bin/env python3
"""Test a neural network model"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Tests a neural network model"""
    return network.predict(data, verbose=verbose)
