#!/usr/bin/env python3
"""Test a neural network model"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network model"""
    return network.evaluate(data, labels, verbose=verbose)
