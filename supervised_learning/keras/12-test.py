#!/usr/bin/env python3
"""Test a neural network model"""


def test_model(network, data, labels, verbose=True):
    """Tests a neural network model"""
    return network.evaluate(data, labels, verbose=verbose)
