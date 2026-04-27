#!/usr/bin/env python3
"""Builds a neural network with the Keras Sequential API"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras Sequential API"""
    model = K.Sequential()
    for i in range(len(layers)):
        model.add(K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha),
            input_shape=(nx,) if i == 0 else None
        ))

        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
