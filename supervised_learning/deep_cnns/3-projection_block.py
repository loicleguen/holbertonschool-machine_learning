#!/usr/bin/env python3
"""Projection block for ResNet-style deep networks."""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Build a projection block as described in ResNet (2015).

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing (F11, F3, F12)
        s: stride of the first convolution in the main path and shortcut

    Returns:
        The activated output of the projection block.
    """
    F11, F3, F12 = filters
    he = K.initializers.he_normal(seed=0)

    # Main path
    X = K.layers.Conv2D(F11, (1, 1), strides=(s, s), padding='same',
                        kernel_initializer=he)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=he)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=he)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    shortcut = K.layers.Conv2D(F12, (1, 1), strides=(s, s), padding='same',
                               kernel_initializer=he)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Merge
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
