#!/usr/bin/env python3
"""Identity block for ResNet-style deep networks."""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Build an identity block as described in ResNet (2015).
    Args:
        A_prev: output from the previous layer
        filters: tuple/list of (F11, F3, F12)

    Returns:
        The activated output of the identity block.
    """
    F11, F3, F12 = filters
    he = K.initializers.he_normal(seed=0)

    X = K.layers.Conv2D(F11, (1, 1), padding='same',
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

    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
