#!/usr/bin/env python3
"""LeNet-5 model using Keras."""
from tensorflow import keras as K


def lenet5(X):
    """Builds a modified LeNet-5 model.

    X: K.Input of shape (m, 28, 28, 1)
    Returns: compiled K.Model"""
    init = K.initializers.he_normal(seed=0)

    layer = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    layer = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(layer)

    layer = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(layer)

    layer = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(layer)

    layer = K.layers.Flatten()(layer)

    layer = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(layer)

    layer = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(layer)

    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(layer)

    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
