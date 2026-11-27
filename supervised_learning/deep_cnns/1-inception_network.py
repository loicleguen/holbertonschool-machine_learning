#!/usr/bin/env python3
"""Module permettant de construire le réseau Inception (GoogLeNet)."""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Construit l'architecture complète du réseau Inception (GoogLeNet)
    telle que décrite dans 'Going Deeper with Convolutions' (2014).

    Returns:
        Le modèle Keras compilable du réseau Inception.
    """
    X_input = K.Input(shape=(224, 224, 3))

    # Couches initiales (Stem)
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        activation='relu'
    )(X_input)
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )(X)
    X = K.layers.Conv2D(
        filters=192,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )(X)
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    # Blocs Inception 3a et 3b
    X = inception_block(X, [64, 96, 128, 16, 32, 32])
    X = inception_block(X, [128, 128, 192, 32, 96, 64])
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    # Blocs Inception 4a à 4e
    X = inception_block(X, [192, 96, 208, 16, 48, 64])
    X = inception_block(X, [160, 112, 224, 24, 64, 64])
    X = inception_block(X, [128, 128, 256, 24, 64, 64])
    X = inception_block(X, [112, 144, 288, 32, 64, 64])
    X = inception_block(X, [256, 160, 320, 32, 128, 128])
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    # Blocs Inception 5a et 5b
    X = inception_block(X, [256, 160, 320, 32, 128, 128])
    X = inception_block(X, [384, 192, 384, 48, 128, 128])

    # Pooling moyen global, Dropout et couche de classification
    X = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )(X)
    X = K.layers.Dropout(rate=0.4)(X)
    Y = K.layers.Dense(units=1000, activation='softmax')(X)

    # Création du modèle Keras
    model = K.models.Model(inputs=X_input, outputs=Y)

    return model
