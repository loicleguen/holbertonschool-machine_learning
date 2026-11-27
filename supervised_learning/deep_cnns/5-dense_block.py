#!/usr/bin/env python3
"""Module permettant de construire un bloc dense pour DenseNet-B."""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Construit un bloc dense avec couches goulot d'étranglement (DenseNet-B)
    tel que décrit dans l'article 'Densely Connected Convolutional Networks'.

    Args:
        X: Sortie de la couche précédente (tenseur Keras).
        nb_filters: Entier représentant le nombre de filtres dans X.
        growth_rate: Taux de croissance (k) pour le bloc dense.
        layers: Nombre de couches dans le bloc dense.

    Returns:
        Un tuple (X, nb_filters) contenant la sortie concaténée de chaque
        couche du bloc dense et le nombre total de filtres en sortie.
    """
    init = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        # Couche goulot d'étranglement : BN -> ReLU -> Conv 1x1 (4 * k filtres)
        bn1 = K.layers.BatchNormalization(axis=-1)(X)
        act1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=init
        )(act1)

        # Convolution principale : BN -> ReLU -> Conv 3x3 (k filtres)
        bn2 = K.layers.BatchNormalization(axis=-1)(conv1)
        act2 = K.layers.Activation('relu')(bn2)
        conv3 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=init
        )(act2)

        # Concaténation de l'entrée précédente avec la nouvelle sortie
        X = K.layers.Concatenate(axis=-1)([X, conv3])
        nb_filters += growth_rate

    return X, nb_filters
