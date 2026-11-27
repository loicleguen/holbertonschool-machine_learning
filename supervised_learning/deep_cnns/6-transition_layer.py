#!/usr/bin/env python3
"""Module permettant de construire une couche de transition pour DenseNet."""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Construit une couche de transition telle que décrite dans l'article
    'Densely Connected Convolutional Networks' (avec compression DenseNet-C).

    Args:
        X: Sortie de la couche précédente (tenseur Keras).
        nb_filters: Entier représentant le nombre de filtres dans X.
        compression: Facteur de compression (flottant entre 0 et 1).

    Returns:
        Un tuple (X, nb_filters) contenant la sortie de la couche de transition
        et le nouveau nombre de filtres en sortie.
    """
    init = K.initializers.he_normal(seed=0)
    nb_filters = int(nb_filters * compression)

    # Normalisation par lots puis activation ReLU
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation('relu')(X)

    # Convolution 1x1 pour réduire le nombre de canaux (compression)
    X = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(X)

    # Pooling moyen 2x2 (stride 2) pour diviser les dimensions spatiales par 2
    X = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(X)

    return X, nb_filters
