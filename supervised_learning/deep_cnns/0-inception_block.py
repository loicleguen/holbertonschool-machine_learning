#!/usr/bin/env python3
"""Module permettant de construire un bloc Inception (GoogLeNet)."""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Construit un bloc Inception tel que décrit dans l'article
    'Going Deeper with Convolutions' (2014).

    Args:
        A_prev: Sortie de la couche précédente (tenseur Keras).
        filters: Tuple ou liste contenant (F1, F3R, F3, F5R, F5, FPP) :
            - F1: Nombre de filtres de la convolution 1x1.
            - F3R: Nombre de filtres 1x1 avant la convolution 3x3.
            - F3: Nombre de filtres de la convolution 3x3.
            - F5R: Nombre de filtres 1x1 avant la convolution 5x5.
            - F5: Nombre de filtres de la convolution 5x5.
            - FPP: Nombre de filtres 1x1 après le max pooling.

    Returns:
        Le tenseur de sortie concaténé du bloc Inception.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Branche 1 : Convolution 1x1
    conv1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    # Branche 2 : Réduction 1x1 suivie d'une convolution 3x3
    conv3_reduce = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)
    conv3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(conv3_reduce)

    # Branche 3 : Réduction 1x1 suivie d'une convolution 5x5
    conv5_reduce = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)
    conv5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )(conv5_reduce)

    # Branche 4 : MaxPooling 3x3 suivi d'une projection 1x1
    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)
    pool_proj = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(pool)

    # Concaténation de toutes les branches sur l'axe des filtres (canaux)
    output = K.layers.Concatenate(axis=-1)([conv1, conv3, conv5, pool_proj])

    return output
