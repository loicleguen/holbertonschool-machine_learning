#!/usr/bin/env python3
"""Module permettant de construire l'architecture DenseNet-121."""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Construit l'architecture complète du réseau DenseNet-121 telle que
    décrite dans l'article 'Densely Connected Convolutional Networks'.

    Args:
        growth_rate: Taux de croissance (k) pour les blocs denses.
        compression: Facteur de compression pour les couches de transition.

    Returns:
        Le modèle Keras du réseau DenseNet-121.
    """
    init = K.initializers.he_normal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # Nombre initial de filtres : 2 * taux de croissance
    nb_filters = 2 * growth_rate

    # Pré-activation (BN -> ReLU) puis convolution initiale 7x7 et MaxPooling
    X = K.layers.BatchNormalization(axis=-1)(X_input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    # Bloc dense 1 (6 couches) + Couche de transition 1
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Bloc dense 2 (12 couches) + Couche de transition 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Bloc dense 3 (24 couches) + Couche de transition 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Bloc dense 4 (16 couches) sans couche de transition finale
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Pooling moyen global 7x7 et couche dense de classification
    X = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )(X)
    Y = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init
    )(X)

    # Création du modèle Keras
    model = K.models.Model(inputs=X_input, outputs=Y)

    return model
