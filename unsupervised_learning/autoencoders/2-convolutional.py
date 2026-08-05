#!/usr/bin/env python3
"""
Module contenant la fonction autoencoder pour construire un autoencodeur
convolutionnel (Convolutional Autoencoder).
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Crée un modèle d'autoencodeur convolutionnel.

    Args:
        input_dims (tuple): Dimensions de l'entrée du modèle (h, w, c).
        filters (list): Liste du nombre de filtres pour chaque couche
                        de convolution dans l'encodeur.
        latent_dims (tuple): Dimensions de l'espace latent (h_l, w_l, c_l).

    Returns:
        encoder (keras.Model): Le modèle de l'encodeur.
        decoder (keras.Model): Le modèle du décodeur.
        auto (keras.Model): Le modèle complet de l'autoencodeur convolutionnel.
    """
    # ------------------- ENCODEUR -------------------
    inputs = keras.Input(shape=input_dims)
    x = inputs

    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    latent = x
    encoder = keras.Model(inputs, latent, name="encoder")

    # ------------------- DÉCODEUR -------------------
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs

    reversed_filters = list(reversed(filters))

    # Convolutions du décodeur sauf les deux dernières
    for f in reversed_filters[:-1]:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Avant-dernière convolution (filtre avec padding 'valid' + UpSampling)
    second_to_last_filter = reversed_filters[-1]
    x = keras.layers.Conv2D(
        filters=second_to_last_filter,
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Dernière convolution (canaux d'origine,
    # padding 'same', sigmoid, pas de UpSampling)
    output_channels = input_dims[-1]
    outputs = keras.layers.Conv2D(
        filters=output_channels,
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # ----------------- AUTOENCODEUR -----------------
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs, name="autoencoder")

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
