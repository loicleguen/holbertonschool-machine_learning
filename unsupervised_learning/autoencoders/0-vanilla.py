#!/usr/bin/env python3
"""
Module contenant la fonction autoencoder
pour construire un autoencodeur classique.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Crée un modèle d'autoencodeur classique (vanilla).

    Args:
        input_dims (int): Dimensions de la donnée d'entrée du modèle.
        hidden_layers (list): Liste contenant le nombre de nœuds pour chaque
                              couche cachée de l'encodeur.
        latent_dims (int): Dimensions de la représentation de l'espace latent.

    Returns:
        encoder (keras.Model): Le modèle de l'encodeur.
        decoder (keras.Model): Le modèle du décodeur.
        auto (keras.Model): Le modèle complet de l'autoencodeur.
    """
    # ------------------- ENCODEUR -------------------
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs, latent, name="encoder")

    # ------------------- DÉCODEUR -------------------
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    # Les couches cachées doivent être inversées pour le décodeur
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # ----------------- AUTOENCODEUR -----------------
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs, name="autoencoder")

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
