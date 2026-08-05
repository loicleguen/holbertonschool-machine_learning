#!/usr/bin/env python3
"""
Module contenant la fonction autoencoder pour construire un autoencodeur
variationnel (Variational Autoencoder - VAE).
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Crée un modèle d'autoencodeur variationnel (VAE).

    Args:
        input_dims (int): Dimensions de la donnée d'entrée du modèle.
        hidden_layers (list): Liste contenant le nombre de nœuds pour chaque
                              couche cachée de l'encodeur.
        latent_dims (int): Dimensions de la représentation de l'espace latent.

    Returns:
        encoder (keras.Model): Le modèle de l'encodeur produisant
                               (z, z_mean, z_log_var).
        decoder (keras.Model): Le modèle du décodeur.
        auto (keras.Model): Le modèle complet du VAE.
    """
    # ------------------- ENCODEUR -------------------
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Couche de reparamétrisation (Sampling Trick)
    def sampling(args):
        mean, log_var = args
        batch = keras.backend.shape(mean)[0]
        dim = keras.backend.int_shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return mean + keras.backend.exp(0.5 * log_var) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name="encoder")

    # ------------------- DÉCODEUR -------------------
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # ----------------- AUTOENCODEUR -----------------
    z_sampled, mean_out, log_var_out = encoder(inputs)
    reconstructed_outputs = decoder(z_sampled)

    auto = keras.Model(inputs, reconstructed_outputs, name="autoencoder")

    # ---------------- FONCTION DE PERTE ------------------
    # 1. Perte de reconstruction (Binary Cross Entropy sommee sur l'image)
    def reconstruction_loss(x_true, x_pred):
        bce = keras.losses.binary_crossentropy(x_true, x_pred)
        return bce * input_dims

    # 2. Divergence KL (Ajoutee directement aux pertes du modele)
    kl_loss = -0.5 * keras.backend.sum(
        1 + log_var_out - keras.backend.square(mean_out) -
        keras.backend.exp(log_var_out),
        axis=-1
    )
    auto.add_loss(keras.backend.mean(kl_loss))

    # Compilation avec la perte de reconstruction uniquement
    # (Keras additionne automatiquement la perte de add_loss lors du fit)
    auto.compile(optimizer='adam', loss=reconstruction_loss)

    return encoder, decoder, auto
