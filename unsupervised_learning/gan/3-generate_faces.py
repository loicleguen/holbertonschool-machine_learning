#!/usr/bin/env python3
"""
Module permettant de construire un générateur et un discriminateur
convolutionnels pour la génération de visages.
"""
from tensorflow import keras
from tensorflow.keras import layers


def convolutional_GenDiscr():
    """
    Construit le réseau du Générateur et du Discriminateur convolutionnels.

    Returns:
        gen: Modèle Keras représentant le Générateur.
        discr: Modèle Keras représentant le Discriminateur.
    """
    def get_generator():
        """Construit le modèle du générateur."""
        inputs = layers.Input(shape=(16,))

        # Projection du vecteur latent
        x = layers.Dense(2048)(inputs)
        x = layers.Reshape((2, 2, 512))(x)

        # Bloc 1 : Passage à (4, 4, 64)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("tanh")(x)

        # Bloc 2 : Passage à (8, 8, 16)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("tanh")(x)

        # Bloc 3 : Passage à (16, 16, 1)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(1, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Activation("tanh")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="generator")
        return model

    def get_discriminator():
        """Construit le modèle du discriminateur."""
        inputs = layers.Input(shape=(16, 16, 1))

        # Bloc 1 : Réduction à (8, 8, 32)
        x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Activation("tanh")(x)

        # Bloc 2 : Réduction à (4, 4, 64)
        x = layers.Conv2D(64, (3, 3), padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Activation("tanh")(x)

        # Bloc 3 : Réduction à (2, 2, 128)
        x = layers.Conv2D(128, (3, 3), padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Activation("tanh")(x)

        # Bloc 4 : Réduction à (1, 1, 256)
        x = layers.Conv2D(256, (3, 3), padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Activation("tanh")(x)

        # Aplatissement et sortie Dense
        x = layers.Flatten()(x)
        outputs = layers.Dense(1)(x)

        model = keras.Model(
            inputs=inputs, outputs=outputs, name="discriminator"
        )
        return model

    return get_generator(), get_discriminator()
