#!/usr/bin/env python3
"""Module permettant d'ajuster aléatoirement le contraste d'une image."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Ajuste aléatoirement le contraste d'une image dans une plage donnée.

    Args:
        image (tf.Tensor): Tenseur 3D représentant l'image d'entrée dont il
            faut modifier le contraste.
        lower (float): Borne inférieure de la plage du facteur de contraste
            aléatoire.
        upper (float): Borne supérieure de la plage du facteur de contraste
            aléatoire.

    Returns:
        tf.Tensor: Le tenseur 3D de l'image avec son contraste ajusté.
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)
