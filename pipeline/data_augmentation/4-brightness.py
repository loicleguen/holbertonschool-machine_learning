#!/usr/bin/env python3
"""Module permettant de modifier aléatoirement la luminosité d'une image."""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Modifie aléatoirement la luminosité d'une image.

    Args:
        image (tf.Tensor): Tenseur 3D contenant l'image à modifier.
        max_delta (float): Valeur maximale d'éclaircissement (ou
            d'assombrissement) à appliquer à l'image (doit être >= 0).

    Returns:
        tf.Tensor: Le tenseur 3D de l'image avec sa luminosité modifiée.
    """
    return tf.image.random_brightness(image, max_delta=max_delta)
