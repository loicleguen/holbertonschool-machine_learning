#!/usr/bin/env python3
"""Module permettant d'effectuer un retournement horizontal d'une image."""
import tensorflow as tf


def flip_image(image):
    """Retourne une image horizontalement (de gauche à droite).

    Args:
        image (tf.Tensor): Tenseur 3D contenant l'image à retourner.

    Returns:
        tf.Tensor: Le tenseur 3D de l'image retournée horizontalement.
    """
    return tf.image.flip_left_right(image)
