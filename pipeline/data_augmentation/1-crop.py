#!/usr/bin/env python3
"""Module permettant d'effectuer un recadrage aléatoire d'une image."""
import tensorflow as tf


def crop_image(image, size):
    """Effectue un recadrage (crop) aléatoire d'une image.

    Args:
        image (tf.Tensor): Tenseur 3D contenant l'image à recadrer.
        size (tuple): Tuple contenant les dimensions souhaitées du recadrage
            au format (hauteur, largeur, canaux).

    Returns:
        tf.Tensor: Le tenseur 3D de l'image recadrée aléatoirement.
    """
    return tf.image.random_crop(image, size=size)
