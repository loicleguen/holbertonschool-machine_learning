#!/usr/bin/env python3
"""Module permettant d'effectuer une rotation d'une image de 90 degrés."""
import tensorflow as tf


def rotate_image(image):
    """Effectue une rotation de 90 degrés dans le sens antihoraire.

    Args:
        image (tf.Tensor): Tenseur 3D contenant l'image à faire pivoter.

    Returns:
        tf.Tensor: Le tenseur 3D de l'image pivotée de 90 degrés dans le
            sens antihoraire.
    """
    return tf.image.rot90(image, k=1)
