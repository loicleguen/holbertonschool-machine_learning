#!/usr/bin/env python3
"""Module permettant de modifier la teinte (hue) d'une image."""
import tensorflow as tf


def change_hue(image, delta):
    """Modifie la teinte d'une image selon un décalage donné.

    Args:
        image (tf.Tensor): Tenseur 3D contenant l'image à modifier.
        delta (float): Valeur de décalage à appliquer à la teinte (doit être
            comprise dans l'intervalle [-1, 1]).

    Returns:
        tf.Tensor: Le tenseur 3D de l'image avec sa teinte modifiée.
    """
    return tf.image.adjust_hue(image, delta=delta)
