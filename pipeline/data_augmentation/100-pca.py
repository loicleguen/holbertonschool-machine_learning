#!/usr/bin/env python3
"""Module permettant d'effectuer une augmentation des couleurs par ACP."""
import tensorflow as tf


def pca_color(image, alphas):
    """Effectue une augmentation des couleurs par ACP (méthode AlexNet).

    Args:
        image (tf.Tensor): Tenseur 3D contenant l'image à modifier.
        alphas (tuple): Tuple (ou tableau) de longueur 3 contenant les
            facteurs de variation pour chaque canal principal.

    Returns:
        tf.Tensor: Le tenseur 3D de l'image avec ses couleurs augmentées.
    """
    orig_dtype = image.dtype
    # Convertir et normaliser les pixels dans [0, 1]
    img_float = tf.cast(image, tf.float32) / 255.0

    # Aplatir l'image en une matrice de pixels (N, 3)
    shape = tf.shape(img_float)
    num_pixels = tf.cast(shape[0] * shape[1], tf.float32)
    pixels = tf.reshape(img_float, [-1, 3])

    # Centrer les canaux RGB autour de leur moyenne
    mean = tf.reduce_mean(pixels, axis=0, keepdims=True)
    centered_pixels = pixels - mean

    # Calculer la matrice de covariance 3x3 des canaux de couleur
    cov = tf.matmul(centered_pixels, centered_pixels, transpose_a=True) / (
        num_pixels - 1.0
    )

    # Extraire les valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = tf.linalg.eigh(cov)

    # Calculer le vecteur de perturbation : P * (alpha * lambda)
    alphas_tensor = tf.cast(alphas, tf.float32)
    delta = tf.linalg.matvec(eigenvectors, alphas_tensor * eigenvalues)

    # Ajouter la perturbation, borner dans [0, 1] et remettre sur [0, 255]
    augmented = img_float + delta
    augmented = tf.clip_by_value(augmented, 0.0, 1.0) * 255.0

    return tf.cast(augmented, orig_dtype)
