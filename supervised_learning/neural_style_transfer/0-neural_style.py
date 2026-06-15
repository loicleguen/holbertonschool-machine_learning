#!/usr/bin/env python3
"""
Module contenant la classe NST pour le Neural Style Transfer.
Ce module initialise les attributs, gère la validation des entrées
et propose une méthode statique pour redimensionner et normaliser les images.
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Classe NST (Neural Style Transfer).
    Permet de configurer et d'exécuter le transfert de style
    entre une image de contenu et une image de style.
    """
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Constructeur de la classe NST.

        Args:
            style_image (np.ndarray): Image de référence pour le style.
            content_image (np.ndarray): Image de référence pour le contenu.
            alpha (float/int): Poids pour le coût du contenu.
            beta (float/int): Poids pour le coût du style.

        Raises:
            TypeError: Si les images ne sont pas des np.ndarray de forme
                       (h, w, 3), ou si alpha/beta ne sont pas des
                       nombres positifs ou nuls.
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Redimensionne et normalise une image.
        La valeur des pixels passe à [0, 1], le plus grand côté devient 512
        et une dimension de lot (batch) est ajoutée au début.

        Args:
            image (np.ndarray): L'image brute à transformer.

        Returns:
            tf.Tensor: L'image transformée sous forme de tenseur TensorFlow.

        Raises:
            TypeError: Si l'image n'est pas un np.ndarray de forme (h, w, 3).
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        # Calcul des nouvelles dimensions proportionnelles
        if h > w:
            h_new = 512
            w_new = round((w * 512) / h)
        else:
            w_new = 512
            h_new = round((h * 512) / w)

        # Redimensionnement avec interpolation bicubique
        # tf.image.resize attend un tenseur de
        # forme (h, w, 3) ou (batch, h, w, 3)
        image_resized = tf.image.resize(
            image,
            [h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        # Normalisation des valeurs de pixels de [0, 255] à [0, 1]
        image_scaled = image_resized / 255.0

        # Limitation stricte des valeurs entre
        # 0 et 1 (sécurité pour le bicubique)
        image_scaled = tf.clip_by_value(image_scaled, 0.0, 1.0)

        # Ajout de la dimension de lot -> (1, h_new, w_new, 3)
        image_tensor = tf.expand_dims(image_scaled, axis=0)

        return image_tensor
