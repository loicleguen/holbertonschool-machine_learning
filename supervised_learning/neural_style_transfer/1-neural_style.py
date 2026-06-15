#!/usr/bin/env python3
"""
Module contenant la classe NST pour le Neural Style Transfer.
Ce module initialise les attributs, gère la validation des entrées,
propose une méthode de mise à l'échelle
et charge le modèle d'extraction VGG-19.
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Classe NST (Neural Style Transfer).
    Permet de configurer, charger le modèle d'extraction de caractéristiques
    et d'exécuter le transfert de style entre images.
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
        self.model = None
        self.load_model()

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

        if h > w:
            h_new = 512
            w_new = round((w * 512) / h)
        else:
            w_new = 512
            h_new = round((h * 512) / w)

        image_resized = tf.image.resize(
            image,
            [h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        image_scaled = image_resized / 255.0
        image_scaled = tf.clip_by_value(image_scaled, 0.0, 1.0)
        image_tensor = tf.expand_dims(image_scaled, axis=0)

        return image_tensor

    def load_model(self):
        """
        Crée et configure le modèle Keras utilisé pour calculer les coûts.
        Le modèle utilise VGG19 pré-entraîné sur ImageNet comme base,
        remplace le MaxPool par de l'AveragePool, fige les poids,
        et renvoie les activations des couches de style et de contenu.
        """
        # Chargement du modèle VGG19 de base
        # (sans les couches denses de la fin)
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        # On extrait la configuration structurelle du réseau VGG19
        vgg_config = vgg.get_config()

        # On parcourt toutes les couches pour remplacer
        # le MaxPooling par de l'AveragePooling
        for layer in vgg_config['layers']:
            if layer['class_name'] == 'MaxPooling2D':
                layer['class_name'] = 'AveragePooling2D'
                # On nettoie les options spécifiques
                # de pooling si elles existent
                if 'options' in layer:
                    del layer['options']

        # Reconstitution du modèle avec
        # l'architecture modifiée (AveragePooling)
        vgg_avg = tf.keras.Model.from_config(vgg_config)

        # On injecte les poids pré-entraînés d'ImageNet
        # dans notre nouveau modèle
        vgg_avg.set_weights(vgg.get_weights())

        # Gel de tous les paramètres du modèle pour
        # qu'ils soient non entraînés (Trainable params: 0)
        for layer in vgg_avg.layers:
            layer.trainable = False

        # Récupération des tenseurs de sortie des couches cibles
        outputs = []
        for name in self.style_layers:
            outputs.append(vgg_avg.get_layer(name).output)
        outputs.append(vgg_avg.get_layer(self.content_layer).output)

        # Création et sauvegarde du sous-modèle
        # final dans l'attribut d'instance
        self.model = tf.keras.Model(inputs=vgg_avg.input, outputs=outputs)
