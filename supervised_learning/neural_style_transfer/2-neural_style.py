#!/usr/bin/env python3
"""
Module contenant la classe NST pour le Neural Style Transfer.
Ce module gère l'initialisation, le chargement du modèle VGG-19,
et le calcul de la matrice de Gram pour extraire le style.
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Classe NST (Neural Style Transfer).
    Permet de configurer, charger le modèle VGG-19 et calculer
    les matrices de Gram pour l'extraction de style artistique.
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
        Remplace le MaxPool par de l'AveragePool et fige les poids.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg_config = vgg.get_config()

        for layer in vgg_config['layers']:
            if layer['class_name'] == 'MaxPooling2D':
                layer['class_name'] = 'AveragePooling2D'
                if 'options' in layer:
                    del layer['options']

        vgg_avg = tf.keras.Model.from_config(vgg_config)
        vgg_avg.set_weights(vgg.get_weights())

        for layer in vgg_avg.layers:
            layer.trainable = False

        outputs = []
        for name in self.style_layers:
            outputs.append(vgg_avg.get_layer(name).output)
        outputs.append(vgg_avg.get_layer(self.content_layer).output)

        self.model = tf.keras.Model(inputs=vgg_avg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calcule la matrice de Gram d'une couche spécifiée.

        Args:
            input_layer (tf.Tensor | tf.Variable): Un tenseur de rang 4
                contenant la sortie d'une couche, de forme (1, h, w, c).

        Returns:
            tf.Tensor: Un tenseur de forme (1, c, c) contenant la matrice
                de Gram normalisée.

        Raises:
            TypeError: Si input_layer n'est pas un tenseur ou une variable
                       TensorFlow de rang 4 (4 dimensions).
        """
        # Vérification du type (Tensor ou Variable) et du rang (doit être 4)
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Extraction des dimensions : batch (toujours 1 ici),
        # hauteur, largeur, canaux
        # On utilise tf.shape pour supporter l'exécution dynamique
        shape = tf.shape(input_layer)
        h = shape[1]
        w = shape[2]
        c = shape[3]

        # Étape 1 : Aplatir les dimensions spatiales (h, w) -> (h * w, c)
        # On passe de (1, h, w, c) à (h * w, c)
        features = tf.reshape(input_layer, (h * w, c))

        # Étape 2 : Produit matriciel (Transposée de features X features)
        # Résultat de forme (c, c)
        gram = tf.matmul(features, features, transpose_a=True)

        # Étape 3 : Normalisation par le nombre de points spatiaux (h * w)
        # Cast obligatoire en float32 pour éviter
        # les erreurs de type lors de la division
        num_locations = tf.cast(h * w, tf.float32)
        gram_normalized = gram / num_locations

        # Étape 4 : Ajouter à nouveau la dimension
        # de batch au début -> (1, c, c)
        gram_tensor = tf.expand_dims(gram_normalized, axis=0)

        return gram_tensor
