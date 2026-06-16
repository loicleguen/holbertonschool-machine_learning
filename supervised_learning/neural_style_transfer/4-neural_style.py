#!/usr/bin/env python3
"""
Module contenant la classe NST pour le Neural Style Transfer.
Ce module initialise l'environnement, configure le modèle VGG-19,
calcule les matrices de Gram et extrait les caractéristiques des images.
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Classe NST (Neural Style Transfer).
    Permet d'extraire les caractéristiques de style et de contenu
    et d'exécuter un algorithme de transfert de style artistique.
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
            TypeError: Si les images ou les poids ne respectent pas les types
                       et formes imposés.
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

        self.gram_style_features = None
        self.content_feature = None
        self.generate_features()

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
        Calcule la matrice de Gram normalisée d'une couche spécifiée.
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        shape = tf.shape(input_layer)
        h = shape[1]
        w = shape[2]
        c = shape[3]

        features = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(features, features, transpose_a=True)

        num_locations = tf.cast(h * w, tf.float32)
        gram_normalized = gram / num_locations
        gram_tensor = tf.expand_dims(gram_normalized, axis=0)

        return gram_tensor

    def generate_features(self):
        """
        Extrait les caractéristiques de style et de contenu à partir
        des images cibles passées au modèle.
        """
        # 1. Remettre à l'échelle [0, 255] et appliquer le prétraitement VGG19
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        # 2. Extraction des activations via le modèle
        style_outputs = self.model(preprocessed_style)
        content_outputs = self.model(preprocessed_content)

        # 3. Extraction par slicing
        style_layers_outputs = style_outputs[:-1]
        self.gram_style_features = [
            self.gram_matrix(layer) for layer in style_layers_outputs
        ]

        # 4. Récupération de la couche de contenu
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calcule le coût de style pour une seule couche.

        Args:
            style_output (tf.Tensor): Tenseur de forme (1, h, w, c) contenant
                                      la sortie de couche de l'image générée.
            gram_target (tf.Tensor): Tenseur de forme (1, c, c) contenant
                                     la matrice de Gram cible de cette couche.

        Raises:
            TypeError: Si les formes ou les types de tenseurs ne correspondent
                       pas aux critères requis.

        Returns:
            tf.Tensor: Le coût de style de la couche (scalaire).
        """
        # 1. Validation de style_output (Rank 4)
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        # Extraction du nombre de canaux (c)
        c = style_output.shape[3]

        # 2. Validation de gram_target (Shape [1, c, c])
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           gram_target.shape != (1, c, c):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]"
            )

        # 3. Calcul de la matrice de Gram de la couche générée (G)
        # Note : self.gram_matrix renvoie déjà une matrice divisée par (h * w)
        gram_style = self.gram_matrix(style_output)

        # 4. Calcul de la perte quadratique
        # Somme des différences au carré : somme((G - A)^2)
        layer_cost = tf.reduce_sum(tf.square(gram_style - gram_target))

        # 5. Normalisation par 4 * C^2
        # (car le facteur (H*W)^2 est géré par gram_matrix)
        # On cast c en float32 pour éviter les
        # erreurs de type lors de la division
        c_float = tf.cast(c, tf.float32)
        normalization_factor = 4.0 * tf.square(c_float)

        return layer_cost / normalization_factor
