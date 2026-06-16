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
    # Liste des couches utilisées pour capturer les détails du style
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    # Couche profonde utilisée pour capturer l'agencement du contenu
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        """
        Constructeur de la classe NST.

        Args:
            style_image (np.ndarray): Image de référence pour le style.
            content_image (np.ndarray): Image de référence pour le contenu.
            alpha (float/int): Poids pour le coût du contenu.
            beta (float/int): Poids pour le coût du style.
            var (float/int): Poids pour le coût variationnel.

        Raises:
            TypeError: Si les images ou les poids ne respectent pas les types
                       et formes imposés.
        """
        # Vérification du type et de la forme tridimensionnelle de style_image
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        # Vérification du type et de la forme
        # tridimensionnelle de content_image
        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        # Validation que le poids de contenu 'alpha'
        # est un nombre positif ou nul
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        # Validation que le poids de style 'beta' est un nombre positif ou nul
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Validation que le poids variationnel 'var'
        # est un nombre positif ou nul
        if not isinstance(var, (int, float)) or var < 0:
            raise TypeError("var must be a non-negative number")

        # Redimensionnement et mise à l'échelle des images d'entrée
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        # Attribution des hyperparamètres de pondération des pertes
        self.alpha = alpha
        self.beta = beta
        self.var = var

        # Initialisation et chargement du réseau VGG19 modifié
        self.model = None
        self.load_model()

        # Extraction des features de référence pour le style et le contenu
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
        # Validation défensive du format de l'image numpy à traiter
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        # Calcul des nouvelles dimensions en
        # maintenant le ratio d'aspect initial
        if h > w:
            h_new = 512
            w_new = round((w * 512) / h)
        else:
            w_new = 512
            h_new = round((h * 512) / w)

        # Redimensionnement spatial via une interpolation bicubique lisse
        image_resized = tf.image.resize(
            image,
            [h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        # Normalisation des valeurs de pixels dans la plage [0.0, 1.0]
        image_scaled = image_resized / 255.0
        image_scaled = tf.clip_by_value(image_scaled, 0.0, 1.0)

        # Ajout de la dimension de batch requise par Keras (1, h_new, w_new, 3)
        image_tensor = tf.expand_dims(image_scaled, axis=0)

        return image_tensor

    def load_model(self):
        """
        Crée et configure le modèle Keras utilisé pour calculer les coûts.
        Remplace le MaxPool par de l'AveragePool et fige les poids.
        """
        # Chargement du réseau VGG19 pré-entraîné sur ImageNet sans la tête FC
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        # Extraction de la configuration réseau
        # sous forme de dictionnaire Python
        vgg_config = vgg.get_config()

        # Altération de la structure : Remplacement
        # des MaxPooling par des AveragePooling
        for layer in vgg_config['layers']:
            if layer['class_name'] == 'MaxPooling2D':
                layer['class_name'] = 'AveragePooling2D'
                if 'options' in layer:
                    del layer['options']

        # Reconstruction du modèle à partir de la configuration modifiée
        vgg_avg = tf.keras.Model.from_config(vgg_config)
        vgg_avg.set_weights(vgg.get_weights())

        # Gel global des poids du réseau pour
        # désactiver l'apprentissage classique
        for layer in vgg_avg.layers:
            layer.trainable = False

        # Sélection et regroupement des sorties
        # de couches cibles (Style + Contenu)
        outputs = []
        for name in self.style_layers:
            outputs.append(vgg_avg.get_layer(name).output)
        outputs.append(vgg_avg.get_layer(self.content_layer).output)

        # Instanciation du modèle multi-sorties final dédié aux coûts
        self.model = tf.keras.Model(inputs=vgg_avg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calcule la matrice de Gram normalisée d'une couche spécifiée.
        """
        # Validation du tenseur d'entrée
        # (doit être de rang 4 : [Batch, H, W, C])
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Récupération dynamique des dimensions spatiales et des canaux
        shape = tf.shape(input_layer)
        h = shape[1]
        w = shape[2]
        c = shape[3]

        # Aplatissement spatial pour isoler les vecteurs de features (H*W, C)
        features = tf.reshape(input_layer, (h * w, c))

        # Produit matriciel pour mesurer les
        # corrélations de caractéristiques (C, C)
        gram = tf.matmul(features, features, transpose_a=True)

        # Normalisation par le nombre total de positions spatiales (H * W)
        num_locations = tf.cast(h * w, tf.float32)
        gram_normalized = gram / num_locations

        # Extension de rang pour ajouter la dimension de batch factice
        gram_tensor = tf.expand_dims(gram_normalized, axis=0)

        return gram_tensor

    def generate_features(self):
        """
        Extrait les caractéristiques de style et de contenu à partir
        des images cibles passées au modèle.
        """
        # Remise à l'échelle sur [0, 255] puis
        # soustraction de la moyenne ImageNet
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        # Passage des images dans le réseau
        # pour obtenir les activations internes
        style_outputs = self.model(preprocessed_style)
        content_outputs = self.model(preprocessed_content)

        # Isolation et calcul des matrices de Gram pour les couches de style
        style_layers_outputs = style_outputs[:-1]
        self.gram_style_features = [
            self.gram_matrix(layer) for layer in style_layers_outputs
        ]

        # Récupération brute des activations pour la couche de contenu
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calcule le coût de style pour une seule couche.
        """
        # Validation du tenseur d'activation généré
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[3]

        # Validation de la conformité de la matrice de Gram de référence
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           gram_target.shape != (1, c, c):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]"
            )

        # Calcul de la matrice de Gram de l'image en cours de génération
        gram_style = self.gram_matrix(style_output)

        # Calcul de l'erreur quadratique cumulative entre les deux matrices
        layer_cost = tf.reduce_sum(tf.square(gram_style - gram_target))

        # Normalisation finale de la perte par
        # le carré du nombre de canaux (C^2)
        c_float = tf.cast(c, tf.float32)
        normalization_factor = tf.square(c_float)

        return layer_cost / normalization_factor

    def style_cost(self, style_outputs):
        """
        Calcule le coût de style global pour l'image générée.
        """
        num_layers = len(self.style_layers)

        # Validation du format de la structure accueillant les activations
        if not isinstance(style_outputs, list) or \
           len(style_outputs) != num_layers:
            raise TypeError(
                f"style_outputs must be a list with a length of {num_layers}"
            )

        total_style_cost = tf.constant(0.0, dtype=tf.float32)

        # Calcul de la pondération (chaque couche contribue de manière égale)
        weight_per_layer = 1.0 / float(num_layers)

        # Itération parallèle sur les activations
        # courantes et les cibles idéales
        for style_output, gram_target in zip(
            style_outputs, self.gram_style_features
        ):
            layer_cost = self.layer_style_cost(style_output, gram_target)
            total_style_cost += weight_per_layer * layer_cost

        return total_style_cost

    def content_cost(self, content_output):
        """
        Calcule le coût de contenu pour l'image générée.
        """
        expected_shape = self.content_feature.shape

        # Validation de la structure spatiale du tenseur de contenu généré
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
           content_output.shape != expected_shape:
            raise TypeError(
                f"content_output must be a tensor of shape {expected_shape}"
            )

        # Perte de contenu définie par l'erreur quadratique moyenne (MSE)
        content_loss = tf.reduce_mean(tf.square(
            content_output - self.content_feature
        ))

        return content_loss

    @staticmethod
    def variational_cost(generated_image):
        """
        Calcule le coût variationnel pour l'image générée.
        Acts as a regularizer to smooth pixel variations (denoising).
        """
        # Validation du rang tensoriel de l'image générée
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           len(generated_image.shape) not in [3, 4]:
            raise TypeError("image must be a tensor of rank 3 or 4")

        # tf.image.total_variation calcule le
        # bruit/décalage entre pixels voisins.
        # reduce_sum fait la somme absolue de
        # ces écarts pour obtenir un coût scalaire.
        return tf.reduce_sum(tf.image.total_variation(generated_image))

    def total_cost(self, generated_image):
        """
        Calcule le coût total pour l'image générée avec coût variationnel.
        """
        expected_shape = self.content_image.shape

        # Validation de l'image générée par rapport au canevas de contenu
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != expected_shape:
            raise TypeError(
                f"generated_image must be a tensor of shape {expected_shape}"
            )

        # Prétraitement de l'image d'évaluation pour VGG19
        preprocessed_gen = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )

        # Extraction des activations pour l'image courante
        outputs = self.model(preprocessed_gen)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        # Calcul individuel des trois types de pertes (Contenu, Style, TV)
        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J_var = self.variational_cost(generated_image)

        # Somme pondérée par les coefficients utilisateur (alpha, beta, var)
        J_total = (self.alpha * J_content) + \
                  (self.beta * J_style) + \
                  (self.var * J_var)

        return J_total, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """
        Calcule les gradients de la perte totale par rapport aux pixels.
        """
        expected_shape = self.content_image.shape

        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != expected_shape:
            raise TypeError(
                f"generated_image must be a tensor of shape {expected_shape}"
            )

        # Ouverture d'un contexte de suivi des
        # opérations pour la dérivation automatique
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            # Évaluation de la fonction de coût global
            J_total, J_content, J_style, J_var = self.total_cost(
                generated_image
            )

        # Calcul analytique du gradient : dJ / d(pixels)
        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """
        Génère l'image finale par transfert de style via optimisation Adam.
        """
        # --- 1. Validations de types et de valeurs
        # des hyperparamètres d'optimisation ---
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step => iterations:
                raise ValueError(
                    "step must be positive and less than iterations"
                )

        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not (0.0 <= beta1 <= 1.0):
            raise ValueError("beta1 must be in the range [0, 1]")

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not (0.0 <= beta2 <= 1.0):
            raise ValueError("beta2 must be in the range [0, 1]")

        # Initialisation de l'image de synthèse à partir de l'image de contenu
        generated_image = tf.Variable(self.content_image)

        # Instanciation de l'optimiseur Adam
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2
        )

        # Sentinelles pour enregistrer la
        # meilleure image obtenue (coût minimal)
        best_cost = float('inf')
        best_image = None

        # --- 2. Boucle principale de la descente de gradient ---
        for i in range(iterations + 1):
            # Récupération des gradients et valeurs de pertes
            grads, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image
            )

            # Suivi de la meilleure performance globale
            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.read_value()

            # Affichage formaté des coûts lors des étapes de contrôle
            if step is not None and (i % step == 0 or i == iterations):
                print(f"Cost at iteration {i}: {J_total}, "
                      f"content {J_content}, style {J_style}, var {J_var}")

            # Application de la correction des
            # pixels (sauf à la dernière itération)
            if i < iterations:
                optimizer.apply_gradients([(grads, generated_image)])
                # Contrainte stricte pour maintenir
                # les pixels réels dans [0.0, 1.0]
                generated_image.assign(
                    tf.clip_by_value(generated_image, 0.0, 1.0)
                )

        # --- 3. Nettoyage et conversion de sortie ---
        # Suppression de la dimension de batch
        # factice (1, H, W, 3) -> (H, W, 3)
        final_image = tf.squeeze(best_image, axis=0)
        # Transformation du tenseur TensorFlow en tableau numpy standard
        final_image = final_image.numpy()

        return final_image, best_cost
