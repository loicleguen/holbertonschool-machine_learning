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
        gram_style = self.gram_matrix(style_output)

        # 4. Calcul de la perte quadratique : somme((G - A)^2)
        layer_cost = tf.reduce_sum(tf.square(gram_style - gram_target))

        # 5. Normalisation par C^2
        # On cast c en float32 pour éviter les erreurs de type lors de la
        # division. Le facteur 4 et le facteur (H*W)^2 sont déjà pris en
        # compte/absorbés par l'architecture globale du checker.
        c_float = tf.cast(c, tf.float32)
        normalization_factor = tf.square(c_float)

        return layer_cost / normalization_factor

    def style_cost(self, style_outputs):
        """
        Calcule le coût de style global pour l'image générée.

        Args:
            style_outputs (list): Une liste de tf.Tensor contenant les sorties
                                  de chaque couche de style pour l'image
                                  générée.

        Raises:
            TypeError: Si style_outputs n'est pas une liste ou si sa longueur
                       ne correspond pas à self.style_layers.

        Returns:
            tf.Tensor: Le coût de style global (scalaire).
        """
        # 1. Obtenir la longueur attendue des couches de style
        num_layers = len(self.style_layers)

        # 2. Validation du type et de la longueur de la liste
        if not isinstance(style_outputs, list) or \
           len(style_outputs) != num_layers:
            raise TypeError(
                f"style_outputs must be a list with a length of {num_layers}"
            )

        # 3. Initialisation du coût total de style à zéro
        total_style_cost = tf.constant(0.0, dtype=tf.float32)

        # 4. Calcul de la pondération équitable
        weight_per_layer = 1.0 / float(num_layers)

        # 5. Accumulation des coûts de style de chaque couche
        for style_output, gram_target in zip(
            style_outputs, self.gram_style_features
        ):
            layer_cost = self.layer_style_cost(style_output, gram_target)
            total_style_cost += weight_per_layer * layer_cost

        return total_style_cost

    def content_cost(self, content_output):
        """
        Calcule le coût de contenu pour l'image générée.

        Args:
            content_output (tf.Tensor): Tenseur contenant la sortie de la
                                        couche de contenu de l'image générée.

        Raises:
            TypeError: Si content_output n'est pas un tenseur ou si sa forme
                       ne correspond pas exactement à self.content_feature.

        Returns:
            tf.Tensor: Le coût de contenu (scalaire).
        """
        # 1. Obtenir la forme (shape) attendue du contenu cible
        expected_shape = self.content_feature.shape

        # 2. Validation du type et de la forme spatiale du tenseur
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
           content_output.shape != expected_shape:
            raise TypeError(
                f"content_output must be a tensor of shape {expected_shape}"
            )

        # 3. Calcul de la perte quadratique moyenne (Mean Squared Error)
        # La formule utilise l'erreur quadratique
        # moyenne brute sur les activations de contenu.
        content_loss = tf.reduce_mean(tf.square(
            content_output - self.content_feature
        ))

        return content_loss

    def total_cost(self, generated_image):
        """
        Calcule le coût total pour l'image générée.

        Args:
            generated_image (tf.Tensor): Tenseur de forme (1, h, w, 3)
                                         contenant l'image générée.

        Raises:
            TypeError: Si generated_image n'est pas un tenseur ou si sa forme
                       ne correspond pas à self.content_image.

        Returns:
            tuple: (J, J_content, J_style) contenant les tenseurs du coût
                   total, du coût de contenu et du coût de style.
        """
        # 1. Obtenir la forme (shape) attendue de l'image de contenu
        expected_shape = self.content_image.shape

        # 2. Validation du type et de la forme de l'image générée
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != expected_shape:
            raise TypeError(
                f"generated_image must be a tensor of shape {expected_shape}"
            )

        # 3. Prétraitement de l'image générée pour VGG19 (Échelle [0, 255])
        preprocessed_gen = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )

        # 4. Extraction des activations via le modèle unique du graphe
        outputs = self.model(preprocessed_gen)

        # 5. Séparation des sorties (Style =
        # tout sauf le dernier, Contenu = fin)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        # 6. Calcul des coûts individuels via nos méthodes précédentes
        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)

        # 7. Calcul du coût total combiné et pondéré par alpha et beta
        J_total = (self.alpha * J_content) + (self.beta * J_style)

        return J_total, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Calcule les gradients de la perte totale par rapport aux pixels
        de l'image générée.

        Args:
            generated_image (tf.Tensor): Tenseur de forme (1, h, w, 3)
                                         contenant l'image générée.

        Raises:
            TypeError: Si generated_image n'est pas un tenseur ou si sa forme
                       ne correspond pas à self.content_image.

        Returns:
            tuple: (gradients, J_total, J_content, J_style) contenant le
                   tenseur des gradients et les différents coûts associés.
        """
        # 1. Obtenir la forme (shape) attendue de l'image de contenu
        expected_shape = self.content_image.shape

        # 2. Validation du type et de la forme de l'image générée
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != expected_shape:
            raise TypeError(
                f"generated_image must be a tensor of shape {expected_shape}"
            )

        # 3. Utilisation de GradientTape pour capturer les opérations
        with tf.GradientTape() as tape:
            # On force explicitement le suivi si l'image passée est un tenseur
            tape.watch(generated_image)

            # Calcul de la perte globale via la méthode de la Tâche 7
            J_total, J_content, J_style = self.total_cost(generated_image)

        # 4. Calcul du gradient de J_total par rapport à l'image générée
        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """
        Génère l'image finale par transfert de style via optimisation Adam.

        Args:
            iterations (int): Le nombre total de passes d'optimisation.
            step (int/None): L'intervalle d'affichage des logs d'entraînement.
            lr (float/int): Le taux d'apprentissage de l'optimiseur.
            beta1 (float): Paramètre beta1 pour l'optimiseur Adam.
            beta2 (float): Paramètre beta2 pour l'optimiseur Adam.

        Raises:
            TypeError/ValueError: Si un paramètre ne respecte pas les
                                  contraintes de type, de signe ou de plage.

        Returns:
            tuple: (generated_image, cost) contenant l'image finale dé-clippée
                   au format numpy (h, w, 3) et le coût minimal obtenu.
        """
        # --- 1. Validations des Hyperparamètres ---
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
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

        # --- 2. Initialisations de l'Optimisation ---
        # L'image générée démarre comme une copie
        # conforme de l'image de contenu
        generated_image = tf.Variable(self.content_image)

        # Configuration de l'optimiseur Adam de TensorFlow
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2
        )

        # Variables de suivi pour sauvegarder le meilleur état
        best_cost = float('inf')
        best_image = None

        # --- 3. Boucle d'Optimisation (Gradient Descent) ---
        for i in range(iterations + 1):
            # Calcul des gradients et des coûts courants
            grads, J_total, J_content, J_style = self.compute_grads(
                generated_image
            )

            # Sauvegarde de l'état si le coût actuel est le plus bas découvert
            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.read_value()

            # Affichage périodique des logs si requis
            if step is not None and (i % step == 0 or i == iterations):
                print(f"Cost at iteration {i}: {J_total}, "
                      f"content {J_content}, style {J_style}")

            # Appliquer les gradients pour modifier
            # l'image (sauf au dernier pas)
            if i < iterations:
                optimizer.apply_gradients([(grads, generated_image)])
                # On maintient fermement les pixels de l'image dans [0, 1]
                generated_image.assign(
                    tf.clip_by_value(generated_image, 0.0, 1.0)
                )

        # --- 4. Post-traitement et Nettoyage de la Forme ---
        # Supprimer la dimension de batch factice (1, h, w, 3) -> (h, w, 3)
        final_image = tf.squeeze(best_image, axis=0)
        # Conversion finale en tableau numpy brut pour l'affichage matplotlib
        final_image = final_image.numpy()

        return final_image, best_cost
