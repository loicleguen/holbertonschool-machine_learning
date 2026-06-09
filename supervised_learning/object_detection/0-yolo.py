#!/usr/bin/env python3
"""Module Yolo.

Ce module définit la classe `Yolo` utilisée pour initialiser
le modèle Darknet/Keras (YOLOv3) et stocker les paramètres
nécessaires à la détection (noms de classes, seuils, ancres).
"""

import tensorflow as tf
import numpy as np


class Yolo:
    """Classe Yolo pour l'initialisation du modèle YOLOv3.

    Attributs publics :
    - model : le modèle Darknet/Keras chargé avec `tf.keras.models.load_model`
    - class_names : liste de noms de classes (ordre d'indexation du modèle)
    - class_t : float, seuil de score pour filtrage initial des boîtes
    - nms_t : float, seuil IoU pour la suppression non-maximale
    - anchors : numpy.ndarray contenant les
    ancres (shape attendu (outputs, anchor_boxes, 2))
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialise une instance Yolo.

        Args:
            model_path (str): chemin vers le fichier Keras (yolo.h5).
            classes_path (str): chemin vers le fichier
            texte des classes (une classe par ligne).
            class_t (float): seuil de score pour
            filtrer les prédictions faibles.
            nms_t (float): seuil IoU pour la suppression non-maximale.
            anchors (numpy.ndarray): tableau d'ancres
            de forme (outputs, anchor_boxes, 2).

        Effets:
            - charge le modèle dans `self.model`
            - lit `classes_path` et remplit `self.class_names`
            - cast `class_t` et `nms_t` en float
            - convertit `anchors` en `numpy.ndarray`
        """
        # Charger le modèle Keras sauvegardé (Darknet -> Keras)
        # Avertissement courant: "No training
        # configuration found..." est normal
        self.model = tf.keras.models.load_model(model_path)

        # Lire les noms de classes depuis le fichier (une ligne = une classe)
        with open(classes_path, 'r') as f:
            # strip() retire \n et les lignes vides sont ignorées
            self.class_names = [line.strip() for line in f if line.strip()]

        # Seuils pour filtrage et NMS (s'assurer que ce sont des float)
        self.class_t = float(class_t)
        self.nms_t = float(nms_t)

        # Stocker les ancres sous forme de
        # numpy.ndarray (utilisation ultérieure)
        self.anchors = np.array(anchors)
