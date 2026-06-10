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

        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        self.class_t = float(class_t)
        self.nms_t = float(nms_t)
        self.anchors = np.array(anchors)
