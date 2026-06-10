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

    def process_outputs(self, outputs, image_size):
        """Traite les sorties du modèle Darknet pour une image.

        Args:
            outputs (list): liste de numpy.ndarrays de forme
                (grid_h, grid_w, anchor_boxes, 4 + 1 + classes)
            image_size (numpy.ndarray): [image_height, image_width]

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size

        # Taille d'entrée du modèle (ex: 416x416)
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, nb_anchors, _ = output.shape

            # --- Extraire tx, ty, tw, th et appliquer sigmoid sur tx, ty ---
            t_xy = output[..., :2]   # tx, ty
            t_wh = output[..., 2:4]  # tw, th

            # sigmoid sur tx, ty pour obtenir bx, by dans la cellule [0,1]
            b_xy = 1 / (1 + np.exp(-t_xy))

            # --- Construire la grille des offsets cx, cy ---
            # cx[i,j] = j (numéro de colonne), cy[i,j] = i (numéro de ligne)
            col = np.arange(grid_w).reshape(1, grid_w, 1, 1)
            row = np.arange(grid_h).reshape(grid_h, 1, 1, 1)

            col = np.tile(col, (grid_h, 1, nb_anchors, 1))  # broadcast
            row = np.tile(row, (1, grid_w, nb_anchors, 1))

            grid = np.concatenate([col, row], axis=-1)  # (gh, gw, anc, 2)

            # bx, by en coordonnées de grille -> normaliser
            # par taille de grille
            b_xy = (b_xy + grid) / [grid_w, grid_h]

            # --- Décoder bw, bh avec les ancres ---
            anchors_wh = self.anchors[i]  # shape (nb_anchors, 2)
            # Normaliser les ancres par la taille d'entrée du modèle
            anchors_wh = anchors_wh / [input_w, input_h]

            # Reshape pour broadcast : (1, 1, nb_anchors, 2)
            anchors_wh = anchors_wh.reshape(1, 1, nb_anchors, 2)

            b_wh = anchors_wh * np.exp(t_wh)

            # --- Convertir (bx, by, bw, bh) -> (x1, y1, x2, y2) ---
            # b_xy et b_wh sont normalisés [0,1] par rapport à l'image
            x1y1 = (b_xy - b_wh / 2) * [image_w, image_h]
            x2y2 = (b_xy + b_wh / 2) * [image_w, image_h]

            box = np.concatenate([x1y1, x2y2], axis=-1)
            boxes.append(box)

            # --- Confidences et probabilités de classes ---
            # sigmoid sur box_confidence (index 4)
            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(confidence)

            # sigmoid sur class probs (index 5 à fin)
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
