#!/usr/bin/env python3
"""Module Yolo - tâche 5 : preprocess_images."""

import tensorflow as tf
import numpy as np
import cv2
from glob import glob


class Yolo:
    """Classe Yolo pour la détection d'objets avec YOLOv3."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialise une instance Yolo."""
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        self.class_t = float(class_t)
        self.nms_t = float(nms_t)
        self.anchors = np.array(anchors)

    def process_outputs(self, outputs, image_size):
        """Traite les sorties brutes du modèle Darknet."""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, nb_anchors, _ = output.shape

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            b_xy = 1 / (1 + np.exp(-t_xy))

            col = np.arange(grid_w).reshape(1, grid_w, 1, 1)
            row = np.arange(grid_h).reshape(grid_h, 1, 1, 1)
            col = np.tile(col, (grid_h, 1, nb_anchors, 1))
            row = np.tile(row, (1, grid_w, nb_anchors, 1))
            grid = np.concatenate([col, row], axis=-1)

            b_xy = (b_xy + grid) / [grid_w, grid_h]

            anchors_wh = self.anchors[i] / [input_w, input_h]
            anchors_wh = anchors_wh.reshape(1, 1, nb_anchors, 2)
            b_wh = anchors_wh * np.exp(t_wh)

            x1y1 = (b_xy - b_wh / 2) * [image_w, image_h]
            x2y2 = (b_xy + b_wh / 2) * [image_w, image_h]
            box = np.concatenate([x1y1, x2y2], axis=-1)
            boxes.append(box)

            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(confidence)

            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filtre les boîtes selon le score de confiance."""
        all_boxes = []
        all_classes = []
        all_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            box_classes = np.argmax(scores, axis=-1)
            box_scores = np.max(scores, axis=-1)
            mask = box_scores >= self.class_t

            all_boxes.append(boxes[i][mask])
            all_classes.append(box_classes[mask])
            all_scores.append(box_scores[mask])

        filtered_boxes = np.concatenate(all_boxes, axis=0)
        box_classes = np.concatenate(all_classes, axis=0)
        box_scores = np.concatenate(all_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Applique la suppression non-maximale par classe."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for c in np.unique(box_classes):
            idx = np.where(box_classes == c)
            c_boxes = filtered_boxes[idx]
            c_scores = box_scores[idx]

            order = np.argsort(c_scores)[::-1]
            c_boxes = c_boxes[order]
            c_scores = c_scores[order]

            while len(c_boxes) > 0:
                box_predictions.append(c_boxes[0])
                predicted_box_classes.append(c)
                predicted_box_scores.append(c_scores[0])

                if len(c_boxes) == 1:
                    break

                x1 = np.maximum(c_boxes[0, 0], c_boxes[1:, 0])
                y1 = np.maximum(c_boxes[0, 1], c_boxes[1:, 1])
                x2 = np.minimum(c_boxes[0, 2], c_boxes[1:, 2])
                y2 = np.minimum(c_boxes[0, 3], c_boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                area_0 = ((c_boxes[0, 2] - c_boxes[0, 0]) *
                          (c_boxes[0, 3] - c_boxes[0, 1]))
                area_rest = ((c_boxes[1:, 2] - c_boxes[1:, 0]) *
                             (c_boxes[1:, 3] - c_boxes[1:, 1]))

                union = area_0 + area_rest - intersection
                iou = intersection / union

                c_boxes = c_boxes[1:][iou < self.nms_t]
                c_scores = c_scores[1:][iou < self.nms_t]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Charge toutes les images d'un dossier."""
        image_paths = (glob(folder_path + '/*.jpg') +
                       glob(folder_path + '/*.jpeg') +
                       glob(folder_path + '/*.png'))
        images = [cv2.imread(path) for path in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """Redimensionne et normalise les images pour le modèle."""
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_shapes = np.array([[img.shape[0], img.shape[1]]
                                 for img in images])

        pimages = np.array([
            cv2.resize(img, (input_h, input_w),
                       interpolation=cv2.INTER_CUBIC) / 255.0
            for img in images
        ])

        return pimages, image_shapes
