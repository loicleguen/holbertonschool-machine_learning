#!/usr/bin/env python3
"""
Initialize Yolo and process outputs
"""
import numpy as np
from tensorflow import keras as K
import cv2
import os


class Yolo:
    """Class that uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo instance.

        Parameters:
        - model_path: path to where a Darknet Keras model is stored
        - classes_path: path to where the list of class names used for
            the Darknet model is found
        - class_t: float representing the box score threshold for the
            initial filtering step
        - nms_t: float representing the IOU threshold for non-max
            suppression
        - anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = file.read().strip().split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Applies sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs of the Darknet model

        Parameters:
        - outputs: list of numpy.ndarrays containing the predictions
            from the Darknet model for a single image
        - image_size: numpy.ndarray containing the image’s original
            size [image_height, image_width]

        Returns:
        - boxes: list of numpy.ndarrays containing the processed
            boundary boxes for each output
        - box_confidences: list of numpy.ndarrays containing the box
            confidences for each output
        - box_class_probs: list of numpy.ndarrays containing the box’s
            class probabilities for each output
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract box confidence, class probabilities, and coordinates
            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])
            box_xy = self.sigmoid(output[..., 0:2])  # Center coordinates
            box_wh = np.exp(output[..., 2:4])  # Width and height
            anchors = self.anchors[i].reshape((1, 1, len(self.anchors[i]), 2))
            box_wh *= anchors

            # Create grid to map coordinates
            col = np.tile(np.arange(0, grid_width),
                          grid_height).reshape(-1, grid_width)
            row = np.tile(np.arange(0, grid_height),
                          grid_width).reshape(-1, grid_width).T

            col = col.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=-2)
            row = row.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=-2)

            # Adjust coordinates
            box_xy += np.concatenate((col, row), axis=-1)
            box_xy /= (grid_width, grid_height)
            box_wh /= (self.model.input.shape[1], self.model.input.shape[2])

            # Convert from center coordinates to corner coordinates
            box_xy -= (box_wh / 2)
            boxes.append(np.concatenate((box_xy, box_xy + box_wh), axis=-1))

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        # Adjust boxes to the original image size
        for i in range(len(boxes)):
            boxes[i][..., 0] *= image_width
            boxes[i][..., 1] *= image_height
            boxes[i][..., 2] *= image_width
            boxes[i][..., 3] *= image_height

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter the bounding boxes using the threshold.

        Parameters:
        - boxes: list of numpy.ndarrays containing the processed boundary
            boxes for each output
        - box_confidences: list of numpy.ndarrays containing the processed
            box confidences for each output
        - box_class_probs: list of numpy.ndarrays containing the processed
            box class probabilities for each output

        Returns:
        - filtered_boxes: numpy.ndarray containing all of the filtered
            bounding boxes
        - box_classes: numpy.ndarray containing the class number that
            each box in filtered_boxes predicts
        - box_scores: numpy.ndarray containing the box scores for each
            box in filtered_boxes
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, box_conf, box_class_prob in zip(boxes, box_confidences,
                                                 box_class_probs):
            # Compute box scores
            scores = box_conf * box_class_prob
            max_scores = np.max(scores, axis=-1)
            max_classes = np.argmax(scores, axis=-1)

            # Filter boxes by score threshold
            mask = max_scores >= self.class_t
            filtered_boxes.append(box[mask])
            box_classes.append(max_classes[mask])
            box_scores.append(max_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply Non-max Suppression to the bounding boxes.

        Parameters:
        - filtered_boxes: numpy.ndarray containing all of the filtered
            bounding boxes
        - box_classes: numpy.ndarray containing the class number
            for each box in filtered_boxes
        - box_scores: numpy.ndarray containing the box scores for each
            box in filtered_boxes

        Returns:
        - box_predictions: numpy.ndarray containing all of the predicted
            bounding boxes ordered by class and box score
        - predicted_box_classes: numpy.ndarray containing the class number
            for each box in box_predictions
        - predicted_box_scores: numpy.ndarray containing the box scores
            for each box in box_predictions
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_box_scores = box_scores[cls_mask]

            sorted_indices = np.argsort(cls_box_scores)[::-1]
            cls_boxes = cls_boxes[sorted_indices]
            cls_box_scores = cls_box_scores[sorted_indices]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_box_scores[0])

                if len(cls_boxes) == 1:
                    break

                iou = self._iou(cls_boxes[0], cls_boxes[1:])
                cls_boxes = cls_boxes[1:][iou < self.nms_t]
                cls_box_scores = cls_box_scores[1:][iou < self.nms_t]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def _iou(self, box1, boxes):
        """
        Calculate Intersection over Union (IoU) between box1 and boxes.

        Parameters:
        - box1: numpy.ndarray of shape (4,) containing a single bounding box
        - boxes: numpy.ndarray of shape (?, 4) containing multiple bounding
            boxes

        Returns:
        - iou: numpy.ndarray of shape (?) containing the IoU values
        """
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        iou = intersection / (box1_area + boxes_area - intersection)

        return iou

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a given folder path.

        Parameters:
        - folder_path: string representing the path to the folder
            holding all the images to load

        Returns:
        - images: list of images as numpy.ndarrays
        - image_paths: list of paths to the individual images in images
        """
        images = []
        image_paths = []

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"The directory \
                                    {folder_path} does not exist")

        valid_extensions = ('.jpg', '.jpeg', '.png')

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(image_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes and rescales images for the Darknet model.

        Parameters:
        - images: a list of images as numpy.ndarrays

        Returns:
        - pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
            containing all of the preprocessed images
        - image_shapes: a numpy.ndarray of shape (ni, 2) containing the
            original height and width of the images
        """
        pimages = []
        image_shapes = []
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for img in images:
            # Resize image with inter-cubic interpolation
            resized_img = cv2.resize(
                img, (input_h, input_w), interpolation=cv2.INTER_CUBIC)

            # Rescale pixel values from [0, 255] to [0, 1]
            pimages.append(resized_img / 255.0)

            # Add image shape to shapes array
            orig_h, orig_w = img.shape[:2]
            image_shapes.append([orig_h, orig_w])

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)
        return pimages, image_shapes
