"""
Remote Sensing Specific Preprocessing & Augmentation Pipeline.
Designed specifically for training detectors from scratch on satellite/aerial imagery:
  1. Rotation Invariance (0, 90, 180, 270 deg) - remote sensing targets have arbitrary orientations
  2. Horizontal & Vertical Flips - aerial imagery has no canonical 'up' direction
  3. Atmospheric / Radiometric Jitter - handles sunlight variation, cloud shadows, and haze
  4. Bounding Box Transformation & Validation
  5. Channel Standardization
"""

import random
import torch
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, List, Dict


class RemoteSensingTrainTransform:
    def __init__(self, target_size: int = 512):
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, image: Image.Image, boxes: np.ndarray, labels: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        w, h = image.size

        # 1. Random Horizontal Flip (p=0.5)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes) > 0:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        # 2. Random Vertical Flip (p=0.5) - Aerial imagery is rotationally symmetric
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if len(boxes) > 0:
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

        # 3. Random 90-degree Rotation (p=0.5)
        if random.random() < 0.5:
            image = image.transpose(Image.ROTATE_90)
            if len(boxes) > 0:
                # [x1, y1, x2, y2] -> [y1, w - x2, y2, w - x1]
                new_boxes = boxes.copy()
                new_boxes[:, 0] = boxes[:, 1]
                new_boxes[:, 1] = w - boxes[:, 2]
                new_boxes[:, 2] = boxes[:, 3]
                new_boxes[:, 3] = w - boxes[:, 0]
                # Fix inverted mins/maxes
                boxes[:, 0] = np.minimum(new_boxes[:, 0], new_boxes[:, 2])
                boxes[:, 1] = np.minimum(new_boxes[:, 1], new_boxes[:, 3])
                boxes[:, 2] = np.maximum(new_boxes[:, 0], new_boxes[:, 2])
                boxes[:, 3] = np.maximum(new_boxes[:, 1], new_boxes[:, 3])
                w, h = h, w

        # 4. Radiometric / Atmospheric Jitter (Brightness & Contrast)
        if random.random() < 0.4:
            enh_b = ImageEnhance.Brightness(image)
            image = enh_b.enhance(random.uniform(0.8, 1.2))
            enh_c = ImageEnhance.Contrast(image)
            image = enh_c.enhance(random.uniform(0.8, 1.2))

        # 5. Scale to target tile size
        image_resized = image.resize((self.target_size, self.target_size))
        sx = self.target_size / w
        sy = self.target_size / h

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            # Clip bounds
            boxes[:, 0] = np.clip(boxes[:, 0], 0, self.target_size - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, self.target_size - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 1, self.target_size)
            boxes[:, 3] = np.clip(boxes[:, 3], 1, self.target_size)

            # Filter valid boxes
            valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[valid]
            labels = labels[valid]

        # 6. Tensor conversion & Channel Standardization
        img_np = np.array(image_resized, dtype=np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

        return tensor, boxes, labels


class RemoteSensingValTransform:
    def __init__(self, target_size: int = 512):
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, image: Image.Image, boxes: np.ndarray, labels: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        w, h = image.size
        image_resized = image.resize((self.target_size, self.target_size))
        sx = self.target_size / w
        sy = self.target_size / h

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            boxes[:, 0] = np.clip(boxes[:, 0], 0, self.target_size - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, self.target_size - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 1, self.target_size)
            boxes[:, 3] = np.clip(boxes[:, 3], 1, self.target_size)

            valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[valid]
            labels = labels[valid]

        img_np = np.array(image_resized, dtype=np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

        return tensor, boxes, labels
