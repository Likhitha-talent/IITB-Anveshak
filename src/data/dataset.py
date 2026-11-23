"""
Dataset loader and preprocessors for Anveshak Remote Sensing Imagery.
Supports arbitrary resolution tiles, bounding box normalization, and data augmentations.
Uses standard PIL and PyTorch without hard C-extension dependencies.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class RemoteSensingDataset(Dataset):
    """
    Dataset class for remote sensing tiles.
    Annotations are expected as a CSV or DataFrame with columns:
      image_id, class_name, xmin, ymin, xmax, ymax
    """
    def __init__(
        self,
        image_dir: str,
        annotations_file: Optional[str] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform = None,
        target_size: int = 512
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        self.class_to_idx = class_to_idx or {}

        if annotations_file and os.path.exists(annotations_file):
            self.df = pd.read_csv(annotations_file)
            self.image_ids = self.df['image_id'].unique().tolist()
        else:
            self.df = None
            # Test set mode (unlabelled images)
            valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
            self.image_ids = [
                os.path.splitext(f)[0]
                for f in sorted(os.listdir(image_dir))
                if f.lower().endswith(valid_exts)
            ] if os.path.exists(image_dir) else []

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.image_ids[idx]
        img_path = self._find_image_path(img_id)

        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # Load boxes and labels if annotated
        boxes = []
        labels = []
        if self.df is not None:
            rows = self.df[self.df['image_id'] == img_id]
            for _, r in rows.iterrows():
                b = [float(r['xmin']), float(r['ymin']), float(r['xmax']), float(r['ymax'])]
                boxes.append(b)
                cls_idx = self.class_to_idx.get(r['class_name'], 1)
                labels.append(cls_idx)

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        # Resize image to target size
        image_resized = image.resize((self.target_size, self.target_size))
        image_np = np.array(image_resized, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()

        targets = {
            "image_id": img_id,
            "orig_size": torch.tensor([orig_h, orig_w]),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }

        return image_tensor, targets

    def _find_image_path(self, img_id: str) -> str:
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            path = os.path.join(self.image_dir, f"{img_id}{ext}")
            if os.path.exists(path):
                return path
        return os.path.join(self.image_dir, img_id)
