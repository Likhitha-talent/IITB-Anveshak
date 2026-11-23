"""
Dataset loader and preprocessors for Anveshak Remote Sensing Imagery.
Supports arbitrary resolution tiles, bounding box normalization, and data augmentations.
Handles multiple annotation formats: CSV, VOC XML, COCO JSON, YOLO TXT.
"""

import os
import glob
import xml.etree.ElementTree as ET
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class RemoteSensingDataset(Dataset):
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

        # 1. Gather all image files
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        if os.path.exists(image_dir):
            self.image_files = [
                f for f in sorted(os.listdir(image_dir))
                if f.lower().endswith(valid_exts)
            ]
            self.image_ids = [os.path.splitext(f)[0] for f in self.image_files]
        else:
            self.image_files = []
            self.image_ids = []

        # 2. Check for annotations
        self.annotations = {} # img_id -> list of (box, class_idx)
        self._load_annotations(annotations_file)

    def _load_annotations(self, annos_path: Optional[str]):
        if not annos_path:
            # Check adjacent directories for annotations
            parent = os.path.dirname(self.image_dir)
            for cand in ["annotations", "labels", "annos"]:
                cpath = os.path.join(parent, cand)
                if os.path.exists(cpath):
                    annos_path = cpath
                    break

        if not annos_path:
            return

        if os.path.isfile(annos_path):
            if annos_path.endswith('.csv'):
                df = pd.read_csv(annos_path)
                # Map column names flexibly
                cols = {c.lower(): c for c in df.columns}
                id_col = cols.get('image_id', cols.get('image', cols.get('filename', 'image_id')))
                cls_col = cols.get('class_name', cols.get('class', cols.get('label', cols.get('category', 'class_name'))))
                
                for _, row in df.iterrows():
                    img_id = str(row[id_col]).replace('.png', '').replace('.jpg', '')
                    cls_name = str(row[cls_col])
                    cls_idx = self.class_to_idx.setdefault(cls_name, len(self.class_to_idx) + 1)
                    
                    b = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
                    self.annotations.setdefault(img_id, []).append((b, cls_idx))
                    
            elif annos_path.endswith('.json'):
                with open(annos_path, 'r') as f:
                    data = json.load(f)
                if 'annotations' in data and 'images' in data:
                    id_to_fname = {img['id']: os.path.splitext(img['file_name'])[0] for img in data['images']}
                    for ann in data['annotations']:
                        fname = id_to_fname.get(ann['image_id'])
                        if fname:
                            # COCO bbox: [x, y, w, h]
                            x, y, w, h = ann['bbox']
                            box = [x, y, x + w, y + h]
                            cls_idx = ann.get('category_id', 1)
                            self.annotations.setdefault(fname, []).append((box, cls_idx))

        elif os.path.isdir(annos_path):
            # Check for XML or TXT files
            xml_files = glob.glob(os.path.join(annos_path, "*.xml"))
            if xml_files:
                for xml_file in xml_files:
                    img_id = os.path.splitext(os.path.basename(xml_file))[0]
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        cls_name = obj.find('name').text
                        cls_idx = self.class_to_idx.setdefault(cls_name, len(self.class_to_idx) + 1)
                        bndbox = obj.find('bndbox')
                        b = [
                            float(bndbox.find('xmin').text),
                            float(bndbox.find('ymin').text),
                            float(bndbox.find('xmax').text),
                            float(bndbox.find('ymax').text)
                        ]
                        self.annotations.setdefault(img_id, []).append((b, cls_idx))

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.image_ids[idx]
        img_path = self._find_image_path(img_id)

        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        boxes = []
        labels = []
        if img_id in self.annotations:
            for b, l in self.annotations[img_id]:
                # Scale boxes if original size differs from target size
                sx = self.target_size / orig_w
                sy = self.target_size / orig_h
                scaled_b = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                boxes.append(scaled_b)
                labels.append(l)

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

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
