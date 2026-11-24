"""
Dataset loader and preprocessors for Anveshak Remote Sensing Imagery.
Supports arbitrary resolution tiles, bounding box normalization, and data augmentations.
Handles individual per-image JSON files in labels/ directories (Labelme, Roboflow, VOC, etc.),
COCO JSON, XML files, and CSV files.
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
        self.annotations = {}  # img_id -> list of (box, class_idx)
        self._load_annotations(annotations_file)

    def _parse_single_json(self, data, img_id: str):
        """Parses an individual JSON file for a single image."""
        boxes = []
        # Format 1: Labelme format ("shapes": [{"label": ..., "points": [[x1, y1], [x2, y2]]}])
        if isinstance(data, dict) and "shapes" in data:
            for s in data["shapes"]:
                cls_name = str(s.get("label", "object"))
                cls_idx = self.class_to_idx.setdefault(cls_name, len(self.class_to_idx) + 1)
                pts = np.array(s.get("points", []))
                if len(pts) >= 2:
                    xmin, ymin = pts.min(axis=0)[:2]
                    xmax, ymax = pts.max(axis=0)[:2]
                    boxes.append(([float(xmin), float(ymin), float(xmax), float(ymax)], cls_idx))

        # Format 2: Objects/annotations list in dict
        elif isinstance(data, dict) and any(k in data for k in ["objects", "annotations", "labels", "bboxes", "targets"]):
            items = data.get("objects", data.get("annotations", data.get("labels", data.get("bboxes", []))))
            for item in items:
                cls_name = str(item.get("category", item.get("class", item.get("label", item.get("name", "object")))))
                cls_idx = self.class_to_idx.setdefault(cls_name, len(self.class_to_idx) + 1)
                b = item.get("bbox", item.get("box", item.get("coordinates", [])))
                if len(b) == 4:
                    # Check if [x, y, w, h] or [xmin, ymin, xmax, ymax]
                    if b[2] < b[0] or b[3] < b[1]:
                        # Likely [x, y, w, h]
                        b = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
                    boxes.append(([float(b[0]), float(b[1]), float(b[2]), float(b[3])], cls_idx))

        # Format 3: Direct list of dicts
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    cls_name = str(item.get("category", item.get("class", item.get("label", item.get("name", "object")))))
                    cls_idx = self.class_to_idx.setdefault(cls_name, len(self.class_to_idx) + 1)
                    b = item.get("bbox", item.get("box", []))
                    if len(b) == 4:
                        if b[2] < b[0] or b[3] < b[1]:
                            b = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
                        boxes.append(([float(b[0]), float(b[1]), float(b[2]), float(b[3])], cls_idx))

        if boxes:
            self.annotations[img_id] = boxes

    def _load_annotations(self, annos_path: Optional[str]):
        if not annos_path:
            parent = os.path.dirname(self.image_dir)
            for cand in ["labels", "annotations", "annos"]:
                cpath = os.path.join(parent, cand)
                if os.path.exists(cpath):
                    annos_path = cpath
                    break

        if not annos_path or not os.path.exists(annos_path):
            return

        print(f"[*] Parsing annotations from: {annos_path}")

        # Check if annos_path is a directory or points to a file in a labels/ directory
        labels_dir = None
        if os.path.isdir(annos_path):
            labels_dir = annos_path
        elif os.path.isfile(annos_path):
            parent_dir = os.path.dirname(annos_path)
            if os.path.basename(parent_dir).lower() in ["labels", "annotations"]:
                labels_dir = parent_dir

        # Case 1: Directory with per-image JSON/XML/TXT files
        if labels_dir:
            json_files = glob.glob(os.path.join(labels_dir, "*.json"))
            if json_files:
                print(f"[*] Found {len(json_files)} individual JSON label files in {labels_dir}. Parsing...")
                for jf in json_files:
                    img_id = os.path.splitext(os.path.basename(jf))[0]
                    try:
                        with open(jf, 'r') as f:
                            data = json.load(f)
                        self._parse_single_json(data, img_id)
                    except Exception:
                        pass
                print(f"[✓] Parsed annotations for {len(self.annotations)} images. Classes: {len(self.class_to_idx)}")
                if self.annotations:
                    return

            xml_files = glob.glob(os.path.join(labels_dir, "*.xml"))
            if xml_files:
                print(f"[*] Found {len(xml_files)} XML label files in {labels_dir}. Parsing...")
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
                print(f"[✓] Parsed annotations for {len(self.annotations)} images.")
                if self.annotations:
                    return

        # Case 2: Monolithic CSV file
        if os.path.isfile(annos_path) and annos_path.endswith('.csv'):
            try:
                df = pd.read_csv(annos_path)
                cols = {c.lower(): c for c in df.columns}
                print(f"    Available columns: {list(df.columns)}")

                id_candidates = ['image_id', 'tile_id', 'tile_name', 'image', 'img', 'filename', 'file_name', 'name', 'id', 'file']
                id_col = next((cols[c] for c in id_candidates if c in cols), df.columns[0])

                cls_candidates = ['class_name', 'class', 'label', 'category', 'category_id', 'class_id', 'cat', 'type']
                cls_col = next((cols[c] for c in cls_candidates if c in cols), None)

                xmin_col = cols.get('xmin', cols.get('x_min', cols.get('minx', cols.get('x1', cols.get('left')))))
                ymin_col = cols.get('ymin', cols.get('y_min', cols.get('miny', cols.get('y1', cols.get('top')))))
                xmax_col = cols.get('xmax', cols.get('x_max', cols.get('maxx', cols.get('x2', cols.get('right')))))
                ymax_col = cols.get('ymax', cols.get('y_max', cols.get('maxy', cols.get('y2', cols.get('bottom')))))

                w_col = cols.get('width', cols.get('w'))
                h_col = cols.get('height', cols.get('h'))
                x_col = cols.get('x')
                y_col = cols.get('y')

                for _, row in df.iterrows():
                    val = row[id_col]
                    if pd.isna(val):
                        continue
                    img_id = os.path.splitext(str(val))[0]

                    cls_name = str(row[cls_col]) if cls_col and not pd.isna(row[cls_col]) else "known_object"
                    cls_idx = self.class_to_idx.setdefault(cls_name, len(self.class_to_idx) + 1)

                    if xmin_col and ymin_col and xmax_col and ymax_col:
                        b = [float(row[xmin_col]), float(row[ymin_col]), float(row[xmax_col]), float(row[ymax_col])]
                        self.annotations.setdefault(img_id, []).append((b, cls_idx))
                    elif x_col and y_col and w_col and h_col:
                        x, y, w, h = float(row[x_col]), float(row[y_col]), float(row[w_col]), float(row[h_col])
                        b = [x, y, x + w, y + h]
                        self.annotations.setdefault(img_id, []).append((b, cls_idx))

                print(f"[✓] Loaded annotations for {len(self.annotations)} images from CSV. Classes: {len(self.class_to_idx)}")
            except Exception as e:
                print(f"[!] Error parsing CSV {annos_path}: {e}")

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
                sx = self.target_size / orig_w
                sy = self.target_size / orig_h
                scaled_b = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                if scaled_b[2] > scaled_b[0] and scaled_b[3] > scaled_b[1]:
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
