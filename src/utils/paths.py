"""
Path discovery and dataset locator utility for Anveshak.
Auto-detects nested dataset folders and prioritizes true annotation files (annotations.csv, labels.csv)
over metadata files (tiles.csv).
"""

import os
import pandas as pd
from typing import Dict, Optional, List


def is_annotation_file(filepath: str) -> bool:
    """Checks if a CSV/JSON file contains bounding box coordinates."""
    if not os.path.isfile(filepath):
        return False
    fl = os.path.basename(filepath).lower()
    # High-confidence naming
    if any(k in fl for k in ["annot", "label", "bbox", "box", "gt", "target", "train"]):
        return True
    if filepath.endswith('.csv'):
        try:
            sample = pd.read_csv(filepath, nrows=2)
            cols = [c.lower() for c in sample.columns]
            has_coords = any(k in cols for k in ['xmin', 'ymin', 'xmax', 'ymax', 'x_min', 'y_min', 'x', 'y', 'w', 'h', 'bbox'])
            return has_coords
        except Exception:
            return False
    return False


def find_annotation_file(dpath: str) -> Optional[str]:
    """Finds the true annotation file inside a dataset directory."""
    all_files = []
    for root, _, files in os.walk(dpath):
        for f in files:
            if not f.startswith("."):
                all_files.append(os.path.join(root, f))

    # 1. Prioritize files named with 'annot', 'label', 'bbox', 'train'
    annot_candidates = [f for f in all_files if any(k in os.path.basename(f).lower() for k in ["annot", "label", "bbox", "target", "train"]) and f.endswith(('.csv', '.json', '.xml', '.txt'))]
    if annot_candidates:
        return annot_candidates[0]

    # 2. Check all CSVs to see which one has bounding box coordinates
    csv_candidates = [f for f in all_files if f.endswith('.csv')]
    for c in csv_candidates:
        if is_annotation_file(c):
            return c

    # 3. Fallback to any JSON or XML
    for f in all_files:
        if f.endswith(('.json', '.xml')):
            return f

    # 4. Fallback to any CSV if nothing else
    if csv_candidates:
        return csv_candidates[0]

    return None


def auto_detect_dataset_paths(base_dir: str) -> Dict[str, Optional[str]]:
    resolved = {
        "set_a_images": None,
        "set_a_annos": None,
        "set_b_images": None,
        "set_b_annos": None,
        "test_images": None,
        "validator_script": None
    }

    # 1. Search for validator script
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "validate_predictions.py":
                resolved["validator_script"] = os.path.join(root, f)

    # 2. Search for dataset folders
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            dl = d.lower()
            dpath = os.path.join(root, d)

            if "dataset_a" in dl or "dataset a" in dl or dl == "dataset_a":
                sub_img = os.path.join(dpath, "images")
                resolved["set_a_images"] = sub_img if os.path.exists(sub_img) else dpath
                resolved["set_a_annos"] = find_annotation_file(dpath)

            elif "dataset_b" in dl or "dataset_b" in dl or dl == "dataset_b":
                sub_img = os.path.join(dpath, "images")
                resolved["set_b_images"] = sub_img if os.path.exists(sub_img) else dpath
                resolved["set_b_annos"] = find_annotation_file(dpath)

            elif "test" in dl:
                sub_img = os.path.join(dpath, "images")
                resolved["test_images"] = sub_img if os.path.exists(sub_img) else dpath

    print(f"[*] Auto-detected dataset paths:")
    for k, v in resolved.items():
        print(f"    - {k}: {v}")

    return resolved
