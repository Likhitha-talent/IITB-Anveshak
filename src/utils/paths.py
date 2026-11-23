"""
Path discovery and dataset locator utility for Anveshak.
Auto-detects nested dataset folders (e.g., data/round1/Dataset_A, data/round1/test_images).
"""

import os
from typing import Dict, Optional, Tuple


def auto_detect_dataset_paths(base_dir: str) -> Dict[str, Optional[str]]:
    """
    Scans base_dir and base_dir/round1 to automatically resolve:
      - set_a_dir
      - set_a_annos
      - set_b_dir
      - set_b_annos
      - test_images_dir
      - validator_script
    """
    candidates = [base_dir, os.path.join(base_dir, "round1")]
    resolved = {
        "set_a_images": None,
        "set_a_annos": None,
        "set_b_images": None,
        "set_b_annos": None,
        "test_images": None,
        "validator_script": None
    }

    # Search for validator script
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "validate_predictions.py":
                resolved["validator_script"] = os.path.join(root, f)
            elif f.endswith(".csv"):
                fl = f.lower()
                if "set_a" in fl or "dataset_a" in fl or "dataset a" in fl:
                    resolved["set_a_annos"] = os.path.join(root, f)
                elif "set_b" in fl or "dataset_b" in fl or "dataset b" in fl:
                    resolved["set_b_annos"] = os.path.join(root, f)

    # Search for image directories
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            dl = d.lower()
            dpath = os.path.join(root, d)
            if "dataset a" in dl or "dataset_a" in dl or "set_a" in dl or dl == "a":
                # Check if it has an images subdirectory or contains images directly
                sub_img = os.path.join(dpath, "images")
                resolved["set_a_images"] = sub_img if os.path.exists(sub_img) else dpath
            elif "dataset b" in dl or "dataset_b" in dl or "set_b" in dl or dl == "b":
                sub_img = os.path.join(dpath, "images")
                resolved["set_b_images"] = sub_img if os.path.exists(sub_img) else dpath
            elif "test" in dl:
                sub_img = os.path.join(dpath, "images")
                resolved["test_images"] = sub_img if os.path.exists(sub_img) else dpath

    print(f"[*] Auto-detected dataset paths:")
    for k, v in resolved.items():
        print(f"    - {k}: {v}")

    return resolved
