"""
Path discovery and dataset locator utility for Anveshak.
Auto-detects nested dataset folders and annotation formats (CSV, XML, JSON, TXT).
"""

import os
from typing import Dict, Optional


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
                # Search inside dataset_A for annotations file or folder
                for r2, _, f2 in os.walk(dpath):
                    for fname in f2:
                        if fname.endswith((".csv", ".json", ".xml", ".txt")) and not fname.startswith("."):
                            resolved["set_a_annos"] = os.path.join(r2, fname)
                            break
                    if resolved["set_a_annos"]:
                        break

            elif "dataset_b" in dl or "dataset_b" in dl or dl == "dataset_b":
                sub_img = os.path.join(dpath, "images")
                resolved["set_b_images"] = sub_img if os.path.exists(sub_img) else dpath
                for r2, _, f2 in os.walk(dpath):
                    for fname in f2:
                        if fname.endswith((".csv", ".json", ".xml", ".txt")) and not fname.startswith("."):
                            resolved["set_b_annos"] = os.path.join(r2, fname)
                            break
                    if resolved["set_b_annos"]:
                        break

            elif "test" in dl:
                sub_img = os.path.join(dpath, "images")
                resolved["test_images"] = sub_img if os.path.exists(sub_img) else dpath

    print(f"[*] Auto-detected dataset paths:")
    for k, v in resolved.items():
        print(f"    - {k}: {v}")

    return resolved
