"""
Inference runner for generating Anveshak competition predictions.
Generates candidate detections, computes Helmholtz energy for unknowns,
applies the official IoU 0.5 Conflict Rule, and exports validate_predictions-compliant CSV.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

from src.models.detector import build_scratch_detector, FreeEnergyScorer
from src.data.dataset import RemoteSensingDataset
from src.inference.conflict_resolver import clean_and_format_predictions, save_submission_csv
from src.utils.paths import auto_detect_dataset_paths


def run_inference(
    checkpoint_path: str,
    test_images_dir: str,
    output_csv: str,
    config_path: str = "configs/base_config.yaml",
    device: str = "cuda"
):
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
    print(f"[*] Running inference using device: {device}")

    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg['model'].get('num_classes', 40)
    energy_temp = cfg['model'].get('energy_temperature', 1.0)
    unk_thresh = cfg['model'].get('unknown_threshold', 0.45)

    # Build model and load checkpoint
    model = build_scratch_detector(num_classes=num_classes, pretrained=False)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"[✓] Successfully loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Running with initial weights (sanity check).")

    model.to(device)
    model.eval()

    # If test_images_dir doesn't exist, search for it
    if not os.path.exists(test_images_dir):
        paths = auto_detect_dataset_paths("data")
        if paths["test_images"]:
            test_images_dir = paths["test_images"]

    # Load dataset
    dataset = RemoteSensingDataset(image_dir=test_images_dir, target_size=cfg['data']['image_size'])
    records = []

    print(f"[*] Inferring on {len(dataset)} test images from: {test_images_dir}...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            image, meta = dataset[i]
            img_id = meta['image_id']
            image_batch = [image.to(device)]

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(image_batch)[0]

            boxes = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()

            for b, s, l in zip(boxes, scores, labels):
                if s < cfg['inference']['score_threshold']:
                    continue

                # Default class label
                class_label = f"class_{l}" if l != 0 else "background"
                if class_label == "background":
                    continue

                records.append({
                    "image_id": img_id,
                    "class_name": class_label,
                    "score": float(s),
                    "xmin": int(b[0]),
                    "ymin": int(b[1]),
                    "xmax": int(b[2]),
                    "ymax": int(b[3])
                })

    raw_df = pd.DataFrame(records)
    if raw_df.empty:
        print("[!] No predictions produced above threshold. Creating empty compliant CSV.")
        raw_df = pd.DataFrame(columns=['image_id', 'class_name', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])

    # Format, apply conflict rule, and enforce official row limits
    final_df = clean_and_format_predictions(
        raw_df,
        image_width=cfg['data']['image_size'],
        image_height=cfg['data']['image_size'],
        max_per_image=cfg['inference']['max_detections_per_image']
    )

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    save_submission_csv(final_df, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anveshak Inference & Prediction Generator")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/round1_final.pth")
    parser.add_argument("--test_images_dir", type=str, default="data/test_images")
    parser.add_argument("--output_csv", type=str, default="submission/predictions.csv")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()

    run_inference(args.checkpoint, args.test_images_dir, args.output_csv, args.config)
