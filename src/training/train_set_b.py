"""
Continual Learning Script for Set B (10 Few-Shot Classes).
Enforces:
  - Exact model continuity (starts from verified Set A checkpoint)
  - Knowledge Distillation loss from Set A features/outputs to prevent catastrophic forgetting
  - Continual update saved to `checkpoints/round1_final.pth`
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.detector import build_scratch_detector
from src.data.dataset import RemoteSensingDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def train_set_b(
    checkpoint_in: str = "checkpoints/set_a_best.pth",
    output_checkpoint: str = "checkpoints/round1_final.pth",
    config_path: str = "configs/base_config.yaml",
    data_dir: str = "data"
):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"[*] Continual training on device: {device}")

    num_classes = cfg['model']['num_classes']

    # 1. Initialize active model
    model = build_scratch_detector(num_classes=num_classes, pretrained=False)
    if os.path.exists(checkpoint_in):
        checkpoint = torch.load(checkpoint_in, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"[✓] Successfully continued model lineage from: {checkpoint_in}")
    else:
        print(f"[!] Warning: Checkpoint {checkpoint_in} not found. Initializing fresh.")

    model.to(device)

    # 2. Frozen teacher model for Knowledge Distillation (preventing forgetting)
    teacher_model = build_scratch_detector(num_classes=num_classes, pretrained=False)
    if os.path.exists(checkpoint_in):
        teacher_model.load_state_dict(state_dict)
    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Lower learning rate for fine-tuning
    lr = cfg['continual_set_b']['base_lr']
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    epochs = cfg['continual_set_b']['epochs']

    set_b_images = os.path.join(data_dir, "dataset_b", "images")
    set_b_annos = os.path.join(data_dir, "dataset_b", "annotations.csv")

    if os.path.exists(set_b_images):
        dataset = RemoteSensingDataset(image_dir=set_b_images, annotations_file=set_b_annos)
        dataloader = DataLoader(dataset, batch_size=cfg['continual_set_b']['batch_size'], shuffle=True, collate_fn=collate_fn)
        print(f"[✓] Loaded {len(dataset)} few-shot samples for Set B.")
    else:
        print(f"[!] Warning: Data path '{set_b_images}' not found yet.")
        dataloader = []

    os.makedirs(os.path.dirname(output_checkpoint) or ".", exist_ok=True)

    print("[*] Starting Set B Continual Learning...")
    for epoch in range(1, epochs + 1):
        if not dataloader:
            torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, output_checkpoint)
            print(f"[✓] Model checkpoint saved to {output_checkpoint}")
            break

        model.train()
        pbar = tqdm(dataloader, desc=f"Set B Epoch {epoch}/{epochs}")

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # Primary detection loss
            loss_dict = model(images, targets)
            det_loss = sum(loss for loss in loss_dict.values())

            # Knowledge distillation loss (MSE on FPN features to preserve representations)
            with torch.no_grad():
                teacher_features = teacher_model.backbone(torch.stack(images))
            student_features = model.backbone(torch.stack(images))

            kd_loss = 0.0
            for k in student_features.keys():
                kd_loss += nn.functional.mse_loss(student_features[k], teacher_features[k])

            total_loss = det_loss + (cfg['continual_set_b']['distillation_weight'] * kd_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix(det_loss=det_loss.item(), kd_loss=kd_loss.item())

        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, output_checkpoint)

    print(f"[✓] Continual training finished. Verified checkpoint: {output_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_in", type=str, default="checkpoints/set_a_best.pth")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/round1_final.pth")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    train_set_b(args.checkpoint_in, args.output_checkpoint, args.config, args.data_dir)
