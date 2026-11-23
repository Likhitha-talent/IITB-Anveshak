"""
Training Script for Set A (Base Detector from Scratch).
Features:
  - Trains from random initialization (NO pretrained weights per competition rules)
  - Group Normalization stability
  - FP16 Automatic Mixed Precision (AMP) to prevent CUDA OOM
  - Gradient Accumulation to maintain effective batch size
  - Saves best checkpoint to `checkpoints/set_a_best.pth`
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.detector import build_scratch_detector
from src.data.dataset import RemoteSensingDataset
from src.utils.paths import auto_detect_dataset_paths


def collate_fn(batch):
    return tuple(zip(*batch))


def train_set_a(
    config_path: str = "configs/base_config.yaml",
    data_dir: str = "data",
    output_dir: str = "checkpoints"
):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on device: {device}")

    # Set CUDA memory fragmentation allocation conf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[*] CUDA Initialized: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)
    num_classes = cfg['model']['num_classes']

    print("[*] Initializing detector with GroupNorm from random initialization...")
    model = build_scratch_detector(num_classes=num_classes, pretrained=False)
    model.to(device)

    # Mixed precision scaler
    use_amp = cfg['training_set_a'].get('mixed_precision', True) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = cfg['training_set_a'].get('gradient_accumulation_steps', 4)
    print(f"[*] Mixed Precision (AMP): {use_amp} | Gradient Accumulation Steps: {accum_steps}")

    # Optimizer with weight decay
    lr = float(cfg['training_set_a']['base_lr'])
    weight_decay = float(cfg['training_set_a']['weight_decay'])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=cfg['training_set_a']['momentum'], weight_decay=weight_decay)

    epochs = cfg['training_set_a']['epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(cfg['training_set_a']['min_lr']))

    # Setup dataset loader with auto-path resolution
    paths = auto_detect_dataset_paths(data_dir)
    set_a_images = paths["set_a_images"]
    set_a_annos = paths["set_a_annos"]

    batch_size = cfg['training_set_a']['batch_size']
    if set_a_images and os.path.exists(set_a_images):
        dataset = RemoteSensingDataset(image_dir=set_a_images, annotations_file=set_a_annos, target_size=cfg['data']['image_size'])
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg['system'].get('num_workers', 2),
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        print(f"[✓] Loaded {len(dataset)} training samples for Set A (Batch size: {batch_size}, Effective: {batch_size * accum_steps}).")
    else:
        print(f"[!] Warning: Data path '{set_a_images}' not found.")
        dataloader = []

    best_loss = float('inf')
    save_path = os.path.join(output_dir, "set_a_best.pth")

    print("[*] Ready to train Set A.")
    for epoch in range(1, epochs + 1):
        if not dataloader:
            torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, save_path)
            print(f"[✓] Initial model checkpoint saved to {save_path}")
            break

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for step, (images, targets) in enumerate(pbar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss = losses / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += losses.item()
            pbar.set_postfix(loss=round(losses.item(), 4))

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "loss": best_loss}, save_path)
            print(f"[*] New best model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train_set_a(args.config, args.data_dir, args.output_dir)
