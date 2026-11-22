# Anveshak: Open-World Object Detection for Remote Sensing
### Techfest 2026-27 | IITB-HRTI Centre of Excellence

---

## 🏗️ System Architecture & Workflow

This project is organized into an asynchronous, reproducible loop between local development and cloud GPU training:

```
              ANVESHAK PROJECT
                     │
          ┌──────────┴──────────┐
          │                     │
       GitHub                Google Drive
          │                     │
     Source code            Datasets
     Model code             Checkpoints
     Configurations         Experiment outputs
          │
     ┌────┴─────┐
     │          │
   VS Code   Colab
     │          │
  Develop    GPU training
     │          │
     └────┬─────┘
          │
     Evaluation
          │
   Final submission
```

---

## 🚀 Quickstart & Directory Layout

### 1. Google Drive Directory Structure
Create the following folder structure in your Google Drive:
```
MyDrive/
└── Anveshak/
    ├── datasets/
    │   ├── dataset_a.zip
    │   ├── dataset_b.zip
    │   └── test_images.zip
    ├── checkpoints/
    │   ├── set_a_best.pth
    │   └── round1_final.pth
    └── submissions/
        ├── predictions.csv
        ├── checkpoint.pth
        └── methodology.pptx
```

### 2. Local Setup (VS Code)
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd IITBBB
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Google Colab Training Workflow
1. Open [`notebooks/anveshak_colab_runner.ipynb`](notebooks/anveshak_colab_runner.ipynb) in Google Colab with GPU runtime (T4 / A100 / V100).
2. The notebook handles:
   - Mounting Google Drive (`/content/drive/MyDrive/Anveshak`).
   - Extracting datasets to fast Colab NVMe SSD (`/content/data/`) for high IOPS.
   - Training Set A from scratch (random initialization, no pretrained weights).
   - Incremental continuation on Set B (distillation + cosine head margin).
   - Syncing checkpoints back to Google Drive automatically.
   - Running inference on test images and generating `predictions.csv`.
   - Running `validate_predictions.py` on the output CSV.

---

## ⚙️ Core Technical Pillars

1. **Training from Scratch (Zero Pretrained Weights):**
   - Standard detectors fail to converge from scratch with small batch sizes due to Batch Normalization drift.
   - We utilize **Group Normalization (GN)** + Linear Warmup + Cosine Annealing learning rate schedule.
2. **Open-World Unknown Mining:**
   - Combines RPN high-objectness proposals with **Helmholtz Free Energy score** over known class logits to detect out-of-distribution (OOD) unknown objects with high precision.
3. **Continual Few-Shot Learning (Set A → Set B):**
   - Fixed cosine prototype separation with bounded margin loss.
   - Feature distillation ($\mathcal{L}_{\text{KD}}$) from Set A checkpoint to prevent catastrophic forgetting.
4. **Conflict Resolution:**
   - Strict implementation of the competition rule: When known and unknown boxes overlap at $\text{IoU} \ge 0.5$, highest confidence score prevails.

---

## 📊 Deliverables Checklist for Round 1
- [ ] `predictions.csv` passing `validate_predictions.py`.
- [ ] `round1_final.pth` checkpoint reproducing predictions within tolerance.
- [ ] Modular training & inference code with this README.
- [ ] 5–7 Slide Methodology Presentation PPT.
- [ ] Single Google Drive link set to *"Anyone with the link can view"*.
