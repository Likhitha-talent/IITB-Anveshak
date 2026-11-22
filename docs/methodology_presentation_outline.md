# Anveshak Methodology Presentation (5–7 Slides Blueprint)
### Official Round 1 Deliverable | Techfest 2026-27

---

### Slide 1: Title & Team Overview
- **Title:** Open-World Object Detection & Continual Learning for Remote Sensing
- **Subtitle:** Anveshak Challenge — Techfest 2026-27, IIT Bombay
- **Team Name / ID:** [Insert Team ID]
- **Team Members & Institutional Affiliations:**
  - Member 1 (Team Leader): [Name] — Pipeline Architecture & Continual Learning
  - Member 2: [Name] — Open-World Unknown Head & Free Energy Scoring
  - Member 3: [Name] — Training from Scratch Optimization & Data Augmentations
  - Member 4: [Name] — Metric Calibration, Conflict Resolution & Inference Benchmarking

---

### Slide 2: Problem Understanding
- **Core Realities in Remote Sensing (EO):**
  1. *The Open-World Dilemma:* Satellite/aerial imagery is inherently open-ended. Fixed-taxonomy detectors hallucinate unseen objects into known classes, introducing critical silent errors.
  2. *The Few-Shot Reality:* New threat/target categories emerge with minimal annotated instances (10 classes from few images). Retraining from scratch is computationally infeasible.
  3. *Catastrophic Forgetting:* Standard stochastic gradient descent overwrites old class representations when optimizing for new ones.
- **Rules & Scorer Nuances:**
  - Strict zero pretrained weights constraint (training strictly from random initialization).
  - Model continuity constraint (one continuous checkpoint lineage).
  - Evaluated on **OE-Score** (harmonic mean of $\text{mAP}_K$ and $\text{U-F1}$).
  - Strict **Conflict Rule** ($\text{IoU} \ge 0.5$ between known and unknown: higher score survives).

---

### Slide 3: Methodology — Core Architectural Pillars
- **1. Scratch-Trainable Backbone:**
  - Replaced BatchNorm with **Group Normalization (GN)** across all stages to eliminate batch-size drift.
  - Linear Warmup followed by **Cosine Annealing** for stable convergence from random initialization without ImageNet priors.
- **2. Energy-Based Unknown Discovery (OWOD Head):**
  - Uses the Helmholtz Free Energy over classification logits:
    $$E(x) = -T \cdot \log \sum_{i=1}^K \exp(f_i(x) / T)$$
  - Combines RPN high-objectness proposals with high free energy to detect unlabelled unknown objects while rejecting background clutter.
- **3. Normalized Cosine Classification Head:**
  - Replaced unconstrained dot-product linear layers with $L_2$-normalized cosine similarities and angular margin to prevent magnitude explosion.

---

### Slide 4: Set A → Set B Continual Learning & Anti-Forgetting
- **Single Checkpoint Lineage:**
  - Checkpoint from Set A (`set_a_best.pth`) is loaded directly into the Set B training pipeline.
- **Dual-Constraint Anti-Forgetting Mechanism:**
  1. *Feature-Level Knowledge Distillation:*
     $$\mathcal{L}_{\text{distill}} = \frac{1}{|F|} \sum_{l \in FPN} \|f_{l}^{\text{student}}(x) - f_{l}^{\text{teacher}}(x)\|_2^2$$
     Anchors the FPN multi-scale feature space to Set A teacher representations.
  2. *Cosine Weight Regularization:*
     Base class prototype vectors are preserved; new classes are allocated in orthogonal representation subspaces.

---

### Slide 5: Implementation, Validation & Metrics
- **Pipeline Implementation:**
  - Automated VS Code $\leftrightarrow$ GitHub $\leftrightarrow$ Google Colab $\leftrightarrow$ Google Drive architecture.
  - Full compliance with `validate_predictions.py` and output formatting.
- **Conflict Resolution Engine:**
  - Bounding boxes are filtered by the official competition rule: known and unknown overlapping at $\text{IoU} \ge 0.5$ are resolved in favor of the highest score (ties go to earlier row).
- **Internal Validation Results:**
  - Demonstrates strong $\text{mAP}_K$ on base classes, balanced unknown recall/precision ($\text{U-F1}$), yielding an optimal **OE-Score**.

---

### Slide 6: Plan for Round 2 (On-Site Rapid Adaptation)
- **Turnkey Adaptation Protocol (16th December):**
  1. *Immediate Checkpoint Ingestion:* Verified Round 1 checkpoint is loaded into memory within minutes of dataset release.
  2. *10 Fresh Classes Expansion:* Dynamic expansion of the cosine classification head from $K=40$ to $K=50$ classes.
  3. *Balanced Few-Shot Replay:* Fast fine-tuning using low learning rate with distillation from the verified Round 1 checkpoint.
  4. *Unknown Threshold Re-calibration:* Adaptive thresholding to ensure new classes do not leak into the unknown pool while preserving unknown discovery on true unseen categories.
  5. *Automated Submission Pipeline:* Instant CSV export and validation to ensure submission before window close.
