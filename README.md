# Per-Zone Uveitis Classification Pipeline

This repository contains a deep learning pipeline for classifying retinal zones in fundus photography (FP) images. On this branch the task is **binary**: spreadsheet severity tiers **1 and 2 are merged into one positive class** versus tier **0**. The pipeline handles preprocessing, zone extraction, and ConvNeXt training with weighted cross-entropy or focal loss for remaining class imbalance.

## Pipeline Overview

The workflow is divided into two main stages: **Data Preparation** and **Model Training**.

### 1. Data Preparation Stage
This stage converts raw patient data into a standardized format suitable for training.

*   **`extract_zones.py`**: A self-contained module that programmatically extracts 10 retinal zones from a fundus image. It automatically detects the fovea/crosshair, removes yellow overlays, and returns cropped zone arrays.
*   **`pre_processing.py`**: The main entry point for data ingestion.
    *   Walks through the `data/` directory (organized by `PatientID/VisitDate/`).
    *   Standardizes images (e.g., horizontally flipping OS images to match the OD-axis convention).
    *   Saves visit-wise standardized arrays as `.npz` files.
    *   Matches images with labels from an annotations spreadsheet.
    *   Calls `extract_zones.py` to generate zone crops and builds `zone_training_table.csv` with **binary** `Zone_Label` values (0 vs 1) by default. Use `--multiclass-zone-labels` if you need raw 0/1/2 in the CSV; training still maps to binary on load.

### 2. Model Training Stage
This stage uses the generated zone crops and labels to train a classifier.

*   **`zone_dataset.py`**: Manages data loading for PyTorch.
    *   Implements a **Patient-Level Split** to ensure that data from the same patient does not leak across training, validation, and test sets.
    *   Provides the `ZoneImageDataset` class for loading zone images and their corresponding labels.
*   **`losses.py`**: Contains loss function implementations.
    *   **CrossEntropyLoss**: Standard classification loss (supports class weighting).
    *   **Focal Loss**: A specialized loss function that down-weights easy examples and focuses training on hard-to-classify samples, ideal for the class imbalance found in this dataset.
*   **`train_convnext.py`**: The main training script.
    *   Uses a pretrained ConvNeXt-Tiny backbone.
    *   Supports toggling between `ce` (CrossEntropy) and `focal` loss via command-line arguments.
    *   Two output logits (classes 0 and 1). Tracks Accuracy, Balanced Accuracy, Macro-F1, per-class recall/specificity, and a 2×2 confusion matrix.
    *   Saves the best model based on validation Macro-F1.
    *   Logs training, validation, and test metrics to [Weights & Biases](https://wandb.ai) by default (`--no-wandb` to disable).

## Project Structure

```text
.
├── data/                   # Raw input data (Patient folders)
├── extract_zones.py        # Zone extraction logic
├── pre_processing.py       # Data standardization and table building
├── zone_dataset.py         # PyTorch dataset and patient splitting
├── losses.py               # Loss functions (CE, Focal Loss)
├── train_convnext.py       # ConvNeXt-Tiny training script
├── train_clip_convnext.py  # OpenCLIP ConvNeXt-Large training script
├── requirements.txt        # Python dependencies (install inside conda env)
└── processed_image_arrays/ # Output of preprocessing (NPZs, crops, table)
```

## Getting Started

### 1. Environment (conda)

Use the conda env named `venv` — **not** the project-local `.venv`. On the LARA GPU cluster the
`.venv` PyTorch build does not match the cluster CUDA driver and will fall back to CPU.

```bash
conda create -n venv python=3.14 pip -y   # skip if the env already exists
conda activate venv
pip install -r requirements.txt
```

Verify GPU access before training (should print `True` and your GPU name):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

On a shared multi-GPU node, pin a free device and run inside `tmux`/`screen` so jobs survive disconnects:

```bash
nvidia-smi                                    # pick a free GPU index
CUDA_VISIBLE_DEVICES=3 conda activate venv    # example: use GPU 3
```

All commands below assume `conda activate venv` is active.

### 2. Run Preprocessing
Standardize the raw data and generate the training table:
```bash
conda activate venv
python pre_processing.py --data-dir ./data --output-dir ./processed_image_arrays
```

For soft-label training, regenerate the multiclass table (raw 0/1/2 labels):

```bash
python pre_processing.py --multiclass-zone-labels --output-dir processed_image_arrays_multiclass
```

### 3. Train the Model

Authenticate with W&B once (if you use cloud logging): `wandb login`.

Quick smoke test with experiment tracking:

```bash
conda activate venv
python train_convnext.py \
  --csv processed_image_arrays/zone_training_table.csv \
  --data-root processed_image_arrays \
  --epochs 2 \
  --wandb-project uveitis-per-zone \
  --wandb-run-name smoke-test
```

Train a ConvNeXt classifier using Focal Loss:

```bash
conda activate venv
python train_convnext.py \
  --csv processed_image_arrays/zone_training_table.csv \
  --data-root processed_image_arrays \
  --loss focal \
  --epochs 50 \
  --output-dir runs/convnext_focal_v2 \
  --wandb-project uveitis-per-zone \
  --wandb-run-name convnext-focal-v2
```

Pick a fresh `--output-dir` for each run so previous `best.pt` checkpoints aren't overwritten before you've confirmed the new run is healthy. Defaults assume an A6000-class GPU: `--batch-size 32`, `--num-workers 8`, cosine LR with backbone at `lr * 0.1`, and early stopping with patience 8. If DataLoader workers crash (`double free or corruption`), use `--num-workers 0`.

To compare with standard Weighted CrossEntropy:

```bash
conda activate venv
python train_convnext.py \
  --csv processed_image_arrays/zone_training_table.csv \
  --data-root processed_image_arrays \
  --loss ce \
  --epochs 50 \
  --output-dir runs/convnext_ce \
  --wandb-project uveitis-per-zone \
  --wandb-run-name convnext-ce
```

OpenCLIP ConvNeXt-Large with soft tier-1 labels (`train_clip_convnext.py`):

```bash
conda activate venv
CUDA_VISIBLE_DEVICES=3 python train_clip_convnext.py \
  --csv processed_image_arrays_multiclass/zone_training_table.csv \
  --data-root processed_image_arrays_multiclass \
  --loss soft_ce \
  --soft-labels \
  --class-weighting inverse \
  --zone-embed-dim 64 \
  --num-workers 0 \
  --output-dir runs/clip_convnext_softlabel_v5 \
  --wandb-run-name clip-convnext-softlabel-v5
```

Disable W&B entirely (no `wandb` calls):

```bash
python train_convnext.py --no-wandb ...
```

**W&B flags:** `--wandb-project` (default: `uveitis-per-zone`), `--wandb-run-name` (optional), `--no-wandb`.

## Metrics and Evaluation

Training results, including a `metrics.json` file and the `best.pt` checkpoint, are saved in the specified `--output-dir`. The pipeline focuses on **Macro-F1** and **Balanced Accuracy** to ensure fair evaluation across imbalanced zone labels.

When W&B is enabled, each epoch logs train/val loss and accuracy, balanced accuracy, macro-F1, per-class F1/recall/specificity, and train–val loss gap. After training, test-set metrics and per-class ROC curves (with AUC) are logged to the same run.
