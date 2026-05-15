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
    *   Two output logits (classes 0 and 1). Tracks Accuracy, Balanced Accuracy, Macro-F1, and a 2×2 confusion matrix.
    *   Saves the best model based on validation Macro-F1.

## Project Structure

```text
.
├── data/                   # Raw input data (Patient folders)
├── extract_zones.py        # Zone extraction logic
├── pre_processing.py       # Data standardization and table building
├── zone_dataset.py         # PyTorch dataset and patient splitting
├── losses.py               # Loss functions (CE, Focal Loss)
├── train_convnext.py       # Model training script
├── requirements.txt        # Python dependencies
└── processed_image_arrays/ # Output of preprocessing (NPZs, crops, table)
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing
Standardize the raw data and generate the training table:
```bash
python pre_processing.py --data-dir ./data --output-dir ./processed_image_arrays
```

### 3. Train the Model
Train a ConvNeXt classifier using Focal Loss:
```bash
python train_convnext.py \
  --csv processed_image_arrays/zone_training_table.csv \
  --data-root processed_image_arrays \
  --loss focal \
  --epochs 50 \
  --output-dir runs/convnext_focal
```

To compare with standard Weighted CrossEntropy:
```bash
python train_convnext.py \
  --csv processed_image_arrays/zone_training_table.csv \
  --data-root processed_image_arrays \
  --loss ce \
  --epochs 50 \
  --output-dir runs/convnext_ce
```

## Metrics and Evaluation
Training results, including a `metrics.json` file and the `best.pt` checkpoint, are saved in the specified `--output-dir`. The pipeline focuses on **Macro-F1** and **Balanced Accuracy** to ensure fair evaluation across imbalanced zone labels.
