#!/usr/bin/env python3
"""Extract penultimate ConvNeXt features per zone for patient-level aggregation.

Example:
    python extract_features.py \\
      --checkpoint runs/protocol_r14_convnext_exclude_tier1/best.pt \\
      --csv processed_image_arrays_multiclass/zone_training_table.csv \\
      --data-root processed_image_arrays_multiclass \\
      --split-json splits/canonical_split.json \\
      --output-dir runs/patient_aggregation/features \\
      --batch-size 64 --num-workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from zone_dataset import (
    ZoneImageDataset,
    load_or_create_split,
    load_zone_records,
    records_for_patients,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 768-dim ConvNeXt zone features grouped by patient split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt.")
    parser.add_argument("--csv", type=Path, required=True, help="zone_training_table.csv path.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root for zone images.")
    parser.add_argument(
        "--split-json",
        type=Path,
        required=True,
        help="Patient split JSON (train/val/test patient ids).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for features.npz and features_meta.json.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=288)
    return parser.parse_args()


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_feature_model(num_classes: int) -> nn.Module:
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def class_distribution(labels: list[int]) -> dict[str, int]:
    counts = Counter(labels)
    return {str(label): counts[label] for label in sorted(counts)}


def extract_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> tuple[list[int], list[int], list[int], list[np.ndarray]]:
    patient_ids: list[int] = []
    zone_numbers: list[int] = []
    labels: list[int] = []
    features: list[np.ndarray] = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                feats = model(images)

            feats_np = feats.float().cpu().numpy()
            batch_size = feats_np.shape[0]
            patient_ids.extend(int(pid) for pid in batch["patient_id"])
            zone_numbers.extend(int(z) for z in batch["zone_number"])
            labels.extend(int(lbl) for lbl in batch["label"])
            for row in feats_np:
                features.append(row.astype(np.float32, copy=False))

            if batch_idx % 100 == 0:
                logger.info(
                    "split=%s batch=%d zones_processed=%d",
                    split_name,
                    batch_idx,
                    len(patient_ids),
                )

    return patient_ids, zone_numbers, labels, features


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    num_classes = int(ckpt_args.get("num_classes", 2))

    model = build_feature_model(num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.classifier[-1] = nn.Identity()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    records = load_zone_records(args.csv, args.data_root)
    split = load_or_create_split(
        records,
        split_json=args.split_json,
        seed=int(ckpt_args.get("seed", 13)),
        attempts=int(ckpt_args.get("split_attempts", 10_000)),
    )

    eval_transform = build_eval_transform(args.image_size)
    split_names = ("train", "val", "test")
    all_patient_ids: list[int] = []
    all_zone_numbers: list[int] = []
    all_labels: list[int] = []
    all_features: list[np.ndarray] = []
    all_splits: list[str] = []
    counts_per_split: dict[str, int] = {}
    class_dist_per_split: dict[str, dict[str, int]] = {}

    for split_name in split_names:
        split_records = records_for_patients(records, split[split_name])
        loader = DataLoader(
            ZoneImageDataset(split_records, eval_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        logger.info(
            "Extracting features for split=%s (%d zones, %d patients)",
            split_name,
            len(split_records),
            len(split[split_name]),
        )
        pids, zones, lbls, feats = extract_split(model, loader, device, split_name)
        all_patient_ids.extend(pids)
        all_zone_numbers.extend(zones)
        all_labels.extend(lbls)
        all_features.extend(feats)
        all_splits.extend([split_name] * len(pids))
        counts_per_split[split_name] = len(pids)
        class_dist_per_split[split_name] = class_distribution(lbls)

    feature_matrix = np.stack(all_features, axis=0)
    np.savez(
        args.output_dir / "features.npz",
        patient_ids=np.array(all_patient_ids, dtype=np.int64),
        zone_numbers=np.array(all_zone_numbers, dtype=np.int64),
        labels=np.array(all_labels, dtype=np.int64),
        features=feature_matrix,
        splits=np.array(all_splits, dtype=str),
    )

    meta: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "csv": str(args.csv.resolve()),
        "split_json": str(args.split_json.resolve()),
        "image_size": args.image_size,
        "feature_dim": int(feature_matrix.shape[1]),
        "total_zones": int(feature_matrix.shape[0]),
        "counts_per_split": counts_per_split,
        "class_distribution_per_split": class_dist_per_split,
    }
    meta_path = args.output_dir / "features_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved %s", args.output_dir / "features.npz")
    logger.info("Saved %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
