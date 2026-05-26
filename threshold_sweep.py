#!/usr/bin/env python3
"""Sweep class-1 decision threshold on val set for a saved CLIP ConvNeXt checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_clip_convnext import (
    batch_to_device,
    build_model_and_eval_transform,
    metrics_from_confusion,
)
from zone_dataset import (
    ZoneImageDataset,
    load_or_create_split,
    load_zone_records,
    records_for_patients,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold sweep on val set.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs") / "clip_convnext_ce_zone64_v3" / "best.pt",
        help="Path to best.pt checkpoint.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("processed_image_arrays") / "zone_training_table.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("processed_image_arrays"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split-attempts", type=int, default=10_000)
    parser.add_argument("--threshold-min", type=float, default=0.25)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    return parser.parse_args()


@torch.inference_mode()
def collect_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    labels_list: list[int] = []
    prob1_list: list[float] = []

    model.eval()
    for batch in loader:
        images, labels, zones = batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, zones)
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        labels_list.extend(labels.cpu().tolist())
        prob1_list.extend(probs.cpu().tolist())

    return np.asarray(labels_list, dtype=np.int64), np.asarray(prob1_list, dtype=np.float64)


def metrics_at_threshold(
    labels: np.ndarray,
    prob1: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (prob1 >= threshold).astype(np.int64)
    confusion = torch.zeros(2, 2, dtype=torch.long)
    for target, prediction in zip(labels, predictions):
        confusion[int(target), int(prediction)] += 1
    metrics = metrics_from_confusion(confusion)
    pc = metrics["per_class"]
    return {
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "f1_class_0": pc["0"]["f1"],
        "f1_class_1": pc["1"]["f1"],
        "recall_0": pc["0"]["recall"],
        "recall_1": pc["1"]["recall"],
        "precision_0": pc["0"]["precision"],
        "precision_1": pc["1"]["precision"],
    }


def print_table(rows: list[dict[str, float]]) -> None:
    headers = [
        "threshold",
        "accuracy",
        "macro_f1",
        "f1_0",
        "f1_1",
        "recall_0",
        "recall_1",
        "prec_0",
        "prec_1",
    ]
    print(
        f"{'threshold':>9}  {'accuracy':>8}  {'macro_f1':>8}  {'f1_0':>6}  {'f1_1':>6}  "
        f"{'recall_0':>8}  {'recall_1':>8}  {'prec_0':>6}  {'prec_1':>6}"
    )
    print("-" * 88)
    for row in rows:
        print(
            f"{row['threshold']:9.3f}  "
            f"{row['accuracy']:8.4f}  "
            f"{row['macro_f1']:8.4f}  "
            f"{row['f1_class_0']:6.4f}  "
            f"{row['f1_class_1']:6.4f}  "
            f"{row['recall_0']:8.4f}  "
            f"{row['recall_1']:8.4f}  "
            f"{row['precision_0']:6.4f}  "
            f"{row['precision_1']:6.4f}"
        )


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    num_classes = int(ckpt_args.get("num_classes", 2))
    zone_embed_dim = int(ckpt_args.get("zone_embed_dim", 64))
    head_hidden = int(ckpt_args.get("head_hidden", 256))
    seed = int(ckpt_args.get("seed", args.seed))

    records = load_zone_records(args.csv, args.data_root)
    split = load_or_create_split(
        records,
        split_json=None,
        seed=seed,
        attempts=args.split_attempts,
    )
    val_records = records_for_patients(records, split["val"])

    model, eval_transform = build_model_and_eval_transform(
        num_classes,
        zone_embed_dim=zone_embed_dim,
        head_hidden=head_hidden,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    val_loader = DataLoader(
        ZoneImageDataset(val_records, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    labels, prob1 = collect_probs(model, val_loader, device, use_amp)
    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2,
        args.threshold_step,
    )

    rows = [metrics_at_threshold(labels, prob1, float(t)) for t in thresholds]
    best_macro = max(rows, key=lambda r: r["macro_f1"])
    best_acc = max(rows, key=lambda r: r["accuracy"])

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Best epoch in checkpoint: {checkpoint.get('epoch', 'unknown')}")
    print(f"Val samples: {len(labels)}")
    print()
    print_table(rows)
    print()
    print(
        f"Best macro_f1: {best_macro['macro_f1']:.4f} at threshold={best_macro['threshold']:.3f} "
        f"(accuracy={best_macro['accuracy']:.4f}, f1_1={best_macro['f1_class_1']:.4f})"
    )
    print(
        f"Best accuracy: {best_acc['accuracy']:.4f} at threshold={best_acc['threshold']:.3f} "
        f"(macro_f1={best_acc['macro_f1']:.4f}, f1_1={best_acc['f1_class_1']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
