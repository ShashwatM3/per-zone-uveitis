#!/usr/bin/env python3
"""Threshold + TTA sweeps for ZoneAwareConvNeXt checkpoints (protocol NEXT 1–2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from train_convnext import (
    batch_to_device,
    build_model,
    build_transforms,
    metrics_from_confusion,
)
from zone_dataset import (
    ZoneImageDataset,
    _load_zone_image,
    load_zone_records,
    records_for_patients,
)

TTA_VIEWS: tuple[tuple[bool, bool], ...] = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConvNeXt threshold + TTA evaluation.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to best.pt",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("processed_image_arrays_multiclass") / "zone_training_table.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("processed_image_arrays_multiclass"),
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("splits") / "canonical_split.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write sweep results JSON.",
    )
    return parser.parse_args()


def load_split(split_json: Path, checkpoint: dict) -> dict[str, list[int]]:
    if split_json.exists():
        with split_json.open(encoding="utf-8") as f:
            data = json.load(f)
        return {name: [int(pid) for pid in data[name]] for name in ("train", "val", "test")}
    split = checkpoint["split"]
    return {name: [int(pid) for pid in split[name]] for name in ("train", "val", "test")}


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
    }


@torch.inference_mode()
def collect_probs_dataloader(
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


@torch.inference_mode()
def collect_tta_probs(
    model: torch.nn.Module,
    records: list,
    eval_transform,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_list: list[int] = []
    baseline_prob1: list[float] = []
    tta_prob1: list[float] = []
    model.eval()
    for record in records:
        image = _load_zone_image(record)
        zone_tensor = torch.tensor([record.zone_number], dtype=torch.long, device=device)
        view_probs: list[np.ndarray] = []
        for hflip, vflip in TTA_VIEWS:
            view = image
            if hflip:
                view = TF.hflip(view)
            if vflip:
                view = TF.vflip(view)
            tensor = eval_transform(view).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(tensor, zone_tensor)
            view_probs.append(torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy())
        baseline_prob1.append(float(view_probs[0][1]))
        tta_prob1.append(float(np.mean(view_probs, axis=0)[1]))
        labels_list.append(record.label)
    return (
        np.asarray(labels_list, dtype=np.int64),
        np.asarray(baseline_prob1, dtype=np.float64),
        np.asarray(tta_prob1, dtype=np.float64),
    )


def sweep_thresholds(
    labels: np.ndarray,
    prob1: np.ndarray,
    t_min: float,
    t_max: float,
    t_step: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    thresholds = np.arange(t_min, t_max + t_step / 2, t_step)
    rows = [metrics_at_threshold(labels, prob1, float(t)) for t in thresholds]
    best = max(rows, key=lambda r: r["macro_f1"])
    return rows, best


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    num_classes = int(ckpt_args.get("num_classes", 2))
    zone_embed_dim = int(ckpt_args.get("zone_embed_dim", 64))
    head_hidden = int(ckpt_args.get("head_hidden", 256))
    image_size = int(ckpt_args.get("image_size", 288))

    records = load_zone_records(args.csv, args.data_root)
    split = load_split(args.split_json, checkpoint)
    val_records = records_for_patients(records, split["val"])
    test_records = records_for_patients(records, split["test"])

    _, eval_transform = build_transforms(image_size)
    model = build_model(
        num_classes,
        pretrained=False,
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
    test_loader = DataLoader(
        ZoneImageDataset(test_records, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device} | best_epoch: {checkpoint.get('epoch', '?')}")
    print()

    val_labels, val_prob1 = collect_probs_dataloader(model, val_loader, device, use_amp)
    test_labels, test_prob1 = collect_probs_dataloader(model, test_loader, device, use_amp)

    val_rows, val_best = sweep_thresholds(
        val_labels, val_prob1, args.threshold_min, args.threshold_max, args.threshold_step
    )
    opt_t = val_best["threshold"]
    test_at_default = metrics_at_threshold(test_labels, test_prob1, 0.5)
    test_at_opt = metrics_at_threshold(test_labels, test_prob1, opt_t)

    print("=== Threshold sweep (val) — top 5 by macro-F1 ===")
    for row in sorted(val_rows, key=lambda r: r["macro_f1"], reverse=True)[:5]:
        print(
            f"  t={row['threshold']:.2f} macro_f1={row['macro_f1']:.4f} "
            f"f1_1={row['f1_class_1']:.4f} recall_1={row['recall_1']:.4f}"
        )
    print(f"\nOptimal val threshold: {opt_t:.3f} (macro_f1={val_best['macro_f1']:.4f})")
    print(f"Test @0.50: macro_f1={test_at_default['macro_f1']:.4f} recall_1={test_at_default['recall_1']:.4f}")
    print(
        f"Test @{opt_t:.3f}: macro_f1={test_at_opt['macro_f1']:.4f} "
        f"recall_1={test_at_opt['recall_1']:.4f}"
    )
    print()

    val_labels_tta, val_base, val_tta = collect_tta_probs(
        model, val_records, eval_transform, device, use_amp
    )
    test_labels_tta, test_base, test_tta = collect_tta_probs(
        model, test_records, eval_transform, device, use_amp
    )
    _, val_tta_best = sweep_thresholds(
        val_labels_tta, val_tta, args.threshold_min, args.threshold_max, args.threshold_step
    )
    tta_t = val_tta_best["threshold"]
    test_tta_default = metrics_at_threshold(test_labels_tta, test_tta, 0.5)
    test_tta_opt = metrics_at_threshold(test_labels_tta, test_tta, tta_t)

    print("=== TTA (4-view flip) ===")
    print(
        f"Val TTA best threshold: {tta_t:.3f} (macro_f1={val_tta_best['macro_f1']:.4f})"
    )
    print(
        f"Test TTA @0.50: macro_f1={test_tta_default['macro_f1']:.4f} "
        f"recall_1={test_tta_default['recall_1']:.4f}"
    )
    print(
        f"Test TTA @{tta_t:.3f}: macro_f1={test_tta_opt['macro_f1']:.4f} "
        f"recall_1={test_tta_opt['recall_1']:.4f}"
    )

    results = {
        "checkpoint": str(args.checkpoint),
        "val_threshold_sweep_best": val_best,
        "test_at_default_threshold": test_at_default,
        "test_at_val_optimal_threshold": test_at_opt,
        "val_tta_threshold_sweep_best": val_tta_best,
        "test_tta_at_default": test_tta_default,
        "test_tta_at_val_optimal": test_tta_opt,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
