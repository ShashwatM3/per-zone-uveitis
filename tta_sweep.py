#!/usr/bin/env python3
"""Compare baseline vs 4-view flip TTA on the val set for a saved CLIP ConvNeXt checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from train_clip_convnext import (
    build_model_and_eval_transform,
    metrics_from_confusion,
)
from zone_dataset import (
    _load_zone_image,
    load_zone_records,
    records_for_patients,
)

RUN_DIR = Path("runs") / "clip_convnext_ce_zone64_v3"
CHECKPOINT_PATH = RUN_DIR / "best.pt"
SPLIT_JSON = RUN_DIR / "split.json"
CSV_PATH = Path("processed_image_arrays") / "zone_training_table.csv"
DATA_ROOT = Path("processed_image_arrays")
THRESHOLD_DEFAULT = 0.5
THRESHOLD_MIN = 0.35
THRESHOLD_MAX = 0.65
THRESHOLD_STEP = 0.025

TTA_VIEWS: tuple[tuple[bool, bool], ...] = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)


def load_split() -> dict[str, list[int]]:
    if SPLIT_JSON.exists():
        with SPLIT_JSON.open(encoding="utf-8") as f:
            data = json.load(f)
        return {name: [int(pid) for pid in data[name]] for name in ("train", "val", "test")}

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if "split" not in checkpoint:
        raise FileNotFoundError(
            f"{SPLIT_JSON} not found and checkpoint has no embedded split."
        )
    print(f"Note: {SPLIT_JSON} not found; using split embedded in {CHECKPOINT_PATH}.")
    split = checkpoint["split"]
    return {name: [int(pid) for pid in split[name]] for name in ("train", "val", "test")}


def apply_flip(image: Image.Image, hflip: bool, vflip: bool) -> Image.Image:
    if hflip:
        image = TF.hflip(image)
    if vflip:
        image = TF.vflip(image)
    return image


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
    per_class = metrics["per_class"]
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "f1_class_0": per_class["0"]["f1"],
        "f1_class_1": per_class["1"]["f1"],
        "recall_0": per_class["0"]["recall"],
        "recall_1": per_class["1"]["recall"],
    }


def print_metrics_block(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print(f"  accuracy:   {metrics['accuracy']:.4f}")
    print(f"  macro_f1:   {metrics['macro_f1']:.4f}")
    print(f"  f1_class_0: {metrics['f1_class_0']:.4f}")
    print(f"  f1_class_1: {metrics['f1_class_1']:.4f}")
    print(f"  recall_0:   {metrics['recall_0']:.4f}")
    print(f"  recall_1:   {metrics['recall_1']:.4f}")


def print_delta(baseline: dict[str, float], tta: dict[str, float]) -> None:
    print("Delta (TTA - baseline) at threshold 0.5:")
    for key in (
        "accuracy",
        "macro_f1",
        "f1_class_0",
        "f1_class_1",
        "recall_0",
        "recall_1",
    ):
        delta = tta[key] - baseline[key]
        sign = "+" if delta >= 0 else ""
        print(f"  {key}: {sign}{delta:.4f}")


@torch.inference_mode()
def collect_baseline_and_tta_probs(
    model: torch.nn.Module,
    records: list,
    eval_transform: transforms.Compose,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_list: list[int] = []
    baseline_prob1_list: list[float] = []
    tta_prob1_list: list[float] = []

    model.eval()
    for record in records:
        image = _load_zone_image(record)
        label = record.label
        zone_number = record.zone_number
        zone_tensor = torch.tensor([zone_number], dtype=torch.long, device=device)

        view_probs: list[np.ndarray] = []
        for hflip, vflip in TTA_VIEWS:
            view_image = apply_flip(image, hflip=hflip, vflip=vflip)
            tensor = eval_transform(view_image).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(tensor, zone_tensor)
            probs = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
            view_probs.append(probs)

        baseline_probs = view_probs[0]
        tta_probs = np.mean(view_probs, axis=0)

        labels_list.append(label)
        baseline_prob1_list.append(float(baseline_probs[1]))
        tta_prob1_list.append(float(tta_probs[1]))

    return (
        np.asarray(labels_list, dtype=np.int64),
        np.asarray(baseline_prob1_list, dtype=np.float64),
        np.asarray(tta_prob1_list, dtype=np.float64),
    )


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    split = load_split()
    records = load_zone_records(CSV_PATH, DATA_ROOT)
    val_records = records_for_patients(records, split["val"])

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    num_classes = int(ckpt_args.get("num_classes", 2))
    zone_embed_dim = int(ckpt_args.get("zone_embed_dim", 64))
    head_hidden = int(ckpt_args.get("head_hidden", 256))

    model, eval_transform = build_model_and_eval_transform(
        num_classes,
        zone_embed_dim=zone_embed_dim,
        head_hidden=head_hidden,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Best epoch in checkpoint: {checkpoint.get('epoch', 'unknown')}")
    print(f"Val patients: {len(split['val'])}")
    print(f"Val samples: {len(val_records)}")
    print(f"Device: {device}")
    print()

    labels, baseline_prob1, tta_prob1 = collect_baseline_and_tta_probs(
        model,
        val_records,
        eval_transform,
        device,
        use_amp,
    )

    baseline_metrics = metrics_at_threshold(labels, baseline_prob1, THRESHOLD_DEFAULT)
    tta_metrics = metrics_at_threshold(labels, tta_prob1, THRESHOLD_DEFAULT)

    print_metrics_block("Baseline (no TTA) at threshold 0.5:", baseline_metrics)
    print()
    print_metrics_block("TTA (4-view) at threshold 0.5:", tta_metrics)
    print()
    print_delta(baseline_metrics, tta_metrics)
    print()

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX + THRESHOLD_STEP / 2,
        THRESHOLD_STEP,
    )
    sweep_rows = [
        (float(threshold), metrics_at_threshold(labels, tta_prob1, float(threshold)))
        for threshold in thresholds
    ]
    best_threshold, best = max(sweep_rows, key=lambda item: item[1]["macro_f1"])
    print(
        f"TTA threshold sweep ({THRESHOLD_MIN:.2f} to {THRESHOLD_MAX:.2f}, "
        f"step {THRESHOLD_STEP:.3f}):"
    )
    print(f"  best macro_f1: {best['macro_f1']:.4f} at threshold={best_threshold:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
