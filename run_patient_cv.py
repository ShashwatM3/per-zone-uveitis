#!/usr/bin/env python3
"""K-fold patient-level CV for protocol_r14 ConvNeXt config (protocol D6)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

# numpy before sklearn/torch stack (MKL threading).
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from zone_dataset import load_zone_records, zone_label_to_binary


def patient_labels(records) -> tuple[list[int], list[int]]:
    per_patient: dict[int, list[int]] = {}
    for r in records:
        per_patient.setdefault(r.patient_id, []).append(zone_label_to_binary(r.raw_label))
    patient_ids = sorted(per_patient)
    # Positive if any zone is positive (binary 1).
    labels = [1 if any(l == 1 for l in per_patient[pid]) else 0 for pid in patient_ids]
    return patient_ids, labels


def make_fold_split(
    patient_ids: list[int],
    train_val_ids: list[int],
    test_ids: list[int],
    seed: int,
    val_fraction: float = 0.15,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = train_val_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    return {
        "train": sorted(shuffled[n_val:]),
        "val": sorted(shuffled[:n_val]),
        "test": sorted(test_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("processed_image_arrays_multiclass/zone_training_table.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("processed_image_arrays_multiclass"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/protocol_cv3_r14_config"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=20, help="Match protocol_r14 (20).")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_zone_records(args.csv, args.data_root)
    patient_ids, y = patient_labels(records)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics: list[dict] = []

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(patient_ids, y)):
        fold_name = f"fold{fold}"
        fold_dir = args.output_dir / fold_name
        metrics_path = fold_dir / "metrics.json"
        if metrics_path.exists():
            fold_metrics.append(json.loads(metrics_path.read_text()))
            print(f"SKIP {fold_name} (exists)")
            continue

        train_val_pids = [patient_ids[i] for i in train_val_idx]
        test_pids = [patient_ids[i] for i in test_idx]
        split = make_fold_split(patient_ids, train_val_pids, test_pids, seed=args.seed + fold)
        split_path = fold_dir / "split.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split, indent=2))

        train_log = fold_dir / "train.log"
        cmd = [
            str(args.python),
            "-u",
            str(Path(__file__).resolve().parent / "train_convnext.py"),
            "--csv", str(args.csv),
            "--data-root", str(args.data_root),
            "--loss", "focal",
            "--focal-gamma", "3.0",
            "--class-weighting", "inverse",
            "--exclude-tier1",
            "--epochs", str(args.epochs),
            "--batch-size", "32",
            "--num-workers", "8",
            "--split-json", str(split_path),
            "--output-dir", str(fold_dir),
            "--wandb-project", "uveitis-per-zone",
            "--wandb-run-name", f"cv{args.folds}_r14_{fold_name}",
            "--no-wandb",
        ]
        print("RUN", " ".join(cmd))
        if args.dry_run:
            continue
        env = os.environ.copy()
        env.setdefault("MKL_THREADING_LAYER", "GNU")
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env["PYTHONUNBUFFERED"] = "1"
        print("RUN", " ".join(cmd), flush=True)
        with train_log.open("w", encoding="utf-8") as logf:
            subprocess.run(cmd, check=True, env=env, stdout=logf, stderr=subprocess.STDOUT)
        fold_metrics.append(json.loads(metrics_path.read_text()))

    if not fold_metrics:
        return 0

    test_f1 = [m["test"]["macro_f1"] for m in fold_metrics]
    test_rec = [m["test"]["per_class"]["1"]["recall"] for m in fold_metrics]
    summary = {
        "reference_run": "protocol_r14_convnext_exclude_tier1",
        "reference_canonical_test_f1": 0.5599,
        "config": "ConvNeXt-Tiny, exclude-tier1, focal gamma=3, inverse weights, zone_embed=64, 20 epochs",
        "epochs": args.epochs,
        "n_folds": len(fold_metrics),
        "test_macro_f1_mean": float(np.mean(test_f1)),
        "test_macro_f1_std": float(np.std(test_f1)),
        "test_class1_recall_mean": float(np.mean(test_rec)),
        "test_class1_recall_std": float(np.std(test_rec)),
        "per_fold_test_f1": test_f1,
        "per_fold_test_class1_recall": test_rec,
    }
    summary_path = args.output_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
