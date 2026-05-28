#!/usr/bin/env python3
"""Train and evaluate a patient-level classifier on pooled zone features.

Example:
    python train_patient_aggregator.py \\
      --features runs/patient_aggregation/features/features.npz \\
      --output-dir runs/patient_aggregation/results \\
      --aggregation meanmax \\
      --classifier svm
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from losses import FocalLoss, class_weights

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate zone features to patients and train a patient-level classifier."
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to features.npz from extract_features.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--aggregation",
        choices=("mean", "max", "meanmax"),
        default="meanmax",
    )
    parser.add_argument(
        "--classifier",
        choices=("svm", "mlp"),
        default="svm",
    )
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patient_label_from_zones(zone_labels: np.ndarray) -> int:
    n_pos = int(np.sum(zone_labels == 1))
    n_total = len(zone_labels)
    if n_pos * 2 > n_total:
        return 1
    if n_pos * 2 < n_total:
        return 0
    return 1


def aggregate_zone_features(zone_features: np.ndarray, aggregation: str) -> np.ndarray:
    if aggregation == "mean":
        return zone_features.mean(axis=0)
    if aggregation == "max":
        return zone_features.max(axis=0)
    if aggregation == "meanmax":
        return np.concatenate(
            [zone_features.mean(axis=0), zone_features.max(axis=0)],
            axis=0,
        )
    raise ValueError(f"Unknown aggregation: {aggregation}")


def input_dim_for(aggregation: str, feature_dim: int) -> int:
    if aggregation == "meanmax":
        return feature_dim * 2
    return feature_dim


def build_patient_split_data(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    splits: np.ndarray,
    split_name: str,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    mask = splits == split_name
    pid_arr = patient_ids[mask]
    lbl_arr = labels[mask]
    feat_arr = features[mask]

    zones_by_patient: dict[int, list[np.ndarray]] = defaultdict(list)
    labels_by_patient: dict[int, list[int]] = defaultdict(list)
    for pid, label, feat in zip(pid_arr, lbl_arr, feat_arr):
        zones_by_patient[int(pid)].append(feat)
        labels_by_patient[int(pid)].append(int(label))

    patient_list = sorted(zones_by_patient)
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for pid in patient_list:
        zone_stack = np.stack(zones_by_patient[pid], axis=0)
        x_rows.append(aggregate_zone_features(zone_stack, aggregation))
        y_rows.append(patient_label_from_zones(np.array(labels_by_patient[pid], dtype=np.int64)))

    return np.stack(x_rows, axis=0), np.array(y_rows, dtype=np.int64), patient_list


def metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    num_classes = 2
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        confusion[int(true), int(pred)] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    for class_idx in range(num_classes):
        tp = confusion[class_idx, class_idx]
        fp = confusion[:, class_idx].sum() - tp
        fn = confusion[class_idx, :].sum() - tp
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        per_class[str(class_idx)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
        recall_values.append(recall)

    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    return {
        "macro_f1": float(sum(f1_values) / len(f1_values)),
        "balanced_accuracy": float(sum(recall_values) / len(recall_values)),
        "accuracy": float(correct / total) if total else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def macro_f1_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    y_pred = (probs >= threshold).astype(np.int64)
    return metrics_from_predictions(y_true, y_pred)["macro_f1"]


def best_threshold_on_val(
    y_val: np.ndarray,
    val_probs: np.ndarray,
) -> tuple[float, float]:
    thresholds = np.arange(0.3, 0.701, 0.05)
    best_threshold = 0.5
    best_macro_f1 = -1.0
    for threshold in thresholds:
        macro_f1 = macro_f1_from_probs(y_val, val_probs, float(threshold))
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(threshold)
    return best_threshold, best_macro_f1


class PatientMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> PatientMLP:
    model = PatientMLP(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_x = torch.tensor(x_train, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_train, dtype=torch.long, device=device)
    val_x = torch.tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.long, device=device)

    alpha = class_weights(y_train.tolist(), num_classes=2, mode="inverse")
    criterion = FocalLoss(gamma=2.0, alpha=alpha.to(device) if alpha is not None else None)

    best_macro_f1 = -1.0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            val_logits = model(val_x)
            val_probs = F.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_macro_f1 = macro_f1_from_probs(y_val, val_probs, 0.5)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs:
            logger.info(
                "MLP epoch=%d/%d train_loss=%.4f val_macro_f1@0.5=%.4f",
                epoch,
                epochs,
                loss.item(),
                val_macro_f1,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def mlp_positive_probs(model: PatientMLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        tensor_x = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(tensor_x)
        return F.softmax(logits, dim=1)[:, 1].cpu().numpy()


def train_svm(x_train: np.ndarray, y_train: np.ndarray) -> Any:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    class_weight="balanced",
                    probability=True,
                    random_state=13,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def svm_positive_probs(model: Any, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def per_patient_rows(
    patient_ids: list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid, true_label, pred_label, prob in zip(patient_ids, y_true, y_pred, probs):
        rows.append(
            {
                "patient_id": int(pid),
                "true_label": int(true_label),
                "predicted_label": int(pred_label),
                "probability": float(prob),
            }
        )
    return rows


def save_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["patient_id", "true_label", "predicted_label", "probability"],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_metrics_block(title: str, metrics: dict[str, Any], threshold: float) -> None:
    print(f"\n=== {title} (threshold={threshold:.2f}) ===")
    print(f"macro_f1:          {metrics['macro_f1']:.4f}")
    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"accuracy:          {metrics['accuracy']:.4f}")
    for class_idx, class_metrics in metrics["per_class"].items():
        print(
            f"class {class_idx}: "
            f"precision={class_metrics['precision']:.4f} "
            f"recall={class_metrics['recall']:.4f} "
            f"f1={class_metrics['f1']:.4f}"
        )
    print("confusion_matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    data = np.load(args.features, allow_pickle=False)
    patient_ids = data["patient_ids"]
    labels = data["labels"]
    features = data["features"]
    splits = data["splits"]
    feature_dim = features.shape[1]
    input_dim = input_dim_for(args.aggregation, feature_dim)

    x_train, y_train, train_pids = build_patient_split_data(
        patient_ids, labels, features, splits, "train", args.aggregation
    )
    x_val, y_val, val_pids = build_patient_split_data(
        patient_ids, labels, features, splits, "val", args.aggregation
    )
    x_test, y_test, test_pids = build_patient_split_data(
        patient_ids, labels, features, splits, "test", args.aggregation
    )

    logger.info(
        "Patients: train=%d val=%d test=%d | feature_dim=%d input_dim=%d",
        len(train_pids),
        len(val_pids),
        len(test_pids),
        feature_dim,
        input_dim,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.classifier == "svm":
        model = train_svm(x_train, y_train)
        val_probs = svm_positive_probs(model, x_val)
        test_probs = svm_positive_probs(model, x_test)
    else:
        mlp = train_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            input_dim=input_dim,
            hidden_dim=args.mlp_hidden,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            device=device,
        )
        val_probs = mlp_positive_probs(mlp, x_val, device)
        test_probs = mlp_positive_probs(mlp, x_test, device)

    best_threshold, val_macro_f1_at_best = best_threshold_on_val(y_val, val_probs)
    val_metrics_default = metrics_from_predictions(
        y_val, (val_probs >= 0.5).astype(np.int64)
    )
    val_metrics_tuned = metrics_from_predictions(
        y_val, (val_probs >= best_threshold).astype(np.int64)
    )
    test_metrics_tuned = metrics_from_predictions(
        y_test, (test_probs >= best_threshold).astype(np.int64)
    )

    test_pred_tuned = (test_probs >= best_threshold).astype(np.int64)
    prediction_rows = per_patient_rows(test_pids, y_test, test_pred_tuned, test_probs)

    print_metrics_block("Validation (default threshold 0.50)", val_metrics_default, 0.5)
    print(
        f"\nBest val threshold by macro-F1: {best_threshold:.2f} "
        f"(val macro_f1={val_macro_f1_at_best:.4f})"
    )
    print_metrics_block("Validation (tuned threshold)", val_metrics_tuned, best_threshold)
    print_metrics_block("Test (tuned threshold)", test_metrics_tuned, best_threshold)

    metrics_payload: dict[str, Any] = {
        "args": {
            "features": str(args.features.resolve()),
            "aggregation": args.aggregation,
            "classifier": args.classifier,
            "mlp_hidden": args.mlp_hidden,
            "mlp_epochs": args.mlp_epochs,
            "mlp_lr": args.mlp_lr,
            "seed": args.seed,
            "input_dim": input_dim,
        },
        "patient_counts": {
            "train": len(train_pids),
            "val": len(val_pids),
            "test": len(test_pids),
        },
        "train_label_distribution": {
            str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))
        },
        "best_threshold": best_threshold,
        "val_macro_f1_at_best_threshold": val_macro_f1_at_best,
        "validation_default_threshold_0_5": val_metrics_default,
        "validation_tuned_threshold": val_metrics_tuned,
        "test_tuned_threshold": test_metrics_tuned,
        "per_patient_predictions": prediction_rows,
    }

    metrics_path = args.output_dir / "patient_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    csv_path = args.output_dir / "patient_predictions.csv"
    save_predictions_csv(csv_path, prediction_rows)

    logger.info("Saved %s", metrics_path)
    logger.info("Saved %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
