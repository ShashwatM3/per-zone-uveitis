#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import wandb
from sklearn.metrics import roc_curve, auc
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from losses import build_loss
from zone_dataset import (
    ZoneImageDataset,
    labels_for,
    load_or_create_split,
    load_zone_records,
    records_for_patients,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a ConvNeXt binary zone-label classifier from zone_training_table.csv "
            "(Zone_Label 0 vs 1+2 merged to {0,1})."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("processed_image_arrays") / "zone_training_table.csv",
        help="Path to zone_training_table.csv.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("processed_image_arrays"),
        help="Root used to resolve relative Zone_Image paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "convnext_zone_classifier",
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Optional patient split JSON. Existing files are reused; missing files are created.",
    )
    parser.add_argument("--loss", choices=("ce", "focal"), default="ce")
    parser.add_argument(
        "--class-weighting",
        choices=("none", "inverse", "effective"),
        default="inverse",
        help="Class weighting used by CE weights or focal alpha.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Must be 2 for the binary zone task (default: 2).",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Classifier-head learning rate. Backbone is trained at lr * 0.1.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split-attempts", type=int, default=10_000)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help=(
            "Max global gradient norm passed to clip_grad_norm_. Set to 0 or "
            "negative to disable clipping (default: 1.0)."
        ),
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="uveitis-per-zone")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, scale=(0.7, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device)
    labels = batch["label"].to(device, dtype=torch.long)
    return images, labels


def update_confusion(
    confusion: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    for target, prediction in zip(targets.view(-1), predictions.view(-1)):
        confusion[target.long(), prediction.long()] += 1


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    confusion = confusion.cpu()
    correct = torch.diag(confusion).sum().item()
    total = confusion.sum().item()
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    specificity_values: list[float] = []

    for class_idx in range(confusion.shape[0]):
        tp = confusion[class_idx, class_idx].item()
        fp = confusion[:, class_idx].sum().item() - tp
        fn = confusion[class_idx, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        tn = confusion.sum().item() - (
            confusion[class_idx, :].sum().item()
            + confusion[:, class_idx].sum().item()
            - tp
        )
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        per_class[str(class_idx)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
        }
        f1_values.append(f1)
        recall_values.append(recall)
        specificity_values.append(specificity)

    return {
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": sum(recall_values) / len(recall_values),
        "macro_f1": sum(f1_values) / len(f1_values),
        "macro_specificity": sum(specificity_values) / len(specificity_values),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    return_probs: bool = False,
    grad_clip_norm: float | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], list]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    collected_probs: list = []

    grad_context = torch.enable_grad() if is_train else torch.inference_mode()
    with grad_context:
        for batch in loader:
            images, labels = batch_to_device(batch, device)
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip_norm
                    )
                optimizer.step()

            if return_probs:
                probs = torch.softmax(logits, dim=1).cpu().tolist()
                for true_label, prob_vec in zip(labels.cpu().tolist(), probs):
                    collected_probs.append((true_label, prob_vec))

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            update_confusion(confusion, logits.argmax(dim=1).cpu(), labels.cpu())

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / total_examples if total_examples else 0.0
    if return_probs:
        return metrics, collected_probs
    return metrics


def summarize_records(records: list) -> dict[str, Any]:
    return {
        "rows": len(records),
        "patients": sorted({record.patient_id for record in records}),
        "labels": dict(sorted(Counter(labels_for(records)).items())),
    }


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> int:
    args = parse_args()
    if args.num_classes != 2:
        raise SystemExit("This branch only supports binary classification; use --num-classes 2.")
    seed_everything(args.seed)
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=serializable_args(args),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_zone_records(args.csv, args.data_root)
    split = load_or_create_split(
        records,
        split_json=args.split_json,
        seed=args.seed,
        attempts=args.split_attempts,
    )
    train_records = records_for_patients(records, split["train"])
    val_records = records_for_patients(records, split["val"])
    test_records = records_for_patients(records, split["test"])

    train_transform, eval_transform = build_transforms(args.image_size)
    train_loader = DataLoader(
        ZoneImageDataset(train_records, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
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

    model = build_model(args.num_classes, pretrained=not args.no_pretrained).to(device)
    criterion = build_loss(
        args.loss,
        labels_for(train_records),
        args.num_classes,
        device,
        class_weighting=args.class_weighting,
        focal_gamma=args.focal_gamma,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": args.lr * 0.1, "name": "backbone"},
            {"params": model.classifier.parameters(), "lr": args.lr, "name": "classifier"},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    best_path = args.output_dir / "best.pt"
    history: list[dict[str, Any]] = []

    print(f"Device: {device}")
    print(f"Loss: {args.loss} class_weighting={args.class_weighting}")
    print(
        f"Scheduler: cosine | "
        f"Head LR: {args.lr} | Backbone LR: {args.lr * 0.1} | "
        f"grad_clip_norm: {args.grad_clip_norm}"
    )
    print(f"Split: {json.dumps(split)}")

    patience = 8
    no_improve_count = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.num_classes,
            optimizer,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, args.num_classes)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)

        group_lrs = {
            group.get("name", f"group_{idx}"): group["lr"]
            for idx, group in enumerate(optimizer.param_groups)
        }
        lr_summary = " ".join(f"lr_{name}={lr:.2e}" for name, lr in group_lrs.items())
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"{lr_summary}"
        )
        print(f"epoch={epoch:03d} val/per_class={val_metrics['per_class']}")
        if use_wandb:
            log_payload = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "train/balanced_accuracy": train_metrics["balanced_accuracy"],
                "train/macro_f1": train_metrics["macro_f1"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "val/balanced_accuracy": val_metrics["balanced_accuracy"],
                "val/macro_f1": val_metrics["macro_f1"],
                "val/f1_class_0": val_metrics["per_class"]["0"]["f1"],
                "val/f1_class_1": val_metrics["per_class"]["1"]["f1"],
                "val/recall_class_0": val_metrics["per_class"]["0"]["recall"],
                "val/recall_class_1": val_metrics["per_class"]["1"]["recall"],
                "val/specificity_class_0": val_metrics["per_class"]["0"]["specificity"],
                "val/specificity_class_1": val_metrics["per_class"]["1"]["specificity"],
                "diagnostics/train_val_loss_gap": val_metrics["loss"] - train_metrics["loss"],
            }
            for name, lr in group_lrs.items():
                log_payload[f"optim/lr_{name}"] = lr
            wandb.log(log_payload)

        scheduler.step()

        if val_metrics["macro_f1"] > best_macro_f1:
            no_improve_count = 0
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": serializable_args(args),
                    "split": split,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, test_probs = run_epoch(
        model, test_loader, criterion, device, args.num_classes, return_probs=True
    )

    metrics = {
        "args": serializable_args(args),
        "split": split,
        "data_summary": {
            "train": summarize_records(train_records),
            "val": summarize_records(val_records),
            "test": summarize_records(test_records),
        },
        "best_epoch": checkpoint["epoch"],
        "best_val": checkpoint["val_metrics"],
        "test": test_metrics,
        "history": history,
    }
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if use_wandb:
        wandb.log({
            "test/loss": test_metrics["loss"],
            "test/accuracy": test_metrics["accuracy"],
            "test/balanced_accuracy": test_metrics["balanced_accuracy"],
            "test/macro_f1": test_metrics["macro_f1"],
            "test/f1_class_0": test_metrics["per_class"]["0"]["f1"],
            "test/f1_class_1": test_metrics["per_class"]["1"]["f1"],
            "test/recall_class_0": test_metrics["per_class"]["0"]["recall"],
            "test/recall_class_1": test_metrics["per_class"]["1"]["recall"],
            "test/specificity_class_0": test_metrics["per_class"]["0"]["specificity"],
            "test/specificity_class_1": test_metrics["per_class"]["1"]["specificity"],
        })

        true_labels = [y for y, _ in test_probs]
        for class_idx in range(args.num_classes):
            binary_labels = [1 if y == class_idx else 0 for y in true_labels]
            scores = [p[class_idx] for _, p in test_probs]
            fpr, tpr, _ = roc_curve(binary_labels, scores)
            roc_auc = auc(fpr, tpr)
            wandb.log({f"test/roc_auc_class_{class_idx}": roc_auc})
            roc_table = wandb.Table(
                data=list(zip(fpr.tolist(), tpr.tolist())),
                columns=["FPR", "TPR"],
            )
            wandb.log({
                f"test/roc_curve_class_{class_idx}": wandb.plot.line(
                    roc_table,
                    "FPR",
                    "TPR",
                    title=f"ROC Curve — Class {class_idx} (AUC={roc_auc:.3f})",
                )
            })

        wandb.finish()

    print(f"Best checkpoint: {best_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Test macro_f1={test_metrics['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
