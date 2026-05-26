#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import open_clip
import torch
import wandb
from sklearn.metrics import auc, roc_curve
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from losses import FocalLoss, build_loss
from zone_dataset import (
    ZoneImageDataset,
    labels_for,
    load_or_create_split,
    load_zone_records,
    records_for_patients,
)


CLIP_MODEL_NAME = "convnext_large_d_320"
CLIP_PRETRAINED = "laion2b_s29b_b131k_ft_soup"
CLIP_OUTPUT_DIM = 768
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

NUM_ZONES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an OpenCLIP ConvNeXt-Large-D binary zone-label classifier from "
            "zone_training_table.csv (Zone_Label 0 vs 1+2 merged to {0,1})."
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
        default=Path("runs") / "clip_convnext_zone_classifier",
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Optional patient split JSON. Existing files are reused; missing files are created.",
    )
    parser.add_argument("--loss", choices=("ce", "soft_ce", "focal"), default="ce")
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
        help="Classifier-head learning rate. Backbone uses layer-wise LRs after unfreeze.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split-attempts", type=int, default=10_000)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=5,
        help="Freeze the visual tower for this many initial epochs.",
    )
    parser.add_argument(
        "--backbone-lr-stages01",
        type=float,
        default=1e-7,
        help="Backbone LR for stem + stages 0-1 after unfreeze.",
    )
    parser.add_argument(
        "--backbone-lr-stages23",
        type=float,
        default=3e-6,
        help="Backbone LR for stages 2-3 after unfreeze.",
    )
    parser.add_argument(
        "--backbone-lr-head-norm",
        type=float,
        default=8e-6,
        help="Backbone LR for trunk norm/head + visual projection head after unfreeze.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help=(
            "Max global gradient norm passed to clip_grad_norm_. Set to 0 or "
            "negative to disable clipping (default: 1.0)."
        ),
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument(
        "--zone-embed-dim",
        type=int,
        default=64,
        help=(
            "Dimensionality of the learned per-zone embedding concatenated with "
            "CLIP image features before the classifier head. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--head-hidden",
        type=int,
        default=256,
        help="Hidden width of the classifier MLP head (0 for linear).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early-stopping patience in epochs (counted on val macro-F1).",
    )
    parser.add_argument("--wandb-project", type=str, default="uveitis-per-zone")
    parser.add_argument("--wandb-run-name", type=str, default="clip-convnext-large-d-320")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument(
        "--soft-labels",
        action="store_true",
        help=(
            "Use soft tier targets [1,0]/[0.5,0.5]/[0,1] for training only. "
            "Requires the multiclass CSV with raw 0/1/2 Zone_Label values."
        ),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
        transforms.RandomErasing(p=0.2),
    ])


class ClipConvNeXtClassifier(nn.Module):
    """OpenCLIP ConvNeXt-Large-D visual tower + zone-aware classifier head.

    ``zone_numbers`` (1..10) gets embedded and concatenated with the visual
    features before the classifier MLP. The model conditions its output on the
    requested anatomical zone instead of treating each zone crop identically.
    """

    def __init__(
        self,
        visual: nn.Module,
        num_classes: int,
        zone_embed_dim: int = 64,
        head_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.zone_embed_dim = zone_embed_dim
        if zone_embed_dim > 0:
            self.zone_embedding = nn.Embedding(NUM_ZONES + 1, zone_embed_dim)
            head_in = CLIP_OUTPUT_DIM + zone_embed_dim
        else:
            self.zone_embedding = None
            head_in = CLIP_OUTPUT_DIM

        if head_hidden > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Dropout(p=0.4),
                nn.Linear(head_in, head_hidden),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Dropout(p=0.3),
                nn.Linear(head_in, num_classes),
            )

    def forward(
        self, images: torch.Tensor, zone_numbers: torch.Tensor
    ) -> torch.Tensor:
        features = self.visual(images)
        if isinstance(features, tuple):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        if self.zone_embedding is not None:
            zone_features = self.zone_embedding(zone_numbers)
            features = torch.cat([features, zone_features], dim=1)
        return self.classifier(features)


def visual_output_dim(visual: nn.Module) -> int | None:
    head_mlp = getattr(getattr(visual, "head", None), "mlp", None)
    head_fc2 = getattr(head_mlp, "fc2", None)
    out_features = getattr(head_fc2, "out_features", None)
    return int(out_features) if out_features is not None else None


def build_model_and_eval_transform(
    num_classes: int,
    zone_embed_dim: int = 64,
    head_hidden: int = 256,
) -> tuple[ClipConvNeXtClassifier, transforms.Compose]:
    clip_model, _, eval_transform = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
    )
    visual = clip_model.visual
    output_dim = visual_output_dim(visual)
    if output_dim != CLIP_OUTPUT_DIM:
        raise ValueError(
            f"Expected {CLIP_MODEL_NAME} visual output dim {CLIP_OUTPUT_DIM}, got {output_dim}."
        )
    del clip_model
    model = ClipConvNeXtClassifier(
        visual,
        num_classes,
        zone_embed_dim=zone_embed_dim,
        head_hidden=head_hidden,
    )
    return model, eval_transform


def set_backbone_trainable(model: ClipConvNeXtClassifier, trainable: bool) -> None:
    for parameter in model.visual.parameters():
        parameter.requires_grad = trainable


def backbone_param_groups(
    visual: nn.Module,
    lr_stages01: float,
    lr_stages23: float,
    lr_head_norm: float,
) -> list[dict[str, Any]]:
    """Layer-wise AdamW groups for the OpenCLIP ConvNeXt visual tower."""
    trunk = visual.trunk
    stages01_params: list[nn.Parameter] = []
    for module in (trunk.stem, trunk.stages[0], trunk.stages[1]):
        stages01_params.extend(module.parameters())

    stages23_params: list[nn.Parameter] = []
    for module in (trunk.stages[2], trunk.stages[3]):
        stages23_params.extend(module.parameters())

    head_norm_params: list[nn.Parameter] = []
    head_norm_params.extend(trunk.head.parameters())
    if getattr(visual, "head", None) is not None:
        head_norm_params.extend(visual.head.parameters())

    return [
        {"params": stages01_params, "lr": lr_stages01, "name": "backbone_stages_01"},
        {"params": stages23_params, "lr": lr_stages23, "name": "backbone_stages_23"},
        {"params": head_norm_params, "lr": lr_head_norm, "name": "backbone_head_norm"},
    ]


def build_optimizer(
    model: ClipConvNeXtClassifier,
    args: argparse.Namespace,
) -> torch.optim.AdamW:
    param_groups = backbone_param_groups(
        model.visual,
        lr_stages01=args.backbone_lr_stages01,
        lr_stages23=args.backbone_lr_stages23,
        lr_head_norm=args.backbone_lr_head_norm,
    )
    param_groups.append(
        {"params": model.classifier.parameters(), "lr": args.lr, "name": "classifier"}
    )
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    return trainable, frozen


def batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    if labels.dtype != torch.float32:
        labels = labels.long()
    zones = batch["zone_number"].to(device, dtype=torch.long)
    return images, labels, zones


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
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
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
            images, labels, zones = batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, zones)
                loss = criterion(logits, labels)

            hard_preds = logits.argmax(dim=1)
            hard_labels = labels.argmax(dim=1) if labels.dtype == torch.float32 else labels

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip_norm is not None and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=grad_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=grad_clip_norm
                        )
                    optimizer.step()

            if return_probs:
                probs = torch.softmax(logits.float(), dim=1).cpu().tolist()
                for true_label, prob_vec in zip(hard_labels.cpu().tolist(), probs):
                    collected_probs.append((true_label, prob_vec))

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            update_confusion(confusion, hard_preds.cpu(), hard_labels.cpu())

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
    use_amp = device.type == "cuda" and not args.no_amp

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

    model, eval_transform = build_model_and_eval_transform(
        args.num_classes,
        zone_embed_dim=args.zone_embed_dim,
        head_hidden=args.head_hidden,
    )
    train_transform = build_train_transform(args.image_size)
    set_backbone_trainable(model, trainable=args.freeze_backbone_epochs <= 0)
    model = model.to(device)

    train_loader = DataLoader(
        ZoneImageDataset(train_records, train_transform, soft_labels=args.soft_labels),
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

    if args.loss == "focal":
        alpha = torch.tensor([1.0, 2.0], device=device)
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha)
    else:
        criterion = build_loss(
            args.loss,
            labels_for(train_records),
            args.num_classes,
            device,
            class_weighting=args.class_weighting,
            focal_gamma=args.focal_gamma,
        )
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    best_path = args.output_dir / "best.pt"
    history: list[dict[str, Any]] = []

    trainable_params, frozen_params = parameter_counts(model)
    print(f"Device: {device}")
    print(f"Loss: {args.loss} class_weighting={args.class_weighting}")
    print(
        f"Backbone: {CLIP_MODEL_NAME} | pretrained={CLIP_PRETRAINED} | "
        f"output_dim={CLIP_OUTPUT_DIM} | trainable_params={trainable_params:,} | "
        f"frozen_params={frozen_params:,}"
    )
    print(
        f"Scheduler: cosine | Head LR: {args.lr} | "
        f"Backbone LRs: stages01={args.backbone_lr_stages01:.1e} "
        f"stages23={args.backbone_lr_stages23:.1e} "
        f"head_norm={args.backbone_lr_head_norm:.1e} | "
        f"grad_clip_norm: {args.grad_clip_norm} | amp: {use_amp}"
    )
    print(f"Split: {json.dumps(split)}")

    patience = args.patience
    no_improve_count = 0
    backbone_is_trainable = args.freeze_backbone_epochs <= 0

    for epoch in range(1, args.epochs + 1):
        should_train_backbone = epoch > args.freeze_backbone_epochs
        if should_train_backbone != backbone_is_trainable:
            set_backbone_trainable(model, trainable=should_train_backbone)
            backbone_is_trainable = should_train_backbone
            status = "unfrozen" if should_train_backbone else "frozen"
            print(f"Backbone {status} at epoch {epoch}")

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.num_classes,
            optimizer,
            grad_clip_norm=args.grad_clip_norm,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            args.num_classes,
            use_amp=use_amp,
        )
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
        model,
        test_loader,
        criterion,
        device,
        args.num_classes,
        return_probs=True,
        use_amp=use_amp,
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
                    title=f"ROC Curve - Class {class_idx} (AUC={roc_auc:.3f})",
                )
            })

        wandb.finish()

    print(f"Best checkpoint: {best_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Test macro_f1={test_metrics['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
