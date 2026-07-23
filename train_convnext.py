#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import timm
import torch
import wandb
from sklearn.metrics import roc_curve, auc
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from losses import build_loss
from zone_dataset import (
    ZoneImageDataset,
    exclude_fovea_fallback,
    labels_for,
    load_or_create_split,
    load_zone_records,
    records_for_patients,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_ZONES = 10  # zone numbers are 1..10; embedding table has NUM_ZONES + 1 slots


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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Classifier-head learning rate. Backbone is trained at lr * 0.1.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--positive-oversample-factor",
        type=float,
        default=1.0,
        help=(
            "Relative sampling weight multiplier for positive-class (label=1) training "
            "examples. Set to 2.0 for 2x positive oversampling; 1.0 disables oversampling."
        ),
    )
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split-attempts", type=int, default=10_000)
    parser.add_argument(
        "--zone-embed-dim",
        type=int,
        default=64,
        help=(
            "Dimensionality of the learned per-zone embedding concatenated with "
            "image features before the classifier head. Set to 0 to disable "
            "zone-conditioning."
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
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help=(
            "Max global gradient norm passed to clip_grad_norm_. Set to 0 or "
            "negative to disable clipping (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--backbone",
        choices=("convnext_tiny", "efficientnet_b3"),
        default="convnext_tiny",
        help="Visual backbone (efficientnet_b3 uses timm ImageNet-21k weights).",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed-precision training.",
    )
    parser.add_argument("--wandb-project", type=str, default="uveitis-per-zone")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument(
        "--soft-labels",
        action="store_true",
        help=(
            "Use soft tier targets [1,0]/[0.5,0.5]/[0,1] for training only. "
            "Requires the multiclass CSV with raw 0/1/2 Zone_Label values."
        ),
    )
    parser.add_argument(
        "--exclude-tier1",
        action="store_true",
        help=(
            "Exclude tier-1 (raw_label==1) samples from training only. "
            "Validation/test are left unchanged for fair comparison."
        ),
    )
    parser.add_argument(
        "--exclude-fovea-fallback",
        action="store_true",
        help=(
            "Drop zone rows whose source FP used image-center fovea fallback "
            "(fovea_fallback=True in the sidecar .json). Their zone masks are "
            "geometrically wrong. Unlike --exclude-tier1 (a label question, "
            "train-only), this is an input-corruption question and is applied to "
            "ALL splits (train/val/test)."
        ),
    )
    parser.add_argument(
        "--balanced-batches",
        action="store_true",
        help=(
            "Use class-balanced WeightedRandomSampler (BBFL batch-balancing). "
            "Mutually exclusive with --positive-oversample-factor > 1."
        ),
    )
    parser.add_argument(
        "--tier-confidence-weights",
        action="store_true",
        help=(
            "Down-weight tier-1 (ambiguous) samples in the loss (0.3 vs 1.0 for tier 0/2). "
            "Requires multiclass CSV with raw 0/1/2 labels."
        ),
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="MixUp Beta(alpha, alpha) on training batches; 0 disables (Branch D4).",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    # After OS-flipping in preprocessing the anatomy is OD-aligned but each
    # zone crop is rotationally roughly symmetric, so we keep rotation/flips
    # but tone down ColorJitter (hue/saturation perturbations can erase the
    # subtle color cues that uveitis findings rely on).
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, scale=(0.75, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
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


class ZoneAwareTimmBackbone(nn.Module):
    """timm backbone (e.g. EfficientNet-B3) + zone embedding + MLP head."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool,
        zone_embed_dim: int,
        head_hidden: int,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features
        self.zone_embed_dim = zone_embed_dim
        if zone_embed_dim > 0:
            self.zone_embedding = nn.Embedding(NUM_ZONES + 1, zone_embed_dim)
            head_in = in_features + zone_embed_dim
        else:
            self.zone_embedding = None
            head_in = in_features

        if head_hidden > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Dropout(p=0.3),
                nn.Linear(head_in, head_hidden),
                nn.GELU(),
                nn.Dropout(p=0.2),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Dropout(p=0.3),
                nn.Linear(head_in, num_classes),
            )

    @property
    def features(self) -> nn.Module:
        return self.backbone

    def forward(self, images: torch.Tensor, zone_numbers: torch.Tensor) -> torch.Tensor:
        image_features = self.backbone(images)
        if self.zone_embedding is not None:
            zone_features = self.zone_embedding(zone_numbers)
            features = torch.cat([image_features, zone_features], dim=1)
        else:
            features = image_features
        return self.classifier(features)


class ZoneAwareConvNeXt(nn.Module):
    """ConvNeXt backbone with a classifier head that also consumes ``zone_number``.

    The model receives ``(image, zone_number)`` per sample. Image features come
    out of the standard ConvNeXt avgpool+flatten stack; ``zone_number`` (1..10)
    is mapped through a small learned embedding table and concatenated with
    the visual features before the final MLP. This lets a single model learn
    zone-conditional decision boundaries (zone-7/8/10 distributions differ
    sharply from zone-1/2/3/4) instead of pretending all zones look the same.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool,
        zone_embed_dim: int,
        head_hidden: int,
    ) -> None:
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        in_features = backbone.classifier[-1].in_features
        # Keep features + avgpool + flatten + layernorm; drop the final Linear.
        backbone.classifier[-1] = nn.Identity()
        self.backbone = backbone
        self.zone_embed_dim = zone_embed_dim
        if zone_embed_dim > 0:
            self.zone_embedding = nn.Embedding(NUM_ZONES + 1, zone_embed_dim)
            head_in = in_features + zone_embed_dim
        else:
            self.zone_embedding = None
            head_in = in_features

        if head_hidden > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Dropout(p=0.3),
                nn.Linear(head_in, head_hidden),
                nn.GELU(),
                nn.Dropout(p=0.2),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Dropout(p=0.3),
                nn.Linear(head_in, num_classes),
            )

    @property
    def features(self) -> nn.Module:
        # Surface the conv stack so the optimizer can split lr groups.
        return self.backbone.features

    def forward(self, images: torch.Tensor, zone_numbers: torch.Tensor) -> torch.Tensor:
        image_features = self.backbone(images)
        if self.zone_embedding is not None:
            zone_features = self.zone_embedding(zone_numbers)
            features = torch.cat([image_features, zone_features], dim=1)
        else:
            features = image_features
        return self.classifier(features)


def build_model(
    num_classes: int,
    pretrained: bool,
    zone_embed_dim: int = 64,
    head_hidden: int = 256,
    backbone: str = "convnext_tiny",
) -> nn.Module:
    if backbone == "convnext_tiny":
        return ZoneAwareConvNeXt(
            num_classes=num_classes,
            pretrained=pretrained,
            zone_embed_dim=zone_embed_dim,
            head_hidden=head_hidden,
        )
    if backbone == "efficientnet_b3":
        return ZoneAwareTimmBackbone(
            backbone_name="efficientnet_b3.ra2_in1k",
            num_classes=num_classes,
            pretrained=pretrained,
            zone_embed_dim=zone_embed_dim,
            head_hidden=head_hidden,
        )
    raise ValueError(f"Unknown backbone: {backbone}")


def batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    if labels.dtype != torch.float32:
        labels = labels.long()
    zones = batch["zone_number"].to(device, dtype=torch.long)
    sample_weight = batch.get("sample_weight")
    if sample_weight is not None:
        sample_weight = sample_weight.to(device, dtype=torch.float32)
    return images, labels, zones, sample_weight


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
    scaler: "torch.amp.GradScaler | None" = None,
    use_amp: bool = False,
    mixup_alpha: float = 0.0,
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
            images, labels, zones, sample_weight = batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if (
                    is_train
                    and mixup_alpha > 0
                    and labels.dtype != torch.float32
                    and images.size(0) > 1
                ):
                    lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item())
                    index = torch.randperm(images.size(0), device=device)
                    mixed_images = lam * images + (1.0 - lam) * images[index]
                    logits = model(mixed_images, zones)
                    loss_a = criterion(logits, labels)
                    loss_b = criterion(logits, labels[index])
                    loss = lam * loss_a + (1.0 - lam) * loss_b
                    hard_labels = labels
                else:
                    logits = model(images, zones)
                    if sample_weight is not None and hasattr(criterion, "forward"):
                        try:
                            loss = criterion(logits, labels, sample_weight=sample_weight)
                        except TypeError:
                            loss = criterion(logits, labels)
                    else:
                        loss = criterion(logits, labels)
                    hard_labels = labels.argmax(dim=1) if labels.dtype == torch.float32 else labels

            hard_preds = logits.argmax(dim=1)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None and grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
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
    if args.exclude_fovea_fallback:
        before = len(records)
        records = exclude_fovea_fallback(records)
        print(
            f"Excluded fovea-fallback: dropped {before - len(records)} zone rows "
            f"(geometrically wrong masks); {len(records)} rows remain."
        )
    split = load_or_create_split(
        records,
        split_json=args.split_json,
        seed=args.seed,
        attempts=args.split_attempts,
    )
    train_records = records_for_patients(records, split["train"])
    val_records = records_for_patients(records, split["val"])
    test_records = records_for_patients(records, split["test"])
    if args.exclude_tier1:
        has_tier2 = any(record.raw_label == 2 for record in train_records)
        if not has_tier2:
            raise SystemExit(
                "--exclude-tier1 requires multiclass source labels (0/1/2). "
                "Use processed_image_arrays_multiclass CSV/data-root."
            )
        train_records = [record for record in train_records if record.raw_label != 1]

    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset = ZoneImageDataset(
        train_records,
        train_transform,
        soft_labels=args.soft_labels,
        tier_confidence_weights=args.tier_confidence_weights,
    )
    train_sampler = None
    train_shuffle = True
    if args.balanced_batches and args.positive_oversample_factor > 1.0:
        raise SystemExit(
            "Use either --balanced-batches or --positive-oversample-factor > 1, not both."
        )
    if args.balanced_batches:
        label_counts = Counter(labels_for(train_records))
        sample_weights = [
            1.0 / label_counts[record.label] for record in train_records
        ]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
    elif args.positive_oversample_factor > 1.0:
        sample_weights = [
            float(args.positive_oversample_factor) if record.label == 1 else 1.0
            for record in train_records
        ]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
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

    model = build_model(
        args.num_classes,
        pretrained=not args.no_pretrained,
        zone_embed_dim=args.zone_embed_dim,
        head_hidden=args.head_hidden,
        backbone=args.backbone,
    ).to(device)
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
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    best_path = args.output_dir / "best.pt"
    history: list[dict[str, Any]] = []

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Loss: {args.loss} class_weighting={args.class_weighting}")
    print(
        f"Image size: {args.image_size} | zone_embed_dim: {args.zone_embed_dim} | "
        f"head_hidden: {args.head_hidden} | epochs: {args.epochs} | "
        f"batch_size: {args.batch_size} | patience: {args.patience}"
    )
    print(
        f"Scheduler: cosine | "
        f"Head LR: {args.lr} | Backbone LR: {args.lr * 0.1} | "
        f"grad_clip_norm: {args.grad_clip_norm}"
    )
    print(f"Split: {json.dumps(split)}")

    patience = args.patience
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
            scaler=scaler,
            use_amp=use_amp,
            mixup_alpha=args.mixup_alpha,
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
