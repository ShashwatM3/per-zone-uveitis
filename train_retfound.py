#!/usr/bin/env python3
"""Fine-tune RETFound MAE (CFP) with zone-aware head on canonical patient split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

import torch
import wandb
from huggingface_hub import hf_hub_download
from sklearn.metrics import auc, roc_curve
from timm.models.layers import trunc_normal_
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

# Official RETFound implementation: https://github.com/rmaphoh/RETFound
RETFOUND_ROOT = Path(__file__).resolve().parent / "external" / "RETFound"
if str(RETFOUND_ROOT) not in sys.path:
    sys.path.insert(0, str(RETFOUND_ROOT))

import models_vit as retfound_models  # noqa: E402
from util.pos_embed import interpolate_pos_embed  # noqa: E402

from losses import build_loss
from train_convnext import (
    NUM_ZONES,
    batch_to_device,
    metrics_from_confusion,
    run_epoch,
    serializable_args,
    summarize_records,
    update_confusion,
)
from zone_dataset import (
    ZoneImageDataset,
    labels_for,
    load_or_create_split,
    load_zone_records,
    records_for_patients,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_FINETUNE_ID = "RETFound_mae_natureCFP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zone-aware RETFound MAE fine-tuning.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=Path("splits/canonical_split.json"))
    parser.add_argument("--loss", choices=("ce", "soft_ce", "focal"), default="focal")
    parser.add_argument("--class-weighting", choices=("none", "inverse", "effective"), default="inverse")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="Head learning rate.")
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-6,
        help="ViT backbone LR after freeze period.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split-attempts", type=int, default=10_000)
    parser.add_argument("--zone-embed-dim", type=int, default=64)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=4,
        help="Train head only for this many epochs, then unfreeze backbone.",
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default=DEFAULT_FINETUNE_ID,
        help="HuggingFace repo id suffix (YukunZhou/{id}) or local .pth path.",
    )
    parser.add_argument("--exclude-tier1", action="store_true")
    parser.add_argument("--balanced-batches", action="store_true")
    parser.add_argument("--tier-confidence-weights", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="uveitis-per-zone")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def weights_file_ready(path: Path, min_bytes: int = 500_000_000) -> bool:
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def resolve_checkpoint_path(finetune: str) -> Path:
    repo_root = Path(__file__).resolve().parent
    local_candidates = [
        repo_root / "weights" / "RETFound_mae_natureCFP.pth",
        repo_root / "RETFound_mae_natureCFP" / "RETFound_mae_natureCFP.pth",
    ]
    path = Path(finetune)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    candidate: Path | None = None
    if path.is_file():
        candidate = path
    elif finetune == DEFAULT_FINETUNE_ID:
        for local in local_candidates:
            if local.is_file():
                candidate = local
                break

    if candidate is not None:
        if weights_file_ready(candidate):
            return candidate
        raise RuntimeError(
            f"Checkpoint at {candidate} exists but is incomplete or corrupt "
            f"(size={candidate.stat().st_size}). Wait for download to finish."
        )

    print(f"Downloading YukunZhou/{finetune} from Hugging Face (gated — requires `huggingface-cli login`).")
    downloaded = hf_hub_download(
        repo_id=f"YukunZhou/{finetune}",
        filename=f"{finetune}.pth",
    )
    return Path(downloaded)


def load_retfound_weights(model: ZoneAwareRETFound, finetune: str, image_size: int) -> None:
    checkpoint_path = resolve_checkpoint_path(finetune)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}
    vit_state = model.vit.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in checkpoint_model and key in vit_state and checkpoint_model[key].shape != vit_state[key].shape:
            del checkpoint_model[key]
    interpolate_pos_embed(model.vit, checkpoint_model)
    missing, unexpected = model.vit.load_state_dict(checkpoint_model, strict=False)
    print(f"Loaded RETFound weights from {checkpoint_path}")
    if missing:
        print(f"  missing keys (sample): {missing[:5]}")
    if unexpected:
        print(f"  unexpected keys (sample): {unexpected[:5]}")


class ZoneAwareRETFound(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int,
        zone_embed_dim: int,
        head_hidden: int,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.vit = retfound_models.RETFound_mae(
            img_size=image_size,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool=True,
        )
        embed_dim = self.vit.embed_dim
        self.vit.head = nn.Identity()
        self.zone_embed_dim = zone_embed_dim
        if zone_embed_dim > 0:
            self.zone_embedding = nn.Embedding(NUM_ZONES + 1, zone_embed_dim)
            head_in = embed_dim + zone_embed_dim
        else:
            self.zone_embedding = None
            head_in = embed_dim
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
        return self.vit

    def forward(self, images: torch.Tensor, zone_numbers: torch.Tensor) -> torch.Tensor:
        image_features = self.vit.forward_features(images).squeeze(1)
        if self.zone_embedding is not None:
            zone_features = self.zone_embedding(zone_numbers)
            features = torch.cat([image_features, zone_features], dim=1)
        else:
            features = image_features
        return self.classifier(features)


def set_backbone_trainable(model: ZoneAwareRETFound, trainable: bool) -> None:
    for param in model.vit.parameters():
        param.requires_grad = trainable


def build_optimizer(model: ZoneAwareRETFound, head_lr: float, backbone_lr: float, weight_decay: float, train_backbone: bool):
    param_groups = [{"params": model.classifier.parameters(), "lr": head_lr, "name": "classifier"}]
    if model.zone_embedding is not None:
        param_groups.append({"params": model.zone_embedding.parameters(), "lr": head_lr, "name": "zone_embed"})
    if train_backbone:
        param_groups.append({"params": model.vit.parameters(), "lr": backbone_lr, "name": "backbone"})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            config=serializable_args(args),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    records = load_zone_records(args.csv, args.data_root)
    split = load_or_create_split(records, split_json=args.split_json, seed=args.seed, attempts=args.split_attempts)
    train_records = records_for_patients(records, split["train"])
    val_records = records_for_patients(records, split["val"])
    test_records = records_for_patients(records, split["test"])
    if args.exclude_tier1:
        train_records = [r for r in train_records if r.raw_label != 1]

    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset = ZoneImageDataset(
        train_records,
        train_transform,
        tier_confidence_weights=args.tier_confidence_weights,
    )
    train_sampler = None
    train_shuffle = True
    if args.balanced_batches:
        label_counts = Counter(labels_for(train_records))
        weights = [1.0 / label_counts[r.label] for r in train_records]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
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

    model = ZoneAwareRETFound(
        num_classes=args.num_classes,
        image_size=args.image_size,
        zone_embed_dim=args.zone_embed_dim,
        head_hidden=args.head_hidden,
    )
    load_retfound_weights(model, args.finetune, args.image_size)
    model = model.to(device)
    set_backbone_trainable(model, trainable=False)

    criterion = build_loss(
        args.loss,
        labels_for(train_records),
        args.num_classes,
        device,
        class_weighting=args.class_weighting,
        focal_gamma=args.focal_gamma,
    )
    optimizer = build_optimizer(model, args.lr, args.backbone_lr, args.weight_decay, train_backbone=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    best_macro_f1 = -1.0
    best_path = args.output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    patience = args.patience
    no_improve_count = 0

    print(f"Device: {device} | AMP: {use_amp} | RETFound finetune: {args.finetune}")
    print(f"Freeze backbone for {args.freeze_backbone_epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        train_backbone = epoch > args.freeze_backbone_epochs
        if train_backbone and not any(g.get("name") == "backbone" for g in optimizer.param_groups):
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(
                model, args.lr, args.backbone_lr, args.weight_decay, train_backbone=True
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1, eta_min=1e-7
            )
            print(f"epoch={epoch:03d} backbone unfrozen (lr={args.backbone_lr})")

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
            model, val_loader, criterion, device, args.num_classes, use_amp=use_amp
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        scheduler.step()

        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
        )
        print(f"epoch={epoch:03d} val/per_class={val_metrics['per_class']}")

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

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
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
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if use_wandb:
        wandb.log({
            "test/macro_f1": test_metrics["macro_f1"],
            "test/recall_class_1": test_metrics["per_class"]["1"]["recall"],
        })
        wandb.finish()

    print(f"Test macro_f1={test_metrics['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
