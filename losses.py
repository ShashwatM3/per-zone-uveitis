from __future__ import annotations

from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F


class SoftCrossEntropyLoss(nn.Module):
    """Cross-entropy loss that accepts soft float target distributions."""

    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dtype in (torch.long, torch.int64):
            targets = F.one_hot(targets, num_classes=logits.shape[1]).float()
        log_probs = F.log_softmax(logits, dim=1)
        if self.weight is not None:
            log_probs = log_probs * self.weight.unsqueeze(0)
        loss = -(targets * log_probs).sum(dim=1)
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for logits shaped [batch, num_classes] (binary or multi-class)."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none")
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            loss = loss * self.alpha.gather(0, targets)

        if sample_weight is not None:
            weight = sample_weight.float()
            return (loss * weight).sum() / weight.sum().clamp(min=1e-8)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def class_weights(
    labels: list[int],
    num_classes: int,
    mode: str,
    beta: float = 0.999,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = Counter(labels)
    values: list[float] = []
    for class_idx in range(num_classes):
        count = counts.get(class_idx, 0)
        if count == 0:
            values.append(0.0)
        elif mode == "inverse":
            values.append(1.0 / count)
        elif mode == "effective":
            values.append((1.0 - beta) / (1.0 - beta**count))
        else:
            raise ValueError("class weighting must be one of: none, inverse, effective")

    weights = torch.tensor(values, dtype=torch.float32)
    nonzero = weights > 0
    if nonzero.any():
        weights[nonzero] = weights[nonzero] / weights[nonzero].mean()
    return weights


def build_loss(
    loss_name: str,
    train_labels: list[int],
    num_classes: int,
    device: torch.device,
    class_weighting: str = "inverse",
    focal_gamma: float = 2.0,
) -> nn.Module:
    weights = class_weights(train_labels, num_classes, class_weighting)
    if weights is not None:
        weights = weights.to(device)

    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=weights)
    if loss_name == "soft_ce":
        return SoftCrossEntropyLoss(weight=weights)
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=weights)
    raise ValueError("loss must be one of: ce, soft_ce, focal")
