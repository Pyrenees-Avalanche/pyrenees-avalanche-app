"""Loss functions for small-positive-aware binary change detection.

This is the single source of truth for the project's losses. Training and
evaluation scripts build losses via :func:`build_loss`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def dice_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Functional soft-Dice loss. ``pred`` is expected to be probabilities."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice_score = (2.0 * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    return 1 - dice_score


class DiceLoss(nn.Module):
    """Sigmoid Dice loss over logits for binary segmentation."""

    def __init__(self, eps: float = 1e-6, ignore_empty: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.ignore_empty = ignore_empty

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        probs = probs.contiguous().view(-1)
        target = target.contiguous().view(-1)

        inter = (probs * target).sum()
        denom = probs.sum() + target.sum()

        if self.ignore_empty and target.sum() < 0.5:
            # If GT is empty, penalize predicted mass softly.
            return (probs.sum() / (probs.numel() + self.eps)).clamp(0, 1)

        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """BCE-with-logits + Dice with device-safe pos_weight and float targets.

    Args:
        bce_weight: weight of BCE term.
        dice_weight: weight of Dice term.
        pos_weight: positive class weight for BCE (float or tensor).
        ignore_empty: pass-through to DiceLoss for empty-GT behavior.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | torch.Tensor = 1.0,
        ignore_empty: bool = False,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        # Register as buffer so it moves with .to(device)
        self.register_buffer("pos_weight", pos_weight)

        self.dice = DiceLoss(ignore_empty=ignore_empty)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )
        dsc = self.dice(logits, target)
        return self.bce_weight * bce + self.dice_weight * dsc


class FocalLoss(nn.Module):
    """Binary Focal Loss over logits.

    Args:
        alpha: weight for positive class (e.g. 0.75).
        gamma: focusing parameter (e.g. 2.0).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)

        eps = 1e-8
        inputs = inputs.clamp(min=eps, max=1.0 - eps)

        p_t = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        loss = -alpha_t * ((1 - p_t) ** self.gamma) * torch.log(p_t)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalTverskyLoss(nn.Module):
    r"""Focal Tversky loss for imbalanced binary segmentation.

    Computes ``(1 - Tversky)^gamma`` with ``p = sigmoid(logits)``. Tversky
    balances FN and FP via ``alpha``/``beta``; ``gamma > 1`` focuses learning
    on hard/small targets. Good for tiny/sparse positives.

    Args:
        alpha: FN weight (increase to favor recall). Default: 0.7
        beta:  FP weight (increase to favor precision). Default: 0.3
        gamma: focal exponent (>1 emphasizes hard examples). Default: 1.33
        eps:   numerical stability. Default: 1e-7
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.33,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.alpha, self.beta, self.gamma, self.eps = alpha, beta, gamma, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = tuple(range(2, probs.ndim))  # spatial dims only

        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1 - targets)).sum(dim=dims)
        fn = ((1 - probs) * targets).sum(dim=dims)

        tversky = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        loss = (1.0 - tversky) ** self.gamma
        return loss.mean()


def build_loss(
    loss_name: str,
    *,
    device: torch.device,
    pos_weight: float = 3.0,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    tversky_alpha: float = 0.7,
    tversky_beta: float = 0.3,
    tversky_gamma: float = 1.33,
) -> nn.Module:
    """Create the chosen loss, emphasizing tiny positives.

    Args:
        loss_name: one of {"bce", "bce_dice", "focal_tversky"}.
        device: device the loss (and its buffers) should live on.
    """
    if loss_name == "bce":
        pos_w = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_w)

    if loss_name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
            ignore_empty=True,
        ).to(device)

    if loss_name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma
        ).to(device)

    raise ValueError(
        f"Unknown loss '{loss_name}'. Choose from: bce, bce_dice, focal_tversky."
    )
