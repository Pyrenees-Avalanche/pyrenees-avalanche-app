#!/usr/bin/env python3
"""
Train a Change Detection model for Avalanches with recall-friendly thresholding
and small-positive–aware losses.

- Threshold selection supports F-beta (β>1 favors recall) and/or a precision
  floor (maximize recall s.t. precision ≥ p0).
- Loss options include BCE, BCE+Dice, and Focal-Tversky (good for tiny positives).

Shared building blocks (losses, model construction, seeding, plotting, runtime
helpers) live in the ``utils`` / ``models`` packages.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecisionRecallCurve,
)
from torchmetrics.functional import jaccard_index
from tqdm import tqdm

from dataset.avalanches import AvalancheDataset
from dataset.sampler import BalancedPosNegSampler
from models.build import build_model, model_choices
from utils.losses import build_loss
from utils.reproducibility import SEED, make_generator, seed_worker, set_seed
from utils.runtime import get_device, log_param_count, setup_logging
from utils.visualization import plot_pr_curve

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

set_seed(SEED)
GEN = make_generator(SEED)

# --------------------------------------------------------------------------- #
# Argparse
# --------------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Train a Change Detection model for Avalanches."
)

# Experiment/config
parser.add_argument("--description", type=str, required=True, help="Experiment note.")
parser.add_argument(
    "--model",
    type=str,
    choices=model_choices(),
    help="Architecture.",
)
parser.add_argument(
    "--model-size",
    type=str,
    choices=["tiny", "small", "base"],
    default="tiny",
    help="SwinUNet size.",
)
parser.add_argument(
    "--fusion-type",
    type=str,
    choices=["diff", "agm", "cross"],
    default="diff",
    help="Fusion inside SwinUNet encoder.",
)
parser.add_argument(
    "--patch-size", type=int, default=128, help="Training/loader patch size."
)
parser.add_argument(
    "--dataset-root",
    type=str,
    default="/home/jovyan/nfs/mgatti/datasets/Avalanches/patches/",
    help="Dataset root (without patch size).",
)
parser.add_argument(
    "--use-aux", action="store_true", help="Enable auxiliary data input."
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=10,
    help="Warmup epochs before cosine schedule/checkpointing.",
)
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Mini-batch size for train/val DataLoaders.",
)

# Threshold selection knobs
parser.add_argument(
    "--beta",
    type=float,
    default=1,
    help="F-beta for threshold selection (β>1 favors recall). Use 1.0 for F1.",
)
parser.add_argument(
    "--precision-floor",
    type=float,
    default=0.5,
    help="If >0, pick threshold maximizing recall s.t. precision ≥ this value.",
)
parser.add_argument(
    "--pos-ratio",
    type=float,
    default=1.0,
    help="ratio = positives : negatives."
)

# Loss selection knobs
parser.add_argument(
    "--loss",
    type=str,
    choices=["bce", "bce_dice", "focal_tversky"],
    default="bce",
    help="Loss to use.",
)
parser.add_argument(
    "--pos-weight",
    type=float,
    default=3.0,
    help="Positive class weight for BCE-based losses.",
)
parser.add_argument(
    "--dice-weight",
    type=float,
    default=0.5,
    help="Weight for Dice term when using bce_dice.",
)
parser.add_argument(
    "--bce-weight",
    type=float,
    default=0.5,
    help="Weight for BCE term when using bce_dice.",
)
parser.add_argument(
    "--tversky-alpha",
    type=float,
    default=0.7,
    help="FN weight for Focal Tversky (↑alpha favors recall).",
)
parser.add_argument(
    "--tversky-beta",
    type=float,
    default=0.3,
    help="FP weight for Focal Tversky (↑beta favors precision).",
)
parser.add_argument(
    "--tversky-gamma",
    type=float,
    default=1.33,
    help="Focal exponent for Focal Tversky.",
)

args = parser.parse_args()

# --------------------------------------------------------------------------- #
# Config & paths
# --------------------------------------------------------------------------- #

MODEL_SIZE = args.model_size
PATCH_SIZE = args.patch_size
USE_AUX = args.use_aux
FUSION_TYPE = args.fusion_type

LOGGER = setup_logging("TrainLogger")

BATCH_SIZE = args.batch_size
MAIN_EPOCHS = 100
WARMUP_EPOCHS = args.warmup_epochs
TOTAL_EPOCHS = WARMUP_EPOCHS + MAIN_EPOCHS
PATIENCE = 20

LR = args.lr

MODEL_SELECTION_METRIC = "AUPRC"  # still select best by AUPRC

DEVICE = get_device()

DATASET_DIR = Path(args.dataset_root) / f"{PATCH_SIZE}"
TRAIN_EVENTS = [
    "Livigno_20240403",
    "Livigno_20250129",
    "Livigno_20250318",
    "Nuuk_20160413",
    "Pish_20230221",
]
VAL_EVENTS = ["Nuuk_20210411"]

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

model = build_model(
    args.model,
    patch_size=PATCH_SIZE,
    use_aux=USE_AUX,
    model_size=MODEL_SIZE,
    fusion_type=FUSION_TYPE,
    device=DEVICE,
)

# --------------------------------------------------------------------------- #
# Loss & optimizer
# --------------------------------------------------------------------------- #

criterion = build_loss(
    args.loss,
    device=DEVICE,
    pos_weight=args.pos_weight,
    bce_weight=args.bce_weight,
    dice_weight=args.dice_weight,
    tversky_alpha=args.tversky_alpha,
    tversky_beta=args.tversky_beta,
    tversky_gamma=args.tversky_gamma,
)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
cosine = CosineAnnealingLR(optimizer, T_max=MAIN_EPOCHS)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, cosine] if WARMUP_EPOCHS > 0 else [cosine],
    milestones=[WARMUP_EPOCHS] if WARMUP_EPOCHS > 0 else [],
)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

train_dataset = AvalancheDataset(DATASET_DIR, TRAIN_EVENTS)
train_sampler = BalancedPosNegSampler(train_dataset, patch_size=PATCH_SIZE, ratio=args.pos_ratio)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=GEN,
)

val_dataset = AvalancheDataset(DATASET_DIR, VAL_EVENTS, apply_transform=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=GEN,
)

DATASET_STATS = train_dataset.stats

# --------------------------------------------------------------------------- #
# Utils
# --------------------------------------------------------------------------- #


def create_experiment_dir() -> Path:
    """Create a uniquely named experiment directory."""
    base_name = f"{args.model}_{PATCH_SIZE}"
    if USE_AUX:
        base_name += "_aux"

    exp_root = Path("exp")
    exp_root.mkdir(exist_ok=True)

    experiment_dir = exp_root / base_name
    count = 2
    while experiment_dir.exists():
        experiment_dir = exp_root / f"{base_name}_{count}"
        count += 1

    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def extract_criterion_params(loss_obj: nn.Module) -> Dict[str, str]:
    """Extract parameters from a loss object, if available."""
    try:
        return {k: v for k, v in vars(loss_obj).items() if not k.startswith("_")}
    except AttributeError:
        return {}


def log_experiment(
    experiment_dir: Path,
    model_name: str,
    criterion_name: str,
    criterion_params: Dict[str, str],
    optimizer_name: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    best_metrics: Dict[str, float],
    best_threshold: float,
    patch_size: int,
    train_events: list[str],
    val_events: list[str],
) -> None:
    """
    Log experiment parameters and best validation metrics into a CSV file.
    Each experiment directory has a single row (update if exists).
    """
    log_file = Path("exp", "experiments.csv")
    headers = [
        "Timestamp",
        "Experiment_Dir",
        "Description",
        "Model",
        "Loss",
        "Loss_Params",
        "Optimizer",
        "LR",
        "Batch_Size",
        "Epochs",
        "Patch_Size",
        "Best_Threshold",
        "Best_AUPRC",
        "Best_F1",
        "Best_FBeta",
        "Best_Recall",
        "Best_Precision",
        "Best_IoU",
        "Validation_Loss",
        "Train_Events",
        "Val_Events",
    ]

    if isinstance(criterion_params, dict):
        criterion_params_str = "; ".join(
            [f"{k}={v}" for k, v in criterion_params.items()]
        )
    else:
        criterion_params_str = "None"

    def fmt(val):
        return f"{val:.4f}" if isinstance(val, (float, int)) else val

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = [
        timestamp,
        str(experiment_dir),
        args.description,
        model_name,
        criterion_name,
        criterion_params_str,
        optimizer_name,
        learning_rate,
        batch_size,
        epochs,
        patch_size,
        fmt(best_threshold),
        fmt(best_metrics.get("AUPRC", 0)),
        fmt(best_metrics.get("F1", 0)),
        fmt(best_metrics.get("FBeta", 0)),
        fmt(best_metrics.get("recall", 0)),
        fmt(best_metrics.get("precision", 0)),
        fmt(best_metrics.get("iou", 0)),
        fmt(best_metrics.get("val_loss", 0)),
        ";".join(train_events),
        ";".join(val_events),
    ]

    rows = []
    if log_file.exists():
        with open(log_file, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if rows and [h.strip().lower() for h in rows[0]] == [
            h.lower() for h in headers
        ]:
            rows = rows[1:]
        rows = [row for row in rows if row[1] != str(experiment_dir)]

    rows.append(new_entry)

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    LOGGER.info("Experiment logged in %s", log_file)


def save_model(
    model: nn.Module,
    epoch: int,
    val_loss: float,
    best_metrics: Dict[str, float],
    best_threshold: float,
    pr_curve_data: Dict[str, torch.Tensor],
    best_idx: int,
    experiment_dir: Path,
    norm_stats: Dict[str, torch.Tensor],
) -> None:
    model_path = experiment_dir / "best_model.pth"

    norm_stats = {k: v.detach().cpu() for k, v in norm_stats.items()}

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_threshold": best_threshold,
            "norm_stats": norm_stats,
        },
        model_path,
    )
    LOGGER.info("Best model saved at %s", model_path)

    pr_df = pd.DataFrame(
        {
            "precision": pr_curve_data["precision"].numpy(),
            "recall": pr_curve_data["recall"].numpy(),
            "threshold": [None] + pr_curve_data["thresholds"].numpy().tolist(),
        }
    )
    pr_csv_path = experiment_dir / "val_pr_curve.csv"
    pr_df.to_csv(pr_csv_path, index=False)

    pr_png_path = experiment_dir / "val_pr_curve.png"
    plot_pr_curve(
        pr_curve_data,
        best_idx=best_idx,
        save_path=str(pr_png_path),
        auprc=best_metrics.get("AUPRC", None),
    )

    log_experiment(
        experiment_dir=experiment_dir,
        model_name=model.__class__.__name__,
        criterion_name=criterion.__class__.__name__,
        criterion_params=extract_criterion_params(criterion),
        optimizer_name=optimizer.__class__.__name__,
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        epochs=TOTAL_EPOCHS,
        best_metrics=best_metrics,
        best_threshold=best_threshold,
        patch_size=PATCH_SIZE,
        train_events=TRAIN_EVENTS,
        val_events=VAL_EVENTS,
    )


# --------------------------------------------------------------------------- #
# Validation (with recall-friendly thresholding)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def validate(
    loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    beta: float,
    precision_floor: float,
) -> Tuple[Dict[str, float], float, Dict[str, torch.Tensor], int]:
    """
    - Streams val set through BinaryPrecisionRecallCurve (no memory blow-up).
    - Computes val loss, AUROC, AUPRC.
    - Chooses threshold by:
        * If precision_floor > 0: max recall s.t. precision ≥ floor.
        * Else: maximize F-beta (beta>1 favors recall; beta=1 => F1).
    """
    device = next(model.parameters()).device
    model.eval()

    pr_curve = BinaryPrecisionRecallCurve().to(device)
    auprc_m = BinaryAveragePrecision().to(device)
    auroc_m = BinaryAUROC().to(device)

    all_probs, all_masks = [], []
    running_loss, n_batches = 0.0, 0

    for batch in tqdm(loader, desc="Validating", ncols=100):
        img_pre = batch["pre"].to(device, non_blocking=True)
        img_post = batch["post"].to(device, non_blocking=True)
        gt_mask = batch["mask"].to(device, non_blocking=True)

        if not USE_AUX:
            logits = model(img_pre, img_post)
        else:
            aux = batch["aux"].to(device, non_blocking=True)
            logits = model(img_pre, img_post, aux)

        loss = loss_fn(logits, gt_mask.float())
        running_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits)
        pr_curve.update(probs, gt_mask)
        auprc_m.update(probs, gt_mask)
        auroc_m.update(probs, gt_mask)

        all_probs.append(probs.detach().cpu())
        all_masks.append(gt_mask.detach().cpu())

    # Precision/Recall curve
    precision, recall, thresholds = pr_curve.compute()  # prec/rec: N+1, thr: N

    # F-beta (same length as precision/recall)
    fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # Pick best index
    if precision_floor > 0:
        ok = precision >= precision_floor
        if ok.any():
            best_idx = torch.argmax(recall * ok).item()
        else:
            best_idx = torch.argmax(fbeta).item()  # fallback
    else:
        best_idx = torch.argmax(fbeta).item()

    # Map PR index (N+1) to threshold index (N)
    if best_idx == 0:
        best_thr = 0.0
    elif best_idx == len(thresholds):
        best_thr = 1.0
    else:
        best_thr = thresholds[best_idx - 1].item()

    # IoU at τ*
    y_prob = torch.cat(all_probs).flatten()
    y_true = torch.cat(all_masks).flatten()
    y_pred = (y_prob >= best_thr).int()
    iou = jaccard_index(y_pred, y_true.int(), task="binary").item()

    pr_data = {
        "precision": precision.cpu(),
        "recall": recall.cpu(),
        "thresholds": thresholds.cpu(),
    }

    auprc = auprc_m.compute().item()
    auroc = auroc_m.compute().item()
    val_loss = running_loss / max(n_batches, 1)

    metrics = {
        "val_loss": val_loss,
        "AUPRC": auprc,
        "AUROC": auroc,
        "F1": f1[best_idx].item(),
        "FBeta": fbeta[best_idx].item(),
        "precision": precision[best_idx].item(),
        "recall": recall[best_idx].item(),
        "iou": iou,
    }
    return metrics, best_thr, pr_data, best_idx


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: SequentialLR,
    epochs: int,
) -> None:
    experiment_dir: Path | None = None
    writer: SummaryWriter | None = None

    best_model_score = 0.0
    best_val_metrics: Dict[str, float] | None = None
    best_threshold = 0.5
    epochs_without_improvement = 0

    log_param_count(model, LOGGER)

    LOGGER.info("LR = %.6e", optimizer.param_groups[0]["lr"])

    for epoch in range(epochs):
        model.train()
        total_loss, total_samples = 0.0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100) as pbar:
            for batch in pbar:
                img_pre = batch["pre"].to(DEVICE)
                img_post = batch["post"].to(DEVICE)
                gt_mask = batch["mask"].to(DEVICE).float()

                optimizer.zero_grad(set_to_none=True)
                if not USE_AUX:
                    logits = model(img_pre, img_post)
                else:
                    aux = batch["aux"].to(DEVICE)
                    logits = model(img_pre, img_post, aux)

                loss = loss_fn(logits, gt_mask)
                loss.backward()
                optimizer.step()

                bs = gt_mask.shape[0]
                total_loss += loss.item() * bs
                total_samples += bs
                pbar.set_postfix(loss=total_loss / max(total_samples, 1))

        avg_train_loss = total_loss / max(total_samples, 1)
        LOGGER.info("Epoch [%d/%d], Train Loss: %.4f", epoch + 1, epochs, avg_train_loss)

        scheduler.step()

        in_warmup = epoch < WARMUP_EPOCHS

        # Create experiment directory after warmup starts
        if experiment_dir is None and not in_warmup:
            experiment_dir = create_experiment_dir()
            writer = SummaryWriter(log_dir=experiment_dir / "logs")

        # Validation with recall-friendly threshold tuning
        val_metrics, epoch_best_thr, pr_curve_data, best_idx = validate(
            val_loader,
            model,
            loss_fn,
            beta=args.beta,
            precision_floor=args.precision_floor,
        )
        model_selection_score = val_metrics[MODEL_SELECTION_METRIC]
        val_loss = val_metrics["val_loss"]

        metrics_str = ", ".join(
            [f"{k}: {v:.4f}" for k, v in val_metrics.items() if isinstance(v, float)]
        )
        LOGGER.info(
            "Validation (thr=%.3f) -> %s",
            epoch_best_thr,
            metrics_str,
        )

        # TensorBoard
        if writer is not None:
            writer.add_scalar("Train/Loss", avg_train_loss, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"Val/{k}", v, epoch)

        # Warmup-aware checkpointing / early stopping
        if in_warmup:
            LOGGER.info(
                "Warmup phase: not saving best model or applying early stopping yet."
            )
            epochs_without_improvement = 0
        else:
            if model_selection_score > best_model_score:
                best_model_score = model_selection_score
                best_val_metrics = val_metrics
                best_threshold = epoch_best_thr
                assert experiment_dir is not None
                save_model(
                    model=model,
                    epoch=epoch,
                    val_loss=val_loss,
                    best_metrics=best_val_metrics,
                    best_threshold=best_threshold,
                    pr_curve_data=pr_curve_data,
                    best_idx=best_idx,
                    experiment_dir=experiment_dir,
                    norm_stats=DATASET_STATS,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                LOGGER.info("No improvement in %d epoch(s).", epochs_without_improvement)
                if epochs_without_improvement >= PATIENCE:
                    LOGGER.info("Early stopping at epoch %d.", epoch + 1)
                    break

    if writer is not None:
        writer.close()

    if best_val_metrics is not None:
        LOGGER.info("\n===== Best Validation Metrics =====")
        LOGGER.info("Best Threshold: %.3f", best_threshold)
        for metric, value in best_val_metrics.items():
            LOGGER.info("%s: %.4f", metric, value)
        LOGGER.info("===================================\n")


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=TOTAL_EPOCHS,
    )