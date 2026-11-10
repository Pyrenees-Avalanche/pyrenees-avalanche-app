"""Plotting / quick-look helpers for training and evaluation.

Kept free of module-level global state so functions can be reused from any
script by passing the normalization stats explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend (scripts save PNGs, never show)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def plot_pr_curve(pr_data, best_idx, save_path: Optional[str] = None, auprc: Optional[float] = None) -> None:
    """Plot a precision-recall curve and mark the chosen operating point.

    Args:
        pr_data: dict with "precision"/"recall" (length N+1) and "thresholds" (length N).
        best_idx: index into precision/recall (0..N) of the selected operating point.
        save_path: path to save a PNG; if None, shows interactively.
        auprc: optional AUPRC to display in the title.
    """
    precision = np.asarray(pr_data["precision"])
    recall = np.asarray(pr_data["recall"])
    thresholds = np.asarray(pr_data["thresholds"])

    # Threshold vector aligned to precision/recall (N+1): [0.0] + thresholds + [1.0]
    thr_aligned = np.concatenate(([0.0], thresholds, [1.0]))

    if auprc is None:
        auprc = float(np.trapz(precision, recall))

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AUPRC = {auprc:.4f})")
    plt.grid(True, alpha=0.3)

    bx, by = recall[best_idx], precision[best_idx]
    bthr = thr_aligned[best_idx]
    plt.scatter([bx], [by], s=60)
    plt.annotate(
        f"best @ tau={bthr:.3f}\nP={by:.3f}, R={bx:.3f}",
        (bx, by), textcoords="offset points", xytext=(8, -18),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def save_test_quicklooks(
    idx: int,
    pre: torch.Tensor,
    post: torch.Tensor,
    aux: torch.Tensor,
    gt: torch.Tensor,
    pred: torch.Tensor,
    bin_mask: torch.Tensor,
    out_dir: Path,
    img_mean_db: torch.Tensor,
    img_std_db: torch.Tensor,
    sentinel_z_img: torch.Tensor,
) -> None:
    """Save per-sample RGB quick-looks, auxiliary rasters, GT/prob/binary masks
    and a coloured confusion map for one test patch.

    Patches containing sentinel (no-data) SAR values are skipped.
    """
    # Skip patches that contain the sentinel (no-data) value in VV/VH.
    if any(
        torch.any(pre[c] == sentinel_z_img[c]) or torch.any(post[c] == sentinel_z_img[c])
        for c in range(2)
    ):
        return

    def z_to_db_band(z, c: int):
        return z * img_std_db[c].to(z.device) + img_mean_db[c].to(z.device)

    def joint_minmax_norm(pre_b: torch.Tensor, post_b: torch.Tensor, c: int):
        pre_db = z_to_db_band(pre_b, c)
        post_db = z_to_db_band(post_b, c)
        joint_min = torch.min(pre_db.min(), post_db.min())
        joint_max = torch.max(pre_db.max(), post_db.max())
        pre_norm = (pre_db - joint_min) / (joint_max - joint_min + 1e-6)
        post_norm = (post_db - joint_min) / (joint_max - joint_min + 1e-6)
        return pre_norm.clamp(0, 1).cpu().numpy(), post_norm.clamp(0, 1).cpu().numpy()

    vv_pre, vv_post = joint_minmax_norm(pre[0], post[0], c=0)
    vh_pre, vh_post = joint_minmax_norm(pre[1], post[1], c=1)

    vv_rgb = (np.stack([vv_pre, vv_post, vv_pre], axis=-1) * 255).astype(np.uint8)
    vh_rgb = (np.stack([vh_pre, vh_post, vh_pre], axis=-1) * 255).astype(np.uint8)

    # Auxiliary rasters
    lia = aux[0].cpu().numpy().astype(np.float32)
    slope = aux[1].cpu().numpy().astype(np.float32)
    lia_img = ((lia - lia.min()) / (np.ptp(lia) + 1e-9) * 255).astype(np.uint8)
    slope_img = ((slope - slope.min()) / (np.ptp(slope) + 1e-9) * 255).astype(np.uint8)

    # Masks
    g = gt.squeeze().cpu().numpy().astype(bool)
    prb = pred.squeeze().cpu().numpy()  # raw probability in [0,1]
    bin_mask = bin_mask.squeeze().cpu().numpy().astype(bool)

    # Coloured confusion map
    err = np.zeros((*g.shape, 3), dtype=np.uint8)
    err[~g & ~bin_mask] = (0, 0, 0)      # TN - black
    err[g & bin_mask] = (0, 255, 0)      # TP - green
    err[~g & bin_mask] = (255, 255, 0)   # FP - yellow
    err[g & ~bin_mask] = (255, 0, 0)     # FN - red

    sample_dir = out_dir / str(idx)
    sample_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(vv_rgb).save(sample_dir / "vv_rgb.png")
    Image.fromarray(vh_rgb).save(sample_dir / "vh_rgb.png")
    Image.fromarray(lia_img).save(sample_dir / "lia.png")
    Image.fromarray(slope_img).save(sample_dir / "slope.png")
    Image.fromarray(err).save(sample_dir / "pred_confusion.png")
    Image.fromarray((g * 255).astype(np.uint8)).save(sample_dir / "gt.png")
    Image.fromarray((prb * 255).astype(np.uint8)).save(sample_dir / "pred_prob.png")
    Image.fromarray((bin_mask * 255).astype(np.uint8)).save(sample_dir / "pred_bin.png")
