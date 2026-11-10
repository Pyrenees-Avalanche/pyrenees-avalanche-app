"""Shared tiled-inference utilities for full-scene change detection.

Used by both ``infer.py`` (single-event inference) and ``test_blending.py``
(blending-mode evaluation). Groups together the raster discovery, region
masking, sliding-window patch extraction, per-patch normalization dataset, and
the patch-stitching / blending logic that used to be duplicated in each script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from scipy.ndimage import binary_fill_holes
from skimage.util.shape import view_as_windows
from torch.utils.data import Dataset

# SAR bands are always required; aux bands only when use_aux.
SAR_KEYS = ["preVH", "preVV", "postVH", "postVV"]
AUX_KEYS = ["LIA", "SLP"]

# SAR dB validity range (values outside are treated as invalid / NaN).
SAR_MIN_DB = -40.0
SAR_MAX_DB = 20.0

NODATA = 255


# --------------------------------------------------------------------------- #
# Raster discovery
# --------------------------------------------------------------------------- #
def find_raster_paths(
    event_path: Path,
    use_aux: bool,
    ensure_slope_fn: Optional[Callable[[Path, Path], Path]] = None,
) -> Dict[str, Path]:
    """Locate the expected rasters inside ``event_path``.

    Always requires the four SAR bands. When ``use_aux`` is set, also requires
    ``LIA`` and a slope raster ``SLP``. If ``ensure_slope_fn`` is provided, it is
    used to create/return the slope raster from a SAR reference; otherwise the
    ``SLP`` raster is expected to already exist in the folder.

    Raises:
        FileNotFoundError: if any required band is missing.
    """
    event_path = Path(event_path)
    required = list(SAR_KEYS) + (["LIA"] if use_aux else [])
    found: Dict[str, Optional[Path]] = {k: None for k in required}

    for tif in event_path.glob("*.tif"):
        for key in required:
            if key in tif.name:
                found[key] = tif

    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing expected raster files for: {missing}")

    if use_aux:
        if ensure_slope_fn is not None:
            found["SLP"] = Path(ensure_slope_fn(event_path, found["preVH"]))
        else:
            slp = next((t for t in event_path.glob("*.tif") if "SLP" in t.name), None)
            if slp is None:
                raise FileNotFoundError("Missing expected raster files for: ['SLP']")
            found["SLP"] = slp

    return {k: v for k, v in found.items()}  # type: ignore[return-value]


def ordered_band_paths(paths: Dict[str, Path], use_aux: bool) -> List[Path]:
    """Return band paths in canonical channel order for stacking."""
    keys = list(SAR_KEYS) + (list(AUX_KEYS) if use_aux else [])
    return [paths[k] for k in keys]


def build_region_mask(sar_paths: Sequence[Path]) -> np.ndarray:
    """Build a filled region mask (uint8) from the four SAR bands.

    A pixel is inside the region if none of the four SAR bands is NaN. Holes are
    filled so partially-missing interior pixels are still considered in-region.
    ``sar_paths`` must be ordered [preVH, preVV, postVH, postVV].
    """
    bands = []
    for p in sar_paths[:4]:
        with rasterio.open(p) as src:
            bands.append(src.read(1))
    valid = np.ones_like(bands[0], dtype=bool)
    for b in bands:
        valid &= ~np.isnan(b)
    return binary_fill_holes(valid).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Sliding-window patch extraction
# --------------------------------------------------------------------------- #
def extract_patches_stack(
    raster_paths: Sequence[Path],
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    region_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], dict]:
    """Load bands (channels-first), optionally mask outside region as NaN, and
    return sliding windows plus grid/original shapes and the raster metadata.
    """
    with rasterio.open(raster_paths[0]) as src0:
        meta = src0.meta.copy()
        height, width = src0.shape

    raster_array = []
    for path in raster_paths:
        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)
            if src.nodata is not None:
                img = np.where(img == src.nodata, np.nan, img)
            if region_mask is not None:
                if region_mask.shape != img.shape:
                    raise ValueError(
                        f"region_mask shape {region_mask.shape} != raster {img.shape}"
                    )
                img = np.where(region_mask == 1, img, np.nan)
            raster_array.append(img)

    raster_array = np.stack(raster_array, axis=0)  # (C, H, W)
    num_bands = raster_array.shape[0]
    ph, pw = patch_size
    sh, sw = stride

    patches = view_as_windows(raster_array, (num_bands, ph, pw), (1, sh, sw))
    grid_shape = patches.shape[1:3]  # (rows, cols)
    return patches, grid_shape, (height, width), meta


# --------------------------------------------------------------------------- #
# Per-patch normalization dataset
# --------------------------------------------------------------------------- #
class TiledInferenceDataset(Dataset):
    """Yields normalized SAR (+aux) patches for tiled full-scene inference.

    Each item is a dict with keys ``pre`` (2,H,W), ``post`` (2,H,W),
    ``aux`` (2,H,W; zeros when aux disabled) and, when ``gt_patches`` is given,
    ``label`` (1,H,W).

    SAR channels are z-normalized with the training stats; non-finite pixels are
    filled with the per-channel sentinel z-value (matching the training
    dataset). Use :meth:`get_valid_indices` to skip mostly-empty patches.
    """

    def __init__(
        self,
        patches: np.ndarray,
        region_patches: np.ndarray,
        stats: Dict[str, torch.Tensor],
        use_aux: bool,
        nan_percent: float = 0.8,
        gt_patches: Optional[np.ndarray] = None,
    ) -> None:
        self.patches = patches                # (N, C, H, W)
        self.region_patches = region_patches  # (N, H, W)
        self.gt_patches = gt_patches          # (N, H, W) or None
        self.stats = stats
        self.use_aux = bool(use_aux)
        self.nan_percent = float(nan_percent)

        self.valid_indices: List[int] = []
        for idx in range(len(self.patches)):
            inside_region = np.all(self.region_patches[idx] == 1)
            sar_patch = self.patches[idx][:4]  # SAR only for validity
            nan_fraction = float(np.isnan(sar_patch).any(axis=0).mean())
            if inside_region or nan_fraction < self.nan_percent:
                self.valid_indices.append(idx)

    def __len__(self) -> int:
        return len(self.patches)

    def _fill_sentinel(self, z: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
        for c in range(z.shape[0]):
            z[c][~torch.isfinite(z[c])] = fill[c]
        return z

    def __getitem__(self, idx):
        raw = self.patches[idx].copy()  # copy so we never mutate the shared array

        # Sanitize SAR channels (first 4) before tensor conversion.
        sar = raw[:4]
        raw[:4] = np.where(
            (~np.isfinite(sar)) | (sar < SAR_MIN_DB) | (sar > SAR_MAX_DB),
            np.nan,
            sar,
        )

        patch = torch.tensor(raw, dtype=torch.float32)
        pre = patch[:2]
        post = patch[2:4]

        mean_img = self.stats["img_mean"].view(-1, 1, 1)
        std_img = self.stats["img_std"].view(-1, 1, 1)
        fill_img = self.stats["sentinel_z_img"].view(-1, 1, 1)

        pre_z = self._fill_sentinel((pre - mean_img) / std_img, fill_img)
        post_z = self._fill_sentinel((post - mean_img) / std_img, fill_img)

        if self.use_aux:
            aux = torch.stack([patch[4], patch[5]], dim=0)
            mean_aux = self.stats["aux_mean"].view(-1, 1, 1)
            std_aux = self.stats["aux_std"].view(-1, 1, 1)
            fill_aux = self.stats["sentinel_z_aux"].view(-1, 1, 1)
            aux_z = self._fill_sentinel((aux - mean_aux) / std_aux, fill_aux)
        else:
            # Placeholder with matching spatial dims (never fed to the model).
            aux_z = torch.zeros(2, pre_z.shape[1], pre_z.shape[2], dtype=torch.float32)

        item = {"pre": pre_z, "post": post_z, "aux": aux_z}
        if self.gt_patches is not None:
            gt = self.gt_patches[idx].astype(np.float32)
            item["label"] = (torch.tensor(gt) == 1).float().unsqueeze(0)  # (1,H,W)
        return item

    def get_valid_indices(self) -> List[int]:
        return self.valid_indices


# --------------------------------------------------------------------------- #
# Patch stitching / blending
# --------------------------------------------------------------------------- #
def merge_patches(
    patch_outputs: Sequence[np.ndarray],
    grid_shape: Tuple[int, int],
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    original_shape: Tuple[int, int],
    valid_indices: Sequence[int],
    mode: str = "center_crop",
) -> np.ndarray:
    """Stitch per-patch probability maps into a full-scene map.

    Supported modes:
        - ``none``:        adjacent-only stitching (writes the stride-sized crop).
        - ``center_crop``: crop half the overlap toward valid neighbors.
        - ``mean``:        average overlapping predictions.
        - ``gaussian``:    Gaussian-weighted average of overlaps.
        - ``max`` / ``min``: element-wise max/min over overlaps.

    Invalid (uncovered) pixels are set to ``NODATA`` (255).
    """
    patch_outputs = np.concatenate(patch_outputs, axis=0)  # (N_valid, ph, pw)
    ph, pw = patch_size
    rows, cols = grid_shape
    H, W = original_shape
    valid_set = set(int(i) for i in valid_indices)

    valid_grid = np.zeros((rows, cols), dtype=bool)
    for idx in valid_indices:
        r, c = divmod(idx, cols)
        valid_grid[r, c] = True

    # --- "none": adjacent-only stitching, no blending, no overlap ---
    if mode == "none":
        full = np.full((H, W), NODATA, dtype=np.float32)
        written = np.zeros((H, W), dtype=bool)
        v_idx = 0
        for patch_idx in range(rows * cols):
            i, j = divmod(patch_idx, cols)
            y0, x0 = i * stride[0], j * stride[1]
            if patch_idx in valid_set:
                patch = patch_outputs[v_idx]
                h_write = min(stride[0], H - y0)
                w_write = min(stride[1], W - x0)
                crop = patch[:h_write, :w_write]
                sub = full[y0:y0 + h_write, x0:x0 + w_write]
                unwritten = ~written[y0:y0 + h_write, x0:x0 + w_write]
                sub[unwritten] = crop[unwritten]
                full[y0:y0 + h_write, x0:x0 + w_write] = sub
                written[y0:y0 + h_write, x0:x0 + w_write] = True
                v_idx += 1
        return full

    if mode == "max":
        full = np.full((H, W), -np.inf, dtype=np.float32)
    elif mode == "min":
        full = np.full((H, W), np.inf, dtype=np.float32)
    else:
        full = np.zeros((H, W), dtype=np.float32)

    count_map = np.zeros((H, W), dtype=np.float32)

    if mode == "gaussian":
        yy = np.linspace(-1, 1, ph)
        xx = np.linspace(-1, 1, pw)
        xv, yv = np.meshgrid(xx, yy)
        weights = np.exp(-(xv ** 2 + yv ** 2) / 0.5).astype(np.float32)
    else:
        weights = None

    # Canonical center-crop split: half the overlap toward top/left, remainder
    # toward bottom/right (handles odd overlaps and non-default strides).
    overlap_y = max(0, ph - stride[0])
    overlap_x = max(0, pw - stride[1])
    half_y, half_x = overlap_y // 2, overlap_x // 2
    rem_y, rem_x = overlap_y - half_y, overlap_x - half_x

    v_idx = 0
    for patch_idx in range(rows * cols):
        i, j = divmod(patch_idx, cols)
        y0, x0 = i * stride[0], j * stride[1]
        if patch_idx not in valid_set:
            continue
        patch = patch_outputs[v_idx]
        v_idx += 1

        if mode == "center_crop":
            top_ok = (i > 0) and valid_grid[i - 1, j]
            bottom_ok = (i < rows - 1) and valid_grid[i + 1, j]
            left_ok = (j > 0) and valid_grid[i, j - 1]
            right_ok = (j < cols - 1) and valid_grid[i, j + 1]

            tc = half_y if top_ok else 0
            bc = rem_y if bottom_ok else 0
            lc = half_x if left_ok else 0
            rc = rem_x if right_ok else 0

            cropped = patch[tc: ph - bc, lc: pw - rc]
            full[y0 + tc: y0 + ph - bc, x0 + lc: x0 + pw - rc] = cropped
            count_map[y0 + tc: y0 + ph - bc, x0 + lc: x0 + pw - rc] += 1

        elif mode == "gaussian":
            full[y0:y0 + ph, x0:x0 + pw] += patch * weights
            count_map[y0:y0 + ph, x0:x0 + pw] += weights

        elif mode == "max":
            region = full[y0:y0 + ph, x0:x0 + pw]
            full[y0:y0 + ph, x0:x0 + pw] = np.maximum(region, patch)
            count_map[y0:y0 + ph, x0:x0 + pw] += 1

        elif mode == "min":
            region = full[y0:y0 + ph, x0:x0 + pw]
            unseen = count_map[y0:y0 + ph, x0:x0 + pw] == 0
            region[unseen] = patch[unseen]
            full[y0:y0 + ph, x0:x0 + pw] = np.minimum(region, patch)
            count_map[y0:y0 + ph, x0:x0 + pw] += 1

        else:  # mean
            full[y0:y0 + ph, x0:x0 + pw] += patch
            count_map[y0:y0 + ph, x0:x0 + pw] += 1

    covered = count_map > 0
    if mode in ("mean", "gaussian"):
        full[covered] /= count_map[covered]
    full[~covered] = NODATA
    return full
