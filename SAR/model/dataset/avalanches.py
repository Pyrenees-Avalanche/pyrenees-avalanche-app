import json
import hashlib
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import random

import numpy as np
import torch
import rasterio
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


PRE_FILENAME = "pre.tif"
POST_FILENAME = "post.tif"
DEM_FILENAME = "dem.tif"
SLOPE_FILENAME = "slope.tif"
ASPECT_FILENAME = "aspect.tif"
LIA_FILENAME = "lia.tif"
MASK_FILENAME = "mask.tif"

# Manifest cache config
MANIFEST_DIRNAME = ".cache"
MANIFEST_VERSION = 1
REQUIRED_KEYS = ["pre", "post", "dem", "slope", "aspect", "lia", "mask"]


class AvalancheDataset(Dataset):
    def __init__(self, patches_dir, events, apply_transform=True, use_manifest_cache=True, gdal_cache_mb=None):
        """
        Args:
            patches_dir (str|Path): Root directory.
            events (list[str]): ["Livigno", "Tajikistan", ...]
            apply_transform (bool): Enable geometric/radiometric augs.
            use_manifest_cache (bool): Use manifest cache for folder discovery.
            gdal_cache_mb (int|None): If set, increases GDAL block cache (MB).
        """
        self.patches_dir = Path(patches_dir)
        self.apply_transform = apply_transform
        self.use_manifest_cache = use_manifest_cache
        self._events = list(events)

        # Optional: bump GDAL cache (helps repeated IO)
        # Can also be set globally via environment.
        self._rio_env = None
        if gdal_cache_mb is not None:
            self._rio_env = rasterio.Env(GDAL_CACHEMAX=int(gdal_cache_mb))
            self._rio_env.__enter__()

        # Discover patches via manifest cache
        self.folders = self._load_or_build_manifest(self._events)

        # Dataset statistics (cached to stats.json)
        self.stats_path = self.patches_dir / "stats.json"
        self.stats = self._load_or_compute_stats()

    # ---------------------- PyTorch protocol ----------------------
    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        paths = self.folders[idx]
        pre = self._read_raster(paths["pre"])
        post = self._read_raster(paths["post"])
        lia = self._read_raster(paths["lia"], single_band=True)
        dem = self._read_raster(paths["dem"], single_band=True)
        slope_deg = self._read_raster(paths["slope"], single_band=True)
        aspect_deg = self._read_raster(paths["aspect"], single_band=True)
        mask = self._read_raster(paths["mask"], single_band=True)

        # Normalize / engineer features
        lia = lia / 180.0
        # relative DEM
        # dem = dem - np.nanmin(dem)
        slope = slope_deg / 90.0
        aspect_sin, aspect_cos = self._aspect_to_vec(aspect_deg, slope_deg, undef_slope_deg=1.0)
        aux = np.stack([dem, slope, aspect_sin, aspect_cos], axis=0)  # (4, H, W)

        # To tensors
        pre = torch.tensor(pre, dtype=torch.float32)
        post = torch.tensor(post, dtype=torch.float32)
        aux = torch.tensor(aux, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)  # (1,H,W)

        # Geometric aug
        if self.apply_transform:
            pre, post, aux, mask = self._geom_aug(pre, post, aux, mask)

        # Z-normalize SAR with cached stats
        mean_img = self.stats["img_mean"].view(-1, 1, 1)     # (2,1,1)
        std_img = self.stats["img_std"].view(-1, 1, 1)       # (2,1,1)
        fill_value_img = self.stats["sentinel_z_img"].view(-1, 1, 1)

        pre = (pre - mean_img) / std_img
        post = (post - mean_img) / std_img

        # Replace non-finite with per-channel sentinel Z
        for c in range(pre.shape[0]):
            pre_c = pre[c]
            post_c = post[c]
            pre_c[~torch.isfinite(pre_c)] = fill_value_img[c].item()
            post_c[~torch.isfinite(post_c)] = fill_value_img[c].item()

        # Radiometric aug
        if self.apply_transform:
            pre, post, aux = self._radiom_aug_z(pre, post, aux)

        return {
            "pre": pre,    # (2,H,W)
            "post": post,  # (2,H,W)
            "aux": aux,    # (4,H,W)
            "mask": mask,  # (1,H,W)
            "event": paths["event"],
        }

    # ---------------------- Raster IO (with LRU) ----------------------
    @staticmethod
    @lru_cache(maxsize=256)  # tune per RAM; cache is per-process/worker
    def _read_raster_cached(path_str, single_band_bool):
        p = Path(path_str)
        with rasterio.open(p) as src:
            img = src.read().astype("float32")  # (B,H,W)
            if src.nodata is not None:
                img = np.where(img == src.nodata, np.nan, img)

            if single_band_bool:
                img = img[0]  # (H,W)
            else:
                # Apply SAR range mask only for multi-band SAR rasters
                img = np.where((~np.isfinite(img)) | (img < -40.0) | (img > 20.0), np.nan, img)

        return img

    def _read_raster(self, raster_path, single_band=False):
        # Copy to make it writable for aug later (LRU returns shared arrays)
        arr = self._read_raster_cached(str(raster_path), bool(single_band))
        return arr.copy()

    # ---------------------- Geometry / Radio augs ----------------------
    def _geom_aug(self, pre, post, aux, mask):
        # 8-way rigid pose
        if random.random() < 0.5:
            pre, post, aux, mask = map(TF.hflip, (pre, post, aux, mask))

        k = random.randint(0, 3)
        if k:
            dims = (1, 2)
            pre = torch.rot90(pre, k, dims)
            post = torch.rot90(post, k, dims)
            aux = torch.rot90(aux, k, dims)
            mask = torch.rot90(mask, k, dims)

        # mild affine jitter
        if random.random() < 0.5:
            ang = random.uniform(-7, 7)
            trans = [random.uniform(-2, 2), random.uniform(-2, 2)]
            scl = random.uniform(0.95, 1.05)
            shr = random.uniform(-3, 3)
            kw_img = dict(angle=ang, translate=trans, scale=scl, shear=shr,
                          interpolation=InterpolationMode.BILINEAR)
            kw_mask = dict(angle=ang, translate=trans, scale=scl, shear=shr,
                           interpolation=InterpolationMode.NEAREST)
            pre = TF.affine(pre, **kw_img)
            post = TF.affine(post, **kw_img)
            aux = TF.affine(aux, **kw_img)
            mask = TF.affine(mask, **kw_mask)
        return pre, post, aux, mask

    def _radiom_aug_z(self, pre_z, post_z, aux_z):
        # Gaussian noise
        if random.random() < 0.5:
            pre_z += torch.randn_like(pre_z) * 0.05
            post_z += torch.randn_like(post_z) * 0.05

        # Random intensity scaling
        if random.random() < 0.5:
            gain = 1.0 + random.uniform(-0.03, 0.03)
            pre_z *= gain
            post_z *= gain

        # AUX small noise
        if random.random() < 0.5:
            aux_z += torch.randn_like(aux_z) * 0.02

        # AUX slight scale/bias
        if random.random() < 0.5:
            scale = 1.0 + random.uniform(-0.05, 0.05)
            bias = random.uniform(-0.05, 0.05)
            aux_z *= scale
            aux_z += bias

        return pre_z, post_z, aux_z

    # ---------------------- Feature helpers ----------------------
    def _aspect_to_vec(self, aspect_deg: np.ndarray, slope_deg: np.ndarray,
                       undef_slope_deg: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        # undef: aspect/slope invalid or slope too low
        undef = (~np.isfinite(aspect_deg)) | (~np.isfinite(slope_deg)) | (slope_deg < undef_slope_deg)
        theta = np.deg2rad(aspect_deg)
        s = np.sin(theta).astype(np.float32)
        c = np.cos(theta).astype(np.float32)
        s[undef] = 0.0
        c[undef] = 0.0
        return s, c

    # ---------------------- Manifest cache ----------------------
    def _manifest_path(self, events) -> Path:
        cache_dir = self.patches_dir / MANIFEST_DIRNAME
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.md5(("|".join(sorted(map(str, events)))).encode("utf-8")).hexdigest()
        return cache_dir / f"manifest_v{MANIFEST_VERSION}_{key}.json"

    def _collect_folders(self, events):
        folders = []
        for event in tqdm(events, desc="Reading events"):
            event_dir = self.patches_dir / event
            if not event_dir.exists():
                continue
            patch_folders = sorted([p for p in event_dir.iterdir() if p.is_dir()])
            for patch_folder in tqdm(patch_folders, desc=f"Reading patches for {event}", leave=False):
                rec = {
                    "event": event,
                    "pre":   patch_folder / PRE_FILENAME,
                    "post":  patch_folder / POST_FILENAME,
                    "dem":   patch_folder / DEM_FILENAME,
                    "slope": patch_folder / SLOPE_FILENAME,
                    "aspect":patch_folder / ASPECT_FILENAME,
                    "lia":   patch_folder / LIA_FILENAME,
                    "mask":  patch_folder / MASK_FILENAME,
                }
                if all(Path(rec[k]).exists() for k in REQUIRED_KEYS):
                    folders.append(rec)
        return folders

    def _snapshot(self, folders):
        """JSON-serializable snapshot with absolute paths + mtimes."""
        out = []
        for rec in folders:
            mtimes = {k: Path(rec[k]).stat().st_mtime for k in REQUIRED_KEYS}
            out.append({
                "event": rec["event"],
                **{k: str(Path(rec[k]).resolve()) for k in REQUIRED_KEYS},
                "_mtimes": mtimes,
            })
        return out

    def _is_manifest_valid(self, data):
        try:
            if data.get("version") != MANIFEST_VERSION:
                return False
            for rec in data["folders"]:
                for k in REQUIRED_KEYS:
                    p = Path(rec[k])
                    if not p.exists():
                        return False
                    if abs(p.stat().st_mtime - float(rec["_mtimes"][k])) > 1e-6:
                        return False
            return True
        except Exception:
            return False

    def _load_or_build_manifest(self, events):
        if self.use_manifest_cache:
            mp = self._manifest_path(events)
            if mp.exists():
                with open(mp, "r") as f:
                    manifest = json.load(f)
                if self._is_manifest_valid(manifest):
                    return [{
                        "event": rec["event"],
                        **{k: Path(rec[k]) for k in REQUIRED_KEYS}
                    } for rec in manifest["folders"]]

        # build fresh
        folders = self._collect_folders(events)
        if self.use_manifest_cache:
            mp = self._manifest_path(events)
            payload = {
                "version": MANIFEST_VERSION,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "events": sorted(map(str, events)),
                "folders": self._snapshot(folders),
            }
            with open(mp, "w") as f:
                json.dump(payload, f, indent=2)
        return folders

    # ---------------------- Stats (cached to stats.json) ----------------------
    def _compute_stats(self):
        """
        Scan whole dataset for per-channel SAR mean/std while ignoring NaNs.
        """
        num_sar_channels = 2  # VV, VH
        sar_sum = torch.zeros(num_sar_channels)
        sar_sq = torch.zeros(num_sar_channels)
        sar_cnt = torch.zeros(num_sar_channels)

        for paths in tqdm(self.folders, desc="Computing mean/std"):
            pre = torch.tensor(self._read_raster(paths["pre"]), dtype=torch.float32)
            post = torch.tensor(self._read_raster(paths["post"]), dtype=torch.float32)

            for c in range(num_sar_channels):
                patch = torch.cat([pre[c], post[c]], 0)  # (2H,W)
                sar_sum[c] += torch.nansum(patch)
                sar_sq[c] += torch.nansum(patch ** 2)
                sar_cnt[c] += torch.sum(torch.isfinite(patch))

        sar_mean = torch.where(sar_cnt > 0, sar_sum / sar_cnt, torch.zeros_like(sar_sum))
        sar_var = torch.where(sar_cnt > 0, sar_sq / sar_cnt - sar_mean ** 2, torch.ones_like(sar_sum))
        sar_std = sar_var.clamp(min=1e-9).sqrt()

        min_valid_db = -50.0
        sentinel_z_img = (min_valid_db - sar_mean) / sar_std  # (2,)

        # keep also as attributes for convenience
        self.sentinel_z_img = sentinel_z_img

        return {
            "img_mean": sar_mean,
            "img_std": sar_std,
            "sentinel_z_img": sentinel_z_img,
        }

    def _load_or_compute_stats(self):
        if self.stats_path.exists():
            print(f"Loading existing dataset statistics from {self.stats_path}")
            with open(self.stats_path, "r") as f:
                raw = json.load(f)
            return {
                "img_mean": torch.tensor(raw["img_mean"]),
                "img_std": torch.tensor(raw["img_std"]),
                "sentinel_z_img": torch.tensor(raw["sentinel_z_img"]),
            }
        else:
            print("No existing statistics found. Computing stats from dataset...")
            stats = self._compute_stats()
            to_save = {
                "img_mean": stats["img_mean"].tolist(),
                "img_std": stats["img_std"].tolist(),
                "sentinel_z_img": stats["sentinel_z_img"].tolist(),
            }
            with open(self.stats_path, "w") as f:
                json.dump(to_save, f, indent=2)
            return stats

    # ---------------------- Cleanup ----------------------
    def __del__(self):
        # Close rasterio.Env if we opened one
        if self._rio_env is not None:
            try:
                self._rio_env.__exit__(None, None, None)
            except Exception:
                pass