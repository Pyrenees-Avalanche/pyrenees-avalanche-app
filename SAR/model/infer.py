import argparse
import re
import timeit
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from shapely.geometry import shape
from skimage.util.shape import view_as_windows
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.build import build_model
from utils.morph import morph_close
from utils.runtime import get_device
from utils.slope import ensure_slope
from utils.tiling import (
    NODATA,
    TiledInferenceDataset,
    build_region_mask,
    extract_patches_stack,
    find_raster_paths,
    merge_patches,
    ordered_band_paths,
)

torch.set_num_threads(8)  # intra-op: threads inside a single op
torch.set_num_interop_threads(4)  # inter-op: parallelism across ops

parser = argparse.ArgumentParser(description="Run inference on a single event with selectable blending strategy.")
parser.add_argument("--event-path", type=str, required=True, help="Path to the event folder")
parser.add_argument("--model-ckpt", type=str, default="exp/swinunet_128/best_model.pth", help="Path to the model checkpoint")
parser.add_argument("--output-dir", type=str, default="output/infer", help="Directory to save output predictions")
parser.add_argument("--patch-size", type=int, default=128)
parser.add_argument("--stride", type=int, default=64)
parser.add_argument("--blending", type=str, choices=["mean", "max", "min", "gaussian", "center_crop"], default="center_crop")
parser.add_argument("--use-aux", action="store_true", help="Enable Auxiliary data usage.")
parser.add_argument("--force", action="store_true", help="Recompute even if outputs already exist.")

args = parser.parse_args()

PATCH_SIZE = (args.patch_size, args.patch_size)
STRIDE = (args.stride, args.stride)
USE_AUX = args.use_aux
event_path = Path(args.event_path)
model_ckpt = Path(args.model_ckpt)


def is_yyyymmdd(s: str) -> bool:
    if not re.fullmatch(r"\d{8}", s):
        return False
    try:
        datetime.strptime(s, "%Y%m%d")
        return True
    except ValueError:
        return False


area = event_path.parts[-2] if len(event_path.parts) >= 2 else "unknown"
date_candidate = event_path.name  # last path part

if is_yyyymmdd(date_candidate):
    output_dir = Path(args.output_dir) / area / date_candidate
else:
    output_dir = Path(args.output_dir) / area

output_dir.mkdir(parents=True, exist_ok=True)


event_name = event_path.name
out_path = output_dir / f"{event_name}.tif"
output_gpkg_path = output_dir / f"{event_name}.gpkg"

if not args.force:
    tif_ok = out_path.exists() and out_path.stat().st_size > 0
    gpkg_ok = output_gpkg_path.exists() and output_gpkg_path.stat().st_size > 0
    if tif_ok and gpkg_ok:
        print(f"Outputs already exist: {out_path} and {output_gpkg_path}. Skipping. "
              f"(Use --force to recompute)")
        exit()

# Load model
device = get_device()
model = build_model("swinunet", patch_size=args.patch_size, use_aux=USE_AUX, device=device)
checkpoint = torch.load(model_ckpt, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
best_thr = checkpoint.get("best_threshold", 0.5)

# Locate rasters (creating the slope raster on demand when using aux)
paths = find_raster_paths(event_path, USE_AUX, ensure_slope_fn=ensure_slope)
raster_paths = ordered_band_paths(paths, USE_AUX)

# Build region mask from SAR valid pixels
region_mask = build_region_mask(raster_paths[:4])

# Extract patches
stacked_patches, grid_shape, original_shape, meta = extract_patches_stack(raster_paths, PATCH_SIZE, STRIDE, region_mask)
patches = stacked_patches.reshape(-1, *stacked_patches.shape[3:])
region_patches = view_as_windows(region_mask, PATCH_SIZE, STRIDE).reshape(-1, *PATCH_SIZE)

# Load stats saved with the checkpoint
stats_tensor = checkpoint["norm_stats"]

# Inference
dataset = TiledInferenceDataset(patches, region_patches, stats_tensor, USE_AUX)
loader = DataLoader(Subset(dataset, dataset.get_valid_indices()), batch_size=32, shuffle=False)

patch_outputs = []

start = timeit.default_timer()

with torch.no_grad():
    for batch in tqdm(loader, desc="Inferencing", ncols=100):
        pre_patch = batch["pre"].to(device)
        post_patch = batch["post"].to(device)
        if not USE_AUX:
            logits = model(pre_patch, post_patch)
        else:
            aux_patch = batch["aux"].to(device)
            logits = model(pre_patch, post_patch, aux_patch)
        preds = torch.sigmoid(logits)
        patch_outputs.append(preds.cpu().numpy().squeeze(1))

# Stitch and save output
reconstructed = merge_patches(
    patch_outputs, grid_shape, PATCH_SIZE, STRIDE,
    original_shape, dataset.get_valid_indices(), mode=args.blending
)

# Build valid mask (exclude nodata)
valid = reconstructed != NODATA

# ---- Threshold then morphological closing on the full map ----
# Set nodata to 0 so they don't create false dilations
prob_2d = reconstructed.copy().astype(np.float32)
prob_2d[~valid] = 0.0



# Save probability map before applying the threshold
prob_path = output_dir / f"{event_name}_prob.tif"

prob_meta = meta.copy()
prob_meta.update(
    driver="GTiff",
    count=1,
    dtype="float32",
    nodata=-9999.0,
    compress="LZW",
)

prob_save = prob_2d.astype(np.float32).copy()
prob_save[~valid] = -9999.0

with rasterio.open(prob_path, "w", **prob_meta) as dst:
    dst.write(prob_save, 1)

print(f"Probability raster saved to: {prob_path}")


# Binary map
binary = (prob_2d > best_thr).astype(np.uint8)

# Morphological closing (fill small holes / bridge small gaps)
# morph_close works on numpy (H, W) arrays
binary_closed = morph_close(binary, kernel_size=3, iterations=1).astype(np.uint8)

duration = timeit.default_timer() - start

print(f"Inference + reconstruction time: {duration:.2f} seconds")

# Restore nodata outside valid area
binary_mask = np.full(binary_closed.shape, NODATA, dtype=np.uint8)
binary_mask[valid] = binary_closed[valid]

meta.update(
    driver="GTiff",
    count=1,
    dtype="uint8",
    nodata=NODATA,
    tiled=True,
    blockxsize=256,
    blockysize=256,
    compress="LZW",
    BIGTIFF="IF_SAFER",
    NUM_THREADS="ALL_CPUS",
)
with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(binary_mask, 1)

print(f"Saved output to: {out_path}")

# Extract polygons from raster
polygons = []
with rasterio.open(out_path) as src:
    image = src.read(1)
    mask = image != NODATA  # Ignore nodata
    transform = src.transform

    for geom, val in shapes(image, mask=mask, transform=transform, connectivity=8):
        if val == 1:  # Keep only predicted class
            polygons.append(shape(geom))

    crs = src.crs  # Coordinate reference system

gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
gdf.to_file(output_gpkg_path, driver="GPKG")

print(f"Polygons saved to: {output_gpkg_path}")