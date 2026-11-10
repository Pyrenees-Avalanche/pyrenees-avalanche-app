"""Slope raster generation from the public Copernicus DEM (30 m).

Ensures a ``*_SLP.tif`` slope raster exists for an event by downloading the
required Copernicus DEM tiles, mosaicking, reprojecting onto the reference SAR
grid, and running ``gdaldem slope``.

Imported by ``infer.py`` (via ``from utils.slope import ensure_slope``).
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import boto3
import rasterio
from botocore import UNSIGNED
from botocore.config import Config
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject

# Public Copernicus DEM (30 m) bucket
S3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
COP30_BUCKET = "copernicus-dem-30m"


def _cop30_tile_key(lat_deg: int, lon_deg: int) -> str:
    """
    Return S3 key for Copernicus DEM 30m COG tile covering the
    integer degree cell (lat_deg, lon_deg).
    Bucket paths use '.../Copernicus_DSM_COG_30_Nxx_00_Exxx_00_DEM/...'
    """
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_"
        f"{ew}{abs(lon_deg):03d}_00_DEM/"
        f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_"
        f"{ew}{abs(lon_deg):03d}_00_DEM.tif"
    )


def _download_cop30_tile(lat_deg: int, lon_deg: int, out_path: Path) -> bool:
    key = _cop30_tile_key(lat_deg, lon_deg)
    try:
        S3.download_file(COP30_BUCKET, key, str(out_path))
        return True
    except Exception as e:
        print(f"[DEM] Could not download {key}: {e}")
        return False


def ensure_slope(event_dir: Path, ref_raster: Path) -> Path:
    """
    Ensure *_SLP.tif exists in event_dir.
    If missing, download COP30 DEM tiles, merge, reproject to ref_raster grid,
    then run gdaldem slope to create the slope raster.
    Returns the Path to the slope raster.
    """
    prefix = ref_raster.name.split("_")[0]
    # Infer expected filenames
    # Adjust this if your naming differs
    slope_path = event_dir / f"{prefix}_SLP.tif"
    dem_mosaic_path = event_dir / f"{prefix}_DEM_mosaic.tif"
    dem_aligned_path = event_dir / f"{prefix}_DEM_aligned.tif"

    if slope_path.exists():
        # print(f"[DEM] Using existing slope: {slope_path.name}")
        return slope_path

    # 1) Determine geographic bounds from the reference raster (e.g., preVH)
    with rasterio.open(ref_raster) as ref:
        ref_bounds = ref.bounds
        ref_crs = ref.crs
        # transform to WGS84 for tile selection
        to_wgs84 = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
        minx, miny = to_wgs84.transform(ref_bounds.left, ref_bounds.bottom)
        maxx, maxy = to_wgs84.transform(ref_bounds.right, ref_bounds.top)

    # 2) Work out which integer-degree tiles we need
    lons = range(math.floor(minx), math.ceil(maxx))
    lats = range(math.floor(miny), math.ceil(maxy))

    # 3) Download tiles
    downloaded = []
    for lat in lats:
        for lon in lons:
            local_path = event_dir / Path(_cop30_tile_key(lat, lon)).name
            if local_path.exists():
                downloaded.append(local_path)
                continue
            ok = _download_cop30_tile(lat, lon, local_path)
            if ok:
                downloaded.append(local_path)

    if not downloaded:
        raise RuntimeError("[DEM] No Copernicus DEM tiles could be downloaded for the AOI.")

    # 4) Mosaic tiles
    srcs = [rasterio.open(str(p)) for p in downloaded]
    mosaic, mosaic_transform = merge(srcs)
    mosaic_meta = srcs[0].meta.copy()
    for s in srcs:
        s.close()
    mosaic_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": mosaic_transform,
        "count": 1,
    })
    with rasterio.open(dem_mosaic_path, "w", **mosaic_meta) as dst:
        dst.write(mosaic)

    # 5) Reproject DEM mosaic to match the reference raster grid (CRS, res, extent)
    with rasterio.open(ref_raster) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        dst_meta = ref.meta.copy()

    dst_meta.update({
        "driver": "GTiff",
        "dtype": mosaic_meta["dtype"],
        "count": 1,
        "crs": dst_crs,
        "transform": dst_transform,
        "width": dst_width,
        "height": dst_height,
    })

    with rasterio.open(dem_aligned_path, "w", **dst_meta) as dst:
        with rasterio.open(dem_mosaic_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

    # 6) Compute slope with gdaldem
    # (slope in degrees; if you prefer percent, add '-p')
    try:
        subprocess.run(
            [
                "gdaldem", "slope",
                str(dem_aligned_path),
                str(slope_path),
                "-of", "GTiff",
                "-s", "1.0",
                "-compute_edges",
            ],
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "gdaldem not found. Please install GDAL (e.g., apt-get install gdal-bin) "
            "or adjust the code to compute slope in Python."
        )

    print(f"[DEM] Created slope: {slope_path.name}")

    # --- Cleanup: remove temporary DEMs and downloaded tiles ---
    for tmp in [dem_mosaic_path, dem_aligned_path]:
        try:
            if tmp.exists():
                tmp.unlink()
                print(f"[DEM] Removed temporary file: {tmp.name}")
        except Exception as e:
            print(f"[DEM] Failed to remove {tmp.name}: {e}")

    # Also remove downloaded source tiles to save space
    for tile_path in downloaded:
        try:
            if tile_path.exists():
                tile_path.unlink()
        except Exception as e:
            print(f"[DEM] Failed to remove tile {tile_path.name}: {e}")

    return slope_path
