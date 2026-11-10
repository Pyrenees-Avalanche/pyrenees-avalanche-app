"""Download small, aligned Sentinel-1 PRE/POST SAR rasters for selected avalanches.

Input:  sentinel_pairs.json (created by find_sentinel_pairs.py)
Output: sentinel_rois/<codiallau>/{pre.tif, post.tif, manifest.json,
                                     pre_userdata.json, post_userdata.json}

Uses Copernicus Data Space Sentinel Hub Process API. Requires OAuth client
credentials in CDSE_CLIENT_ID and CDSE_CLIENT_SECRET environment variables.
No full SAFE products are downloaded. One event is processed at a time.

This variant approximates the original SNAP chain using GAMMA0_TERRAIN,
Copernicus 30m DEM, Lee filtering (not SNAP Refined Lee), and bilinear
resampling. VV/VH are saved in VH,VV order, followed by a valid-data mask.
Output may be dB or linear, controlled by OUTPUT_SCALE. This is not an
exact reproduction of SNAP, and model-specific normalization is still required.
"""

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import rasterio
import requests

from pyproj import Transformer

load_dotenv()

# Config
PAIRS_FILE = Path(r".\SAR\icgc\sentinel_pairs.json")
OUT_ROOT = Path(r".\SAR\icgc\sentinel_rois")
TARGET_SUCCESSFUL_PAIRS = 100         # None attempts every input pair
MAX_PAIRS = None                # Optional cap on how many pairs to try
PIXEL_SIZE_M = 10
MAX_IMAGE_SIDE_PIXELS = 2500         # Process API per-dimension bound
MIN_VALID_FRACTION = 0.98          # From returned dataMask, each image
TIME_TOLERANCE_SECONDS = 2         # Narrow around chosen acquisition timestamp
SKIP_COMPLETED = True
SLEEP_BETWEEN_EVENTS_SECONDS = 0.3

# Baseline processing only
BACKSCATTER_COEFF = "GAMMA0_TERRAIN"
DEM_INSTANCE = "COPERNICUS_30"
SPECKLE_FILTER = "LEE"         
LEE_WINDOW = 5
UPSAMPLING = "BILINEAR"
OUTPUT_SCALE = "DB"
RADIOMETRIC_TERRAIN_OVERSAMPLING = 2             
DB_FLOOR_LINEAR = 1e-10              # Avoid log10(0); masked pixels remain excluded

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
MAX_ARCHIVE_BYTES = 180 * 1024 * 1024


def make_evalscript():
    """Return VH,VV,dataMask (training channel order). Preserve float32."""
    if OUTPUT_SCALE not in ("DB", "LINEAR"):
        raise ValueError("OUTPUT_SCALE must be 'DB' or 'LINEAR'")
    if OUTPUT_SCALE == "DB":
        conversion = f"""
  var vh = 10.0 * Math.log(Math.max(s.VH, {DB_FLOOR_LINEAR})) / Math.LN10;
  var vv = 10.0 * Math.log(Math.max(s.VV, {DB_FLOOR_LINEAR})) / Math.LN10;"""
    else:
        conversion = """
  var vh = s.VH;
  var vv = s.VV;"""
    return """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH", "dataMask"]}],
    mosaicking: Mosaicking.ORBIT,
    output: {id: "default", bands: 3, sampleType: SampleType.FLOAT32}
  };
}

function evaluatePixel(samples) {
  for (var i = 0; i < samples.length; i++) {
    var s = samples[i];

    // Require valid source data and finite backscatter
    if (
      !s ||
      s.dataMask !== 1 ||
      !isFinite(s.VH) ||
      !isFinite(s.VV) ||
      s.VH <= 0 ||
      s.VV <= 0
    ) {
      continue;
    }

    // Apply the configured conversion
    __CONVERSION__

    if (!isFinite(vh) || !isFinite(vv)) {
      continue;
    }

    return [vh, vv, 1.0];
  }

  // No valid SAR measurement
  return [0.0, 0.0, 0.0];
}

function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  var orbits = scenes.orbits || [];
  var tiles = [];
  for (var i = 0; i < orbits.length; i++) {
    var these = orbits[i].tiles || [];
    for (var j = 0; j < these.length; j++) {
      tiles.push({
        sentinel1ProductId: these[j].sentinel1ProductId || null,
        date: these[j].date || null
      });
    }
  }
  outputMetadata.userData = {orbitCount: orbits.length, tiles: tiles};
}
""".replace("__CONVERSION__", conversion)



def parse_utc(text):
    timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError(f"Acquisition timestamp lacks timezone: {text}")
    return timestamp.astimezone(timezone.utc)


def iso_z(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def acquisition_key(product_id):
    """Key shared by a raw S1 GRD and COG product for the SAME acquisition.

    A STAC COG can have a different last unique-product identifier than the
    non-COG GRD used by Sentinel Hub. Compare platform/mode/type/polarizations/
    start/end times/absolute orbit/datatake, but not the final product UID.
    """
    name = str(product_id).removesuffix(".SAFE")
    fields = name.split("_")
    if len(fields) < 9 or not re.fullmatch(r"S1[A-D]", fields[0]):
        raise ValueError(f"Unsupported Sentinel-1 product identifier: {product_id}")
    if not re.fullmatch(r"\d{8}T\d{6}", fields[4]):
        raise ValueError(f"Unexpected Sentinel-1 product name: {product_id}")
    return "_".join(fields[:8])


def dimensions(roi_bbox):
    if len(roi_bbox) != 4:
        raise ValueError("Expected 4 ROI bounds")
    minx, miny, maxx, maxy = map(float, roi_bbox)
    if not (maxx > minx and maxy > miny):
        raise ValueError(f"Invalid ROI: {roi_bbox}")
    width_f = (maxx - minx) / PIXEL_SIZE_M
    height_f = (maxy - miny) / PIXEL_SIZE_M
    width, height = round(width_f), round(height_f)
    if abs(width - width_f) > 1e-5 or abs(height - height_f) > 1e-5:
        raise ValueError("ROI is not aligned to a 10-meter grid")
    if min(width, height) <= 0 or max(width, height) > MAX_IMAGE_SIDE_PIXELS:
        raise ValueError(f"ROI too large for single request: {width}x{height} pixels")
    return width, height



def request_roi_bbox(pair):
    """Convert the ROI to EPSG:32631 and align to 10 m."""

    bbox = pair["roi_bbox"]
    source_crs = pair["roi_crs"]

    if source_crs == "EPSG:25831":
        transformer = Transformer.from_crs(
            25831, 32631, always_xy=True
        )
        bbox = transformer.transform_bounds(*bbox)

    elif source_crs != "EPSG:32631":
        raise ValueError(f"Unexpected CRS: {source_crs}")

    minx, miny, maxx, maxy = bbox
    r = PIXEL_SIZE_M

    minx = math.floor(minx / r) * r
    miny = math.floor(miny / r) * r
    maxx = math.ceil(maxx / r) * r
    maxy = math.ceil(maxy / r) * r

    def align_pixels(metres):
        pixels = round(metres / r)
        return 128 + max(
            0, math.ceil((pixels - 128) / 64)
        ) * 64

    width = align_pixels(maxx - minx)
    height = align_pixels(maxy - miny)

    return [
        minx,
        miny,
        minx + width * r,
        miny + height * r,
    ]


def request_body(pair, which):
    entry = pair[which]
    stamp = parse_utc(entry["datetime"])
    start = stamp - timedelta(seconds=TIME_TOLERANCE_SECONDS)
    end = stamp + timedelta(seconds=TIME_TOLERANCE_SECONDS)
    bbox_32631 = request_roi_bbox(pair)
    width, height = dimensions(bbox_32631)
    orbit_direction = str(entry["orbit_state"]).upper()
    if orbit_direction not in ("ASCENDING", "DESCENDING"):
        raise ValueError(f"Unknown orbit direction: {orbit_direction}")
    # `id` is NOT a product-id filter in Process API. Limit time and
    # direction, then check reported source product(s) in userdata.json.
    processing = {
        "backCoeff": BACKSCATTER_COEFF,
        "orthorectify": True,
        "demInstance": DEM_INSTANCE,
        "upsampling": UPSAMPLING,
        "radiometricTerrainOversampling": RADIOMETRIC_TERRAIN_OVERSAMPLING,
        "speckleFilter": {"type": SPECKLE_FILTER},
    }
    if SPECKLE_FILTER == "LEE":
        processing["speckleFilter"].update(
            windowSizeX=LEE_WINDOW, windowSizeY=LEE_WINDOW
        )
    return {
        "input": {
            "bounds": {
                "bbox": bbox_32631,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/32631"
                },
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": iso_z(start), "to": iso_z(end)},
                    "acquisitionMode": "IW",
                    "polarization": "DV",
                    "resolution": "HIGH",
                    "orbitDirection": orbit_direction,
                },
                "processing": processing,
            }],
        },
        "output": {
            "width": width, "height": height,
            "responses": [
                {"identifier": "default", "format": {"type": "image/tiff"}},
                {"identifier": "userdata", "format": {"type": "application/json"}},
            ],
        },
        "evalscript": make_evalscript(),
    }


def request_signature(pair, which):
    payload = request_body(pair, which)
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class SentinelHubClient:
    def __init__(self):
        self.client_id = os.getenv("CDSE_CLIENT_ID", default=os.environ.get("CDSE_CLIENT_ID"))
        self.client_secret = os.getenv("CDSE_CLIENT_SECRET", default=os.environ.get("CDSE_CLIENT_SECRET"))
        if not self.client_id or not self.client_secret:
            raise EnvironmentError(
                "Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET (OAuth client "
                "from the Copernicus Sentinel Hub Dashboard)"
            )
        self.session = requests.Session()
        self.token = None
        self.token_expiry = 0.0

    def authenticate(self):
        r = self.session.post(TOKEN_URL, data={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }, timeout=(15, 60))
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        self.token_expiry = time.monotonic() + int(data.get("expires_in", 300)) - 30

    def fetch_tar(self, body, path):
        # No automatic POST retries: a repeated processing request can count
        # towards account quotas. Retry explicitly if a transient failure occurs.
        for attempt in range(2):
            if self.token is None or time.monotonic() >= self.token_expiry:
                self.authenticate()
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/x-tar",
            }
            with self.session.post(
                PROCESS_URL, json=body, headers=headers,
                timeout=(20, 240), stream=True,
            ) as r:
                if r.status_code == 401 and attempt == 0:
                    self.token = None
                    continue
                if not r.ok:
                    error_text = r.text[:1200]
                    raise RuntimeError(
                        f"Process API HTTP {r.status_code}: {error_text}"
                    )
                size = 0
                with path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            size += len(chunk)
                            if size > MAX_ARCHIVE_BYTES:
                                raise RuntimeError("Process response is unexpectedly large")
                            f.write(chunk)
                if size < 1024:
                    raise RuntimeError("Process API returned an unexpectedly short archive")
                return
        raise RuntimeError("Sentinel Hub authentication failed after refresh")

    def close(self):
        self.session.close()


def unpack_result(tar_path, destination):
    """Safely read only default.tif and userdata.json without extracting paths."""
    tif_path = destination / "data.tif"
    metadata = None
    found_tif = False
    with tarfile.open(tar_path, mode="r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name not in ("default.tif", "userdata.json"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            if name == "default.tif":
                if found_tif or member.size > 250 * 1024 * 1024:
                    raise ValueError("Duplicate/oversized TIFF in archive")
                with tif_path.open("wb") as out:
                    shutil.copyfileobj(f, out, length=1024 * 1024)
                found_tif = True
            else:
                if member.size > 1024 * 1024:
                    raise ValueError("Metadata JSON too large")
                metadata = json.load(f)
    if not found_tif or not isinstance(metadata, dict):
        raise RuntimeError("Process response lacks default.tif or userdata.json")
    return tif_path, metadata


def verify_source(metadata, expected_product):
    """Fail closed if we cannot prove which satellite product was used."""
    expected_key = acquisition_key(expected_product)
    scene_ids = [
        t.get("sentinel1ProductId")
        for t in metadata.get("tiles", [])
        if isinstance(t, dict)
    ]
    if not scene_ids or any(not x for x in scene_ids):
        raise RuntimeError(
            "No usable source product IDs in Sentinel Hub userdata. "
            "Do not trust this raster until product provenance is resolved."
        )
    actual_keys = {acquisition_key(s) for s in scene_ids}
    if actual_keys != {expected_key}:
        raise RuntimeError(
            f"Process API sources do not match selected STAC product. "
            f"Expected key: {expected_key}; received keys: {sorted(actual_keys)}"
        )
    return scene_ids


def validate_raster(path, pair):
    """Inspect a TIFF window-by-window to avoid loading whole rasters in RAM."""
    expected_bounds = np.asarray(request_roi_bbox(pair), dtype=float)
    expected_width, expected_height = dimensions(expected_bounds)
    total_valid = 0
    total_pixels = expected_width * expected_height
    with rasterio.open(path) as ds:
        if ds.count != 3 or ds.width != expected_width or ds.height != expected_height:
            raise ValueError("Unexpected bands or raster dimensions")
        if ds.crs is None or ds.crs.to_epsg() != 32631:
            raise ValueError(f"Unexpected raster CRS: {ds.crs}")
        if not np.allclose(list(ds.bounds), expected_bounds, rtol=0, atol=0.02):
            raise ValueError("Raster bounds differ from requested ROI")
        if not np.isclose(abs(ds.transform.a), PIXEL_SIZE_M, atol=1e-5):
            raise ValueError("Unexpected pixel size")
        if not np.isclose(abs(ds.transform.e), PIXEL_SIZE_M, atol=1e-5):
            raise ValueError("Unexpected pixel size")
        for _, window in ds.block_windows(1):
            mask = ds.read(3, window=window) > 0.5
            total_valid += np.count_nonzero(mask)
            if np.any(mask):
                vh = ds.read(1, window=window)
                vv = ds.read(2, window=window)
                if not (np.isfinite(vv[mask]).all() and np.isfinite(vh[mask]).all()):
                    raise ValueError("NaN/Inf in valid SAR pixels")
                if OUTPUT_SCALE == "LINEAR":
                    if (vv[mask] < 0).any() or (vh[mask] < 0).any():
                        raise ValueError("Negative backscatter in linear SAR output")
    fraction = float(total_valid / total_pixels)
    if fraction < MIN_VALID_FRACTION:
        raise ValueError(
            f"Only {fraction:.1%} valid pixels (< {MIN_VALID_FRACTION:.1%})"
        )
    return {"valid_fraction": round(fraction, 5),
            "width": expected_width, "height": expected_height,
            "bands": ["VH", "VV", "dataMask"],
            "units": "dB" if OUTPUT_SCALE == "DB" else "linear_backscatter"}


def sidecar_path(event_dir, which):
    return event_dir / f"{which}_userdata.json"


# value that infer.py uses to determine there is not reliable data
NODATA_DB = np.nan

def split_roi_tif(src_path, output_dir, which):
    """Split VH/VV/mask into two single-band SAR GeoTIFFs."""
    if OUTPUT_SCALE != "DB":
        raise ValueError("Single-band export expects dB values")

    outputs = {}

    with rasterio.open(src_path) as src:
        valid = src.read(3) > 0.5

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="float32",
            nodata=NODATA_DB,
            compress="deflate",
        )

        for band, polarization in [(1, "VH"), (2, "VV")]:
            data = src.read(band).astype(np.float32)

            # Invalid pixels become NoData = Nan
            data[~valid] = NODATA_DB

            output = output_dir / f"{which}{polarization}.tif"

            with rasterio.open(output, "w", **profile) as dst:
                dst.write(data, 1)

            outputs[polarization] = output

    return outputs


def download_one(client, pair, which, event_dir):
    band_targets = {
        pol: event_dir / f"{which}{pol}.tif"
        for pol in ("VH", "VV")
    }
    meta_path = sidecar_path(event_dir, which)
    signature = request_signature(pair, which)
    if (
        SKIP_COMPLETED
        and all(p.exists() for p in band_targets.values())
        and meta_path.exists()
    ):
        try:
            sidecar = json.loads(meta_path.read_text(encoding="utf-8"))

            if sidecar["request_signature"] == signature:
                verify_source(sidecar["userdata"], pair[which]["id"])
                qc = sidecar["qc"]

                expected_width, expected_height = dimensions(
                    request_roi_bbox(pair)
                )

                for path in band_targets.values():
                    with rasterio.open(path) as ds:
                        if (
                            ds.count != 1
                            or ds.width != expected_width
                            or ds.height != expected_height
                            or ds.crs.to_epsg() != 32631
                            or ds.nodata is None 
                            or not np.isnan(ds.nodata)
                        ):
                            raise ValueError("Invalid cached SAR band")

                print(f"    {which}: cached")
                return {
                    "status": "cached",
                    "files": {k: str(v) for k, v in band_targets.items()},
                    **qc,
                }

        except (KeyError, ValueError, RuntimeError, OSError):
            print(f"    {which}: existing files need re-download")
    body = request_body(pair, which)
    with tempfile.TemporaryDirectory(dir=event_dir, prefix=f".{which}_") as tmp:
        tmp = Path(tmp)
        archive = tmp / "result.tar"
        print(f"    {which}: requesting {body['output']['width']}x{body['output']['height']} pixels")
        client.fetch_tar(body, archive)

        tif, userdata = unpack_result(archive, tmp)

        sources = verify_source(userdata, pair[which]["id"])

        # Diagnose the newly downloaded raster
        with rasterio.open(tif) as ds:
            vh = ds.read(1)
            vv = ds.read(2)
            mask = ds.read(3) > 0.5

            print("Valid coverage:", mask.mean())
            print("Mask values:", np.unique(ds.read(3)))

            if mask.any():
                for name, band in (("VH", vh), ("VV", vv)):
                    values = band[mask]
                    finite = values[np.isfinite(values)]
                    print(f"{name} finite: {finite.size}/{values.size}")
                    if finite.size:
                        print(f"{name} range ({OUTPUT_SCALE}): {finite.min()} to {finite.max()}")

        qc = validate_raster(tif, pair)
        bands = split_roi_tif(tif, tmp, which)
        document = {
            "selected_stac_id": pair[which]["id"],
            "qc": qc,
            "sources_used": sources,
            "selected_datetime": pair[which]["datetime"],
            "request_signature": signature,
            "request": body,
            "userdata": userdata,
        }
        metadata_tmp = tmp / "metadata.json"
        metadata_tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
        # Publish complete files only after all checks have passed.
        for polarization, path in bands.items():
            path.replace(band_targets[polarization])
        metadata_tmp.replace(meta_path)
        # Remove any old 3-band output after successful export
        (event_dir / f"{which}.tif").unlink(missing_ok=True)

    print(f"    {which}: saved ({qc['valid_fraction']:.1%} valid)")
    return {
        "status": "downloaded",
        "files": {k: str(v) for k, v in band_targets.items()},
        **qc,
    }


def main():
    if not PAIRS_FILE.exists():
        raise FileNotFoundError(f"Cannot find {PAIRS_FILE}")
    pairs = json.loads(PAIRS_FILE.read_text(encoding="utf-8"))
    if not isinstance(pairs, list):
        raise ValueError("sentinel_pairs.json must contain a list")
    selected = pairs if MAX_PAIRS is None else pairs[:MAX_PAIRS]
    if not selected:
        print("No selected pairs to download")
        return
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    client = SentinelHubClient()
    completed = 0
    try:
        for index, pair in enumerate(selected, start=1):
            code = str(pair["codiallau"])
            if not re.fullmatch(r"[A-Za-z0-9_-]+", code):
                raise ValueError(f"Unsafe avalanche identifier: {code}")
            event_dir = OUT_ROOT / code
            event_dir.mkdir(parents=True, exist_ok=True)
            row = {"codiallau": code, "status": "pending"}
            print(f"\n[{index}/{len(selected)}] Avalanche {code}")
            try:
                # Validate this event before spending processing units.
                request_body(pair, "pre")
                request_body(pair, "post")
                row["pre"] = download_one(client, pair, "pre", event_dir)
                row["post"] = download_one(client, pair, "post", event_dir)
                row["status"] = "completed"
                completed += 1
                manifest = {
                    "codiallau": code,
                    "occurrence_date": pair.get("occurrence_date"),
                    "roi_bbox": request_roi_bbox(pair),
                    "roi_crs": "EPSG:32631",
                    "pre": {"selected": pair["pre"], **row["pre"]},
                    "post": {"selected": pair["post"], **row["post"]},
                    "processing": {
                        "backscatter": BACKSCATTER_COEFF,
                        "dem": DEM_INSTANCE,
                        "speckle_filter": SPECKLE_FILTER,
                        "speckle_filter_window": LEE_WINDOW if SPECKLE_FILTER == "LEE" else None,
                        "upsampling": UPSAMPLING,
                        "radiometric_terrain_oversampling": RADIOMETRIC_TERRAIN_OVERSAMPLING,
                        "pixel_size_m": PIXEL_SIZE_M,
                        "units": OUTPUT_SCALE,
                        "bands": ["preVH", "preVV", "postVH", "postVV"],
                        "nodata": "NaN"
                    },
                    "note": "Preprocessing only approximates SNAP; validate and apply training normalization before inference",
                }
                temp_manifest = event_dir / ".manifest.partial.json"
                temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                temp_manifest.replace(event_dir / "manifest.json")
            except (requests.RequestException, RuntimeError, ValueError,
                    OSError, tarfile.TarError) as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                print(f"    ERROR {code}: {exc}")
        
            if TARGET_SUCCESSFUL_PAIRS is not None and completed >= TARGET_SUCCESSFUL_PAIRS:
                break
            time.sleep(SLEEP_BETWEEN_EVENTS_SECONDS)
    finally:
        client.close()


if __name__ == "__main__":
    main()