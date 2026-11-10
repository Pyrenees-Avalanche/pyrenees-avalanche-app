from pathlib import Path

import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize


CODE = "TO_DEFINE"
PROB_PATH = Path(f"{CODE}_prob.tif")
ICGC_PATH = Path(r".\SAR\icgc\icgc_events.geojson")

with rasterio.open(PROB_PATH) as src:
    probabilities = src.read(1)
    transform = src.transform
    crs = src.crs
    shape = src.shape
    nodata = src.nodata

events = gpd.read_file(ICGC_PATH)
event = events.loc[
    events["codiallau"] == CODE
].to_crs(crs)

assert not event.empty

gt = rasterize(
    [(geom, 1) for geom in event.geometry],
    out_shape=shape,
    transform=transform,
    fill=0,
).astype(bool)

valid = np.isfinite(probabilities)
valid &= (probabilities >= 0) & (probabilities <= 1)

if nodata is not None and np.isfinite(nodata):
    valid &= probabilities != nodata

for label, mask in [
    ("INSIDE", gt & valid),
    ("OUTSIDE", ~gt & valid),
]:
    values = probabilities[mask]
    if not values.size:
        print(label, "No valid pixels")
        continue

    print(f"\n{label}")
    print("Mean:", values.mean())
    print("Median:", np.median(values))
    print("90th percentile:", np.percentile(values, 90))
    print("99th percentile:", np.percentile(values, 99))
    print("Maximum:", values.max())

print("\n--- THRESHOLD COMPARISON ---")
for threshold in [0.1, 0.2, 0.3, 0.5, 0.97]:
    pred = (probabilities > threshold) & valid
    truth = gt & valid

    tp = np.count_nonzero(pred & truth)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(truth & ~pred)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(
        f"Threshold {threshold:.2f}: "
        f"Precision={precision:.3f}, "
        f"Recall={recall:.3f}"
    )