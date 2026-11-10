from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import plotting_extent

CODE = "TO_DEFINE"
EVENT = Path(f"{CODE}")
PROB = Path(f"{CODE}_prob.tif")
ICGC = Path(r".\SAR\icgc\icgc_events.geojson")


def read_tif(path):
    with rasterio.open(path) as ds:
        return (
            ds.read(1).astype(np.float32),
            ds.crs,
            ds.transform,
            plotting_extent(ds),
        )

pre_vh, crs, transform, extent = read_tif(EVENT / "preVH.tif")
post_vh, _, _, _ = read_tif(EVENT / "postVH.tif")

pre_vv, _, _, _ = read_tif(EVENT / "preVV.tif")
post_vv, _, _, _ = read_tif(EVENT / "postVV.tif")

prob, prob_crs, prob_transform, prob_extent = read_tif(PROB)

assert prob_crs == crs
assert prob.shape == pre_vh.shape
assert prob_transform.almost_equals(transform)

events = gpd.read_file(ICGC)
polygon = events.loc[events["codiallau"] == CODE].to_crs(crs)

assert not polygon.empty

delta_vh = post_vh - pre_vh
delta_vv = post_vv - pre_vv

# Log scale makes very small probabilities visible.
log_prob = np.log10(
    np.clip(prob, 1e-6, 1.0)
)
log_prob[~np.isfinite(prob) | (prob < 0)] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

plots = [
    (delta_vh, "VH: POST - PRE", "RdBu_r", -10, 10),
    (delta_vv, "VV: POST - PRE", "RdBu_r", -10, 10),
    (log_prob, "Log10 model probability", "magma", -6, 0),
]

for ax, (image, title, cmap, vmin, vmax) in zip(axes, plots):
    im = ax.imshow(
        image,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    polygon.boundary.plot(
        ax=ax,
        edgecolor="lime",
        linewidth=2,
    )

    ax.set_title(title)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, shrink=0.75)

plt.tight_layout()
plt.savefig("icgc_diagnostic.png", dpi=180)
plt.show()