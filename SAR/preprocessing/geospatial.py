from pathlib import Path
import matplotlib.pyplot as plt


import numpy as np
import rasterio
from rasterio.transform import from_origin, rowcol, array_bounds
from rasterio.warp import reproject, Resampling


### Configs ###

# Avalanche location in ETRS89 / UTM 31N
E = 334185.14
N = 4728518.23

# Grid configuration
TARGET_CRS = "EPSG:25831"
RESOLUTION = 10
SIZE = 512

# "linear" for Gamma0 power, "db" for Gamma0 in dB
INPUT_SCALE = "linear"

# IMPORTANT: verify this order against SNAP!
BANDS = {
    "VH": 1,
    "VV": 2,
}

# For our SNAP exports, zeros appeared to be
# background fill where source NoData was absent.
ZERO_IS_NODATA = True

DST_NODATA = np.nan

PRE_PATH = Path(r"SAR/processed/pre_linear.tif")
POST_PATH = Path(r"SAR/processed/post_linear.tif")
OUTPUT_DIR = Path(r"SAR/processed/Manaud/20220403")


### Create the grid ###

# Calculate the upper-left corner
left = E - SIZE * RESOLUTION / 2
top = N + SIZE * RESOLUTION / 2

# Construct the affine transformation
target_transform = from_origin(
    left,
    top,
    RESOLUTION,
    RESOLUTION
)

# Verify the grid
bounds = array_bounds(SIZE, SIZE, target_transform)
row, col = rowcol(target_transform, E, N)

print("CRS:", TARGET_CRS)
print("Transform:\n", target_transform)
print("Bounds:", bounds)
print("Avalanche pixel:", (row, col))


### REPROJECT ONE POLARIZATION ###

def align_band(src, band_index):

    aligned = np.full(
        (SIZE, SIZE),
        DST_NODATA,
        dtype=np.float32
    )

    if src.nodata is not None:
        source_nodata = src.nodata
    elif ZERO_IS_NODATA:
        source_nodata = 0.0
    else:
        source_nodata = None

    reproject(
        source=rasterio.band(src, band_index),
        destination=aligned,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=source_nodata,
        dst_transform=target_transform,
        dst_crs=TARGET_CRS,
        dst_nodata=DST_NODATA,
        resampling=Resampling.bilinear,
        init_dest_nodata=True,
        num_threads=2,
    )

    # Convert linear Gamma0 power to decibels.
    if INPUT_SCALE == "linear":
        valid = np.isfinite(aligned) & (aligned > 0)

        db = np.full_like(aligned, np.nan)
        db[valid] = 10 * np.log10(aligned[valid])
        aligned = db

    elif INPUT_SCALE != "db":
        raise ValueError("INPUT_SCALE must be 'linear' or 'db'")

    return aligned


### WRITE FOUR SINGLE-BAND GEOTIFFS ###

def save_band(data, output_path):

    profile = {
        "driver": "GTiff",
        "height": SIZE,
        "width": SIZE,
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": target_transform,
        "nodata": np.nan,
        "compress": "deflate",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {}
    arrays = {}

    for date, source_path in [
        ("pre", PRE_PATH),
        ("post", POST_PATH),
    ]:

        with rasterio.open(source_path) as src:

            print(f"\n{date}:")
            print("Source CRS:", src.crs)
            print("Source bands:", src.count)
            print("Descriptions:", src.descriptions)

            if src.count < 2 or src.crs is None:
                raise ValueError(
                    f"Invalid source raster: {source_path}"
                )

            for pol, band_idx in BANDS.items():

                key = f"{date}{pol}"
                data = align_band(src, band_idx)

                arrays[key] = data
                print(
                    key,
                    "valid fraction:",
                    np.mean(np.isfinite(data))
                )

    # Only keep pixels observed in all four channels.
    common_valid = np.logical_and.reduce([
        np.isfinite(data) for data in arrays.values()
    ])

    print("\nCommon valid fraction:", common_valid.mean())
    print(
        "Avalanche centre valid:",
        bool(common_valid[SIZE // 2, SIZE // 2])
    )

    if not common_valid[SIZE // 2, SIZE // 2]:
        raise RuntimeError(
            "Missing SAR data at avalanche centre. "
            "Check coverage, processing, band order and masks."
        )

    if common_valid.mean() < 0.8:
        print("WARNING: Less than 80% common coverage.")

    for key, data in arrays.items():
        data[~common_valid] = np.nan

        path = OUTPUT_DIR / f"{key}.tif"
        save_band(data, path)
        paths[key] = path

        values = data[np.isfinite(data)]

        print(
            key,
            "dB percentiles:",
            np.percentile(values, [1, 50, 99])
        )
        print("Saved:", path)

    # Verify the geographical grids match.
    with rasterio.open(paths["preVH"]) as reference:
        for path in paths.values():
            with rasterio.open(path) as src:
                assert src.crs == reference.crs
                assert src.transform == reference.transform
                assert src.shape == reference.shape

    print("\nAll four images are aligned!")


### Visualize ###

def visualize(folder_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    for ax, name in zip(
        axes.flat,
        ["preVH", "postVH", "preVV", "postVV"]
    ):
        with rasterio.open(folder_path / f"{name}.tif") as src:
            data = src.read(1)

        image = ax.imshow(
            data,
            cmap="gray",
            vmin=-30,
            vmax=0,
        )

        ax.scatter(256, 256, c="red", s=25, marker="+")
        ax.set_title(name)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()

    folder = Path(r"SAR/processed/Manaud/20220403")

    visualize(folder)