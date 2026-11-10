import subprocess
import time
from pathlib import Path

import rasterio
from pyproj import Transformer
import xml.etree.ElementTree as ET



# Adapt the SNAP graph template to our pipeline.
graph_path = Path("./SAR/graphs/gamma0_template.xml")
tree = ET.parse(graph_path)
root = tree.getroot()


def set_parameter(node_id, parameter, value):
    node = root.find(f".//node[@id='{node_id}']")

    if node is None:
        raise ValueError(f"Missing SNAP node: {node_id}")

    element = node.find(f"parameters/{parameter}")

    if element is None:
        element = ET.SubElement(node.find("parameters"), parameter)

    element.text = str(value)


# Configure the output projection, pixel size, and speckle filter.
set_parameter("Terrain-Correction", "mapProjection", "EPSG:32631")
set_parameter("Terrain-Correction", "pixelSpacingInMeter", "10.0")
set_parameter("Speckle-Filter", "filter", "Refined Lee")

tree.write(
    "./SAR/graphs/pyrenees_preprocessing.xml",
    encoding="utf-8",
    xml_declaration=True,
)

GPT = Path(r"C:\Program Files\esa-snap\bin\gpt.exe")
GRAPH = Path("./SAR/graphs/pyrenees_preprocessing.xml")


def create_aoi_wkt(
    easting=334185.14,
    northing=4728518.23,
    buffer_m=10000,
):
    """Create a 20 x 20 km subset around the Manaud avalanche."""

    corners = [
        (easting - buffer_m, northing - buffer_m),
        (easting + buffer_m, northing - buffer_m),
        (easting + buffer_m, northing + buffer_m),
        (easting - buffer_m, northing + buffer_m),
        (easting - buffer_m, northing - buffer_m),
    ]

    transformer = Transformer.from_crs(
        "EPSG:25831",
        "EPSG:4326",
        always_xy=True,
    )

    coordinates = [transformer.transform(x, y) for x, y in corners]
    wkt = "POLYGON((" + ", ".join(
        f"{lon} {lat}" for lon, lat in coordinates
    ) + "))"

    return wkt


def run_snap(input_zip: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aoi = create_aoi_wkt()

    command = [
        str(GPT),
        str(GRAPH.resolve()),
        f"-Pinput={input_zip.resolve()}",
        f"-Poutput={output_path.resolve()}",
        f"-Paoi={aoi}",
        "-q",
        "4",
        "-c",
        "1G",
    ]

    print("Processing:", input_zip.name, flush=True)
    print("AOI:", aoi, flush=True)

    started_at = time.time()
    started = time.monotonic()
    with subprocess.Popen(command) as process:
        while True:
            try:
                return_code = process.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                elapsed = int(time.monotonic() - started)
                output = output_path.stat() if output_path.exists() else None
                if output and output.st_mtime >= started_at:
                    if output.st_size == 0:
                        status = "GeoTIFF created; waiting for data"
                    elif output.st_size < 1024**2:
                        status = f"GeoTIFF output: {output.st_size:,} bytes"
                    else:
                        status = f"GeoTIFF output: {output.st_size / 1024**2:.1f} MiB"
                else:
                    status = "SNAP graph running"
                print(f"{status} | elapsed {elapsed // 60:02d}:{elapsed % 60:02d}", flush=True)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"SNAP finished without a usable GeoTIFF: {output_path}")
    elapsed = int(time.monotonic() - started)
    print(f"SNAP graph complete in {elapsed // 60:02d}:{elapsed % 60:02d}.", flush=True)


run_snap(
    input_zip=Path(r".\SAR\raw\S1A_IW_GRDH_1SDV_20220330T174729_20220330T174754_042554_05137B_B22D.SAFE.zip"),
    output_path=Path(r".\SAR\processed\pre_linear.tif"),
)

run_snap(
    input_zip=Path(r".\SAR\raw\S1A_IW_GRDH_1SDV_20220411T174729_20220411T174754_042729_05195C_E436.SAFE.zip"),
    output_path=Path(r".\SAR\processed\post_linear.tif"),
)

print("Inspecting GeoTIFF metadata...", flush=True)
with rasterio.open(r".\SAR\processed\pre_linear.tif") as src:
    print("Shape:", src.shape)
    print("Bands:", src.count)
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Descriptions:", src.descriptions)
    print("NoData:", src.nodata)