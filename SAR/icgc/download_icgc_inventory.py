""" Downloads all the geometries (geospatial maps) for the recorded avalanches"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


BASE_URL = "https://geoserveis.icgc.cat/geoserver/nivoallaus/ows"
LAYER = "nivoallaus:observacions"
OUTPUT = Path(r".\SAR\icgc\icgc_observacions_full.geojson")
PAGE_SIZE = 250

COMMON_PARAMS = {
    "service": "WFS",
    "version": "2.0.0",
    "request": "GetFeature",
    "typeNames": LAYER,
}


def get_total_count(session):
    """Ask WFS how many observations exist."""
    params = {
        **COMMON_PARAMS,
        "resultType": "hits",
    }

    response = session.get(
        BASE_URL, params=params, timeout=60
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    total = root.attrib.get("numberMatched")

    if total is None or not total.isdigit():
        raise ValueError(
            f"Unexpected WFS count: {total}"
        )

    return int(total)


def fetch_page(session, start_index):
    """Download one page of GeoJSON features."""
    params = {
        **COMMON_PARAMS,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "sortBy": "gid",
        "count": PAGE_SIZE,
        "startIndex": start_index,
    }

    response = session.get(
        BASE_URL, params=params, timeout=120
    )
    response.raise_for_status()

    data = response.json()

    if data.get("type") != "FeatureCollection":
        raise ValueError("Unexpected GeoJSON response")

    return data["features"]


def main():
    seen_ids = set()
    downloaded = 0
    temp_file = OUTPUT.with_suffix(".partial.geojson")

    with requests.Session() as session:
        total = get_total_count(session)
        print(f"Total ICGC observations: {total}")

        # Stream features to disk to avoid keeping
        # all their geometries in RAM.
        with temp_file.open("w", encoding="utf-8") as f:
            f.write('{"type":"FeatureCollection","features":[')
            first = True

            while downloaded < total:
                features = fetch_page(session, downloaded)

                if not features:
                    raise RuntimeError(
                        "WFS returned an empty page before "
                        "all records were downloaded"
                    )

                for feature in features:
                    gid = feature["properties"]["gid"]

                    if gid in seen_ids:
                        raise RuntimeError(
                            f"Duplicate gid encountered: {gid}"
                        )

                    seen_ids.add(gid)

                    if not first:
                        f.write(",")

                    json.dump(feature, f, ensure_ascii=False)
                    first = False

                downloaded += len(features)

                print(
                    f"Downloaded {downloaded}/{total} "
                    f"observations"
                )

            f.write("]}")

    if downloaded != total:
        raise RuntimeError(
            f"Expected {total} features, got {downloaded}"
        )

    temp_file.replace(OUTPUT)

    print(f"\nSaved: {OUTPUT}")
    print(f"Unique observations: {len(seen_ids)}")


if __name__ == "__main__":
    main()