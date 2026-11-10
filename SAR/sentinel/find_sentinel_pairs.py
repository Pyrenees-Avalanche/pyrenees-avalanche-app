"""Find spatially complete, orbit-compatible Sentinel-1 pre/post pairs.

Reads dated avalanche polygons from icgc_events.geojson and queries the
Copernicus Data Space STAC catalogue. No satellite pixel data is downloaded.
"""

import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import pandas as pd
import requests
from pyproj import Transformer
from requests.adapters import HTTPAdapter
from shapely.geometry import shape
from shapely.ops import transform
from urllib3.util.retry import Retry

INPUT = Path(r".\SAR\data\icgc_events.geojson")
OUTPUT = Path(r".\SAR\data\sentinel_pairs.json")

CACHE = Path("cache/stac")
STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"

TARGET_PAIRS = 100    
SEARCH_DAYS = 10       # Search up to 10 days before and after the event (20 days difference in total)
MAX_PAIR_GAP_DAYS = 14
BUFFER_M = 640         # Context around the avalanche
MIN_SIDE_M = 1280      # Minimum 128 x 128 pixels at 10 m
PIXEL_M = 10
MIN_COVERAGE = 0.995   # Each product must cover >= 99.5% of the ROI

TO_WGS84 = Transformer.from_crs(25831, 4326, always_xy=True)
TO_METRIC = Transformer.from_crs(4326, 25831, always_xy=True)


def make_roi(geom):
    """Make a buffered, 10 m-grid-aligned rectangular region in EPSG:25831."""
    minx, miny, maxx, maxy = geom.buffer(BUFFER_M).bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half_w = max((maxx - minx) / 2, MIN_SIDE_M / 2)
    half_h = max((maxy - miny) / 2, MIN_SIDE_M / 2)
    from shapely.geometry import box
    return box(
        math.floor((cx - half_w) / PIXEL_M) * PIXEL_M,
        math.floor((cy - half_h) / PIXEL_M) * PIXEL_M,
        math.ceil((cx + half_w) / PIXEL_M) * PIXEL_M,
        math.ceil((cy + half_h) / PIXEL_M) * PIXEL_M,
    )


def get_catalog_items(session, bbox, event_date):
    """Query all metadata pages, caching successful replies by search inputs."""
    CACHE.mkdir(parents=True, exist_ok=True)
    start = event_date - timedelta(days=SEARCH_DAYS)
    end = event_date + timedelta(days=SEARCH_DAYS + 1)
    params = {
        "collections": "sentinel-1-grd",
        "bbox": ",".join(f"{v:.7f}" for v in bbox),
        "datetime": f"{start}T00:00:00Z/{end}T00:00:00Z",
        "limit": 100,
    }

    # Cache identity includes bbox, event date and search window.
    import hashlib
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
    cache_path = CACHE / f"{key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    items, seen_urls = [], set()
    url, request_params = STAC_URL, params
    next_method, next_body = "GET", None
    for _ in range(50):  # Guard against a broken pagination loop
        page_key = (next_method, url, json.dumps(
            next_body if next_method == "POST" else request_params,
            sort_keys=True,
        ))
        if page_key in seen_urls:
            raise RuntimeError("STAC returned a repeated pagination request")
        seen_urls.add(page_key)

        if next_method == "POST":
            response = session.post(url, json=next_body, timeout=60)
        else:
            response = session.get(url, params=request_params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload.get("type") != "FeatureCollection":
            raise ValueError("Unexpected STAC response")
        items.extend(payload.get("features", []))

        next_link = next((x for x in payload.get("links", [])
                          if x.get("rel") == "next"), None)
        if not next_link:
            cache_path.write_text(json.dumps(items), encoding="utf-8")
            return items
        url = urljoin(response.url, next_link["href"])
        next_method = next_link.get("method", "GET").upper()
        next_body = next_link.get("body")
        if next_method == "POST" and next_body is None:
            raise RuntimeError("STAC POST pagination link has no body")
        request_params = None  # Next link must contain its own query

    raise RuntimeError("STAC pagination exceeded 50 pages")


def item_to_candidate(item, roi, event_date):
    """Reject acquisitions with wrong mode, polarizations, or footprint."""
    props = item.get("properties", {})
    if props.get("sar:instrument_mode") != "IW":
        return None

    polar = (props.get("sar:polarizations") or
             props.get("sar:polarization") or
             props.get("s1:polarization") or [])
    if isinstance(polar, str):
        polar = {"DV": ["VV", "VH"]}.get(polar.upper(),
                                           polar.upper().replace("+", " ").split())
    if not {"VV", "VH"}.issubset(set(polar)):
        return None

    orbit = props.get("sat:relative_orbit")
    direction = str(props.get("sat:orbit_state", "")).lower()
    if orbit is None or direction not in {"ascending", "descending"}:
        return None  # We won't guess missing orbit information

    timestamp = props.get("datetime") or props.get("start_datetime")
    if not timestamp or not item.get("geometry"):
        return None
    acquired = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if acquired.tzinfo is None:
        acquired = acquired.replace(tzinfo=timezone.utc)
    acquired = acquired.astimezone(timezone.utc)
    if acquired.date() == event_date:
        return None  # Time of avalanche is not known precisely

    scene = transform(TO_METRIC.transform, shape(item["geometry"]))
    coverage = roi.intersection(scene).area / roi.area
    if coverage < MIN_COVERAGE:
        return None

    return {
        "id": item["id"], "datetime": acquired.isoformat(),
        "date": acquired.date(), "platform": str(props.get("platform", "")),
        "relative_orbit": int(orbit), "orbit_state": direction,
        "coverage": round(coverage, 5),
    }


def select_pair(candidates, event_date):
    """Find closest bracketing acquisitions with matching orbit geometry."""
    pre = [x for x in candidates if x["date"] < event_date]
    post = [x for x in candidates if x["date"] > event_date]
    eligible_pairs = []
    for before in pre:
        for after in post:
            if (before["relative_orbit"] != after["relative_orbit"] or
                    before["orbit_state"] != after["orbit_state"]):
                continue
            gap = (datetime.fromisoformat(after["datetime"]) -
                   datetime.fromisoformat(before["datetime"])).total_seconds() / 86400
            if gap > MAX_PAIR_GAP_DAYS:
                continue
            same_satellite = before["platform"] == after["platform"]
            eligible_pairs.append((gap, not same_satellite, before, after))

    if not eligible_pairs:
        return None
    # Shortest time interval first; prefer same satellite as tie-breaker.
    _, _, before, after = min(eligible_pairs, key=lambda x: (x[0], x[1]))
    return before, after


def main():
    events = gpd.read_file(INPUT)
    if events.crs is None:
        raise ValueError("The input dataset has no CRS")
    events = events.sort_values("occurrence_date", ascending=False)
    metric = events.to_crs(25831)

    retries = Retry(total=2, backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))

    pairs = []
    with session:
        for _, event in metric.iterrows():
            code = str(event["codiallau"])
            status, n_items, n_ok = "unknown", 0, 0
            try:
                event_date = date.fromisoformat(str(event["occurrence_date"])[:10])
                if event.geometry is None or event.geometry.is_empty:
                    raise ValueError("Missing geometry")
                roi = make_roi(event.geometry)
                bbox_wgs84 = transform(TO_WGS84.transform, roi).bounds
                items = get_catalog_items(session, bbox_wgs84, event_date)
                n_items = len(items)
                candidates = [c for item in items
                              if (c := item_to_candidate(item, roi, event_date)) is not None]
                n_ok = len(candidates)
                pair = select_pair(candidates, event_date)
                if pair is None:
                    status = "no_compatible_pair"
                    if n_items and not n_ok:
                        print("  Sample STAC fields:", list(items[0].get("properties", {}).keys()))
                    elif n_ok:
                        print("  Eligible acquisitions:", [
                            (c["id"], str(c["date"]), c["relative_orbit"], c["orbit_state"])
                            for c in candidates[:10]
                        ])
                else:
                    before, after = pair
                    status = "paired"
                    pairs.append({
                        "codiallau": code, "occurrence_date": str(event_date),
                        "roi_crs": "EPSG:25831", "roi_bbox": list(roi.bounds),
                        "pre": {k: v for k, v in before.items() if k != "date"},
                        "post": {k: v for k, v in after.items() if k != "date"},
                    })
            except (ValueError, KeyError, requests.RequestException,
                    RuntimeError, TypeError) as exc:
                status = "error"
                print(f"  {code}: {exc}")
    
            print(f"{code}: {status} | catalog={n_items} eligible={n_ok} "
                  f"| pairs={len(pairs)}")
            if TARGET_PAIRS is not None and len(pairs) >= TARGET_PAIRS:
                break
            time.sleep(0.2)

    OUTPUT.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()