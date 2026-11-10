""" Adds the date of occurrence of the avalanches and performs some filterings."""

import requests
import re
import time

from pathlib import Path
from datetime import date

import pandas as pd
import geopandas as gpd
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


MOST_RECENT_FIRST = True
TARGET_SAMPLES = 100    
MIN_AREA_M2 = 1500      
MIN_SEASON = 2014       # Sentinel-1 era


INPUT = Path(r".\SAR\icgc\icgc_observacions_full.geojson")
OUTPUT = Path(r".\SAR\icgc\icgc_events.geojson")
CACHE = Path(r".\SAR\icgc\cache\icgc_reports")

BASE_URL = (
    "https://geoserveis.icgc.cat/siapc/"
    "submitAllauListFormQCUD.do"
)

CACHE.mkdir(parents=True, exist_ok=True)


def get_report(code, session):
    """Retrieve one report, using the local cache if possible."""
    path = CACHE / f"{code}.html"

    if path.exists():
        return path.read_bytes()

    response = session.get(
        BASE_URL,
        params={"reportA": code},
        timeout=30,
    )
    response.raise_for_status()

    html = response.content
    path.write_bytes(html)

    # Avoid sending too many requests to ICGC
    time.sleep(0.5)
    return html


def parse_report(html):
    """Extract the two distinct dates from the ICGC report."""
    soup = BeautifulSoup(html, "html.parser")
    text = " ".join(soup.stripped_strings)
    text = re.sub(r"\s+", " ", text)

    if not re.search(
        r"INFORME DE OBSERVACI[OÓ]N DE ALUD",
        text, re.IGNORECASE
    ):
        raise ValueError("Unrecognized report page")

    # Date on which the avalanche was observed
    obs_match = re.search(
        r"Fecha de observaci[oó]n\s*"
        r"\(aaaa-mm-dd\)\s*(\d{4}-\d{2}-\d{2})",
        text,
        re.IGNORECASE,
    )

    # Restrict the occurrence date search to its section
    section = re.search(
        r"DATOS DE CA[IÍ]DA DEL ALUD"
        r"(.*?)DATOS DEL ALUD",
        text,
        re.IGNORECASE,
    )

    fall_match = None
    if section:
        fall_match = re.search(
            r"Fecha\s*\(aaaa-mm-dd\)\s*"
            r"(\d{4}-\d{2}-\d{2})",
            section.group(1),
            re.IGNORECASE,
        )

    def checked_date(match):
        if match is None:
            return None

        value = match.group(1)
        date.fromisoformat(value)  # Reject impossible dates
        return value

    return checked_date(fall_match), checked_date(obs_match)


def main():
    gdf = gpd.read_file(INPUT)

    gdf["codiallau"] = (
        gdf["codiallau"].astype("string").str.strip()
    )

    years = gdf["codiallau"].str.extract(
        r"^[A-Z]{3}\d{3}(\d{4})\d{2}$",
        expand=False,
    )

    gdf["season_start"] = pd.to_numeric(
        years, errors="coerce"
    ).astype("Int64")

    valid_geometry = (
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf.geometry.is_valid
    )

    eligible = gdf.loc[
        valid_geometry
        & gdf["season_start"].ge(MIN_SEASON).fillna(False)
    ].copy()

    # Project before calculating areas
    eligible = eligible.to_crs(epsg=25831)

    # Combine multiple geometry rows belonging to
    # the same avalanche identifier
    candidates = eligible.dissolve(
        by="codiallau",
        as_index=False,
        aggfunc="first",
    )

    candidates["area_m2"] = candidates.geometry.area

    # Remove small avalanches
    before = len(candidates)

    candidates = candidates.loc[
        candidates["area_m2"] >= MIN_AREA_M2
    ].copy()

    print(f"Removed small avalanches: {before - len(candidates)}")

    # Newest seasons first, larger areas first
    candidates = candidates.sort_values(
        ["season_start", "area_m2"],
        ascending=[False, False],
    )

    # Configure HTTP retries
    retry = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)

    valid_events = []

    with requests.Session() as session:
        session.mount("https://", adapter)

        # Process complete winter seasons
        for season, season_rows in candidates.groupby(
            "season_start", sort=False
        ):
            print(
                f"\nSeason {season}: "
                f"{len(season_rows)} candidate avalanches"
            )

            for _, row in season_rows.iterrows():
                code = row["codiallau"]

                record = {
                    "codiallau": code,
                    "season_start": int(season),
                    "area_m2": row["area_m2"],
                    "status": "pending",
                    "error": None,
                }

                try:
                    html = get_report(code, session)
                    occurrence, observation = parse_report(html)

                    if occurrence is None:
                        record["status"] = "missing_date"
                        continue

                    event_date = date.fromisoformat(occurrence)

                    if observation:
                        obs_date = date.fromisoformat(observation)
                        if event_date > obs_date:
                            record["status"] = "inconsistent_date"
                            continue

                    # Check reported date belongs to
                    # the expected winter-season interval
                    season_begin = date(int(season), 9, 1)
                    season_end = date(int(season) + 1, 8, 31)

                    if not (season_begin <= event_date <= season_end):
                        record["status"] = "outside_season"
                        continue

                    event = row.to_dict()
                    event["occurrence_date"] = occurrence
                    event["observation_date"] = observation
                    event["report_url"] = (
                        f"{BASE_URL}?reportA={code}"
                    )

                    valid_events.append(event)
                    record["status"] = "valid"

                except requests.RequestException as exc:
                    print(f"  Skipping {code}: {exc}")

                except ValueError as exc:
                    print(f"  Invalid report {code}: {exc}")


            # Finish the season before checking N.
            # This ensures all its avalanches have
            # been considered for largest-area selection.
            distinct_dates = len({
                e["occurrence_date"] for e in valid_events
            })

            print(
                f"Valid unique dates collected: "
                f"{distinct_dates}/{TARGET_SAMPLES}"
            )

            if distinct_dates >= TARGET_SAMPLES:
                break

    if not valid_events:
        raise RuntimeError("No valid events found.")

    # Select largest avalanche per date
    dataset = gpd.GeoDataFrame(
        valid_events,
        geometry="geometry",
        crs=candidates.crs,
    )

    dataset = dataset.sort_values(
        ["occurrence_date", "area_m2"],
        ascending=[False, False],
    )

    # Because we sorted by area descending,
    # keep='first' retains the largest event
    # whenever multiple avalanches share a date.
    dataset = dataset.drop_duplicates(
        subset="occurrence_date",
        keep="first",
    )

    # Keep N most recent distinct dates
    dataset = dataset.head(TARGET_SAMPLES)

    # Export
    dataset = dataset.to_crs(epsg=4326)
    dataset.to_file(OUTPUT, driver="GeoJSON")

    print("\n--- FINAL DATASET ---")
    print(f"Final samples: {len(dataset)}")
    print(f"Unique dates: {dataset['occurrence_date'].nunique()}")

    print("\nReport statuses:")

    print("\nSelected events:")
    print(
        dataset[
            ["codiallau", "occurrence_date", "area_m2"]
        ].to_string(index=False)
    )

    print(f"\nDataset saved to: {OUTPUT}")

    if len(dataset) < TARGET_SAMPLES:
        print(
            f"\nWARNING: Only {len(dataset)} eligible events "
            f"were available; target was {TARGET_SAMPLES}."
        )


if __name__ == "__main__":
    main()