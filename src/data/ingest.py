import os
import time
import json
import hashlib
from math import asin, cos, radians, sin, sqrt

import pandas as pd
import requests

SCHEMA_VERSION = 2

DEFAULT_CITY_CONFIG_PATH = "data/config/city_centers.json"

# Candidate sampling pattern around each city center.
# Offset unit is degree (~11 km latitude per 0.1 degree).
DEFAULT_NODE_OFFSETS = [
    (0.00, 0.00),
    (0.12, 0.00),
    (-0.12, 0.00),
    (0.00, 0.12),
    (0.00, -0.12),
    (0.08, 0.08),
    (0.08, -0.08),
    (-0.08, 0.08),
    (-0.08, -0.08),
]

START_DATE = "2005-01-01"
END_DATE = "2026-01-01"  # exclusive end -> includes data up to 2025-12-31
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

REQUIRED_VARIABLES = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "soil_moisture_0_to_7cm_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "shortwave_radiation_sum",
    "wind_speed_10m_mean",
]


def load_city_centers(config_path=DEFAULT_CITY_CONFIG_PATH):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"City config not found: {config_path}. "
            "Create config JSON with city_id -> {lat, lon}."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        city_map = json.load(f)
    if not isinstance(city_map, dict) or not city_map:
        raise ValueError("City config must be a non-empty object.")
    for city_id, coords in city_map.items():
        if "lat" not in coords or "lon" not in coords:
            raise ValueError(f"City {city_id} missing lat/lon in config.")
    return city_map


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    )
    return 2 * radius * asin(sqrt(a))


def build_city_nodes(city_id, center_lat, center_lon, node_offsets=None):
    if node_offsets is None:
        node_offsets = DEFAULT_NODE_OFFSETS
    nodes = []
    for idx, (d_lat, d_lon) in enumerate(node_offsets):
        lat = center_lat + d_lat
        lon = center_lon + d_lon
        node_local_id = f"n{idx:02d}"
        # Collision-safe deterministic node_id from stable node metadata.
        coord_token = hashlib.sha1(f"{lat:.6f},{lon:.6f}".encode("utf-8")).hexdigest()[:8]
        node_id = f"{city_id}__{node_local_id}__{coord_token}"
        nodes.append(
            {
                "city_id": city_id,
                "node_local_id": node_local_id,
                "node_id": node_id,
                "raw_node_id": node_id,
                "lat": lat,
                "lon": lon,
                "distance_km_from_city_center": haversine_km(
                    center_lat, center_lon, lat, lon
                ),
            }
        )
    return nodes


def fetch_node_data(node_meta, max_retries=3):
    params = {
        "latitude": node_meta["lat"],
        "longitude": node_meta["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(REQUIRED_VARIABLES),
        "timezone": "Asia/Jakarta",
    }

    node_name = node_meta["raw_node_id"]
    for attempt in range(max_retries):
        try:
            print(f"Fetching {node_name} (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(BASE_URL, params=params, timeout=25)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "").strip()
                if retry_after.isdigit():
                    sleep_s = max(1, int(retry_after))
                else:
                    sleep_s = 4 * (attempt + 1)
                print(f"RATE LIMIT {node_name}: HTTP 429, sleeping {sleep_s}s...")
                time.sleep(sleep_s)
                continue
            if response.status_code >= 500:
                sleep_s = 3 * (attempt + 1)
                print(
                    f"SERVER ERROR {node_name}: HTTP {response.status_code}, "
                    f"retry in {sleep_s}s..."
                )
                time.sleep(sleep_s)
                continue
            if response.status_code != 200:
                print(f"FAILED {node_name}: HTTP {response.status_code}")
            response.raise_for_status()

            payload = response.json()
            daily = payload.get("daily")
            if not daily:
                print(f"WARNING: No daily data for {node_name}.")
                continue

            df_data = {
                "time": daily["time"],
                "precipitation_sum": daily["precipitation_sum"],
                "et0_fao_evapotranspiration": daily["et0_fao_evapotranspiration"],
                "soil_moisture": daily["soil_moisture_0_to_7cm_mean"],
                "temperature_2m_max": daily["temperature_2m_max"],
                "temperature_2m_min": daily["temperature_2m_min"],
                "relative_humidity_2m_mean": daily["relative_humidity_2m_mean"],
                "shortwave_radiation_sum": daily["shortwave_radiation_sum"],
                "wind_speed_10m_mean": daily["wind_speed_10m_mean"],
            }
            lengths = {k: len(v) for k, v in df_data.items()}
            if len(set(lengths.values())) > 1:
                min_len = min(lengths.values())
                print(f"Length mismatch in {node_name}: {lengths}. Trimming to {min_len}.")
                for key in df_data:
                    df_data[key] = df_data[key][:min_len]

            df = pd.DataFrame(df_data)
            df["schema_version"] = SCHEMA_VERSION
            df["city_id"] = node_meta["city_id"]
            df["node_local_id"] = node_meta["node_local_id"]
            df["node_id"] = node_meta["node_id"]
            df["raw_node_id"] = node_meta["raw_node_id"]
            # Keep backward-compatibility column but ensure uniqueness.
            df["location_id"] = node_meta["raw_node_id"]
            df["lat"] = node_meta["lat"]
            df["lon"] = node_meta["lon"]
            df["distance_km_from_city_center"] = node_meta[
                "distance_km_from_city_center"
            ]
            df["elevation"] = payload.get("elevation", 0.0)
            return df

        except Exception as exc:
            print(f"Error fetching {node_name}: {exc}")
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch data for {node_name} after {max_retries} attempts")


def _validate_city_coverage(df, city_centers, min_nodes_per_city):
    required_cities = set(city_centers.keys())
    present_cities = set(df["city_id"].dropna().unique().tolist())
    missing_cities = sorted(required_cities - present_cities)

    node_counts = (
        df[["city_id", "node_id"]]
        .drop_duplicates()
        .groupby("city_id")["node_id"]
        .nunique()
        .to_dict()
    )
    insufficient_nodes = {
        city_id: int(node_counts.get(city_id, 0))
        for city_id in sorted(required_cities)
        if int(node_counts.get(city_id, 0)) < int(min_nodes_per_city)
    }
    return missing_cities, insufficient_nodes


def main(
    output_path="data/raw/weather_history_east_java.parquet",
    city_config_path=DEFAULT_CITY_CONFIG_PATH,
    min_nodes_per_city=5,
    strict_coverage=True,
    resume_existing=True,
    persist_partial=True,
    max_retries=6,
    request_delay=1.5,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    city_centers = load_city_centers(city_config_path)

    all_nodes = []
    for city_id, coords in sorted(city_centers.items(), key=lambda x: x[0]):
        all_nodes.extend(build_city_nodes(city_id, coords["lat"], coords["lon"]))

    existing_df = None
    fetched_nodes = set()
    if resume_existing and os.path.exists(output_path):
        try:
            existing_df = pd.read_parquet(output_path)
            if (
                "schema_version" in existing_df.columns
                and sorted(existing_df["schema_version"].dropna().unique().tolist()) == [SCHEMA_VERSION]
                and {"raw_node_id", "city_id", "node_id", "time"}.issubset(existing_df.columns)
            ):
                existing_df["time"] = pd.to_datetime(existing_df["time"])
                fetched_nodes = set(existing_df["raw_node_id"].dropna().unique().tolist())
                print(
                    f"Resume mode: found {len(fetched_nodes)} existing nodes in {output_path}. "
                    "Will fetch missing nodes only."
                )
            else:
                existing_df = None
        except Exception as exc:
            print(f"WARNING: failed reading existing raw dataset for resume: {exc}")
            existing_df = None

    all_dfs = []
    for node_meta in all_nodes:
        if node_meta["raw_node_id"] in fetched_nodes:
            continue
        try:
            df_node = fetch_node_data(node_meta, max_retries=max_retries)
            all_dfs.append(df_node)
            print(f"Fetched {len(df_node)} rows for {node_meta['raw_node_id']}")
            time.sleep(request_delay)
        except Exception as exc:
            print(f"CRITICAL: skipping {node_meta['raw_node_id']} due to {exc}")

    if not all_dfs and existing_df is None:
        raise RuntimeError("No data fetched. Aborting.")

    parts = []
    if existing_df is not None:
        parts.append(existing_df)
    if all_dfs:
        parts.extend(all_dfs)

    full_df = pd.concat(parts, ignore_index=True)
    full_df["time"] = pd.to_datetime(full_df["time"])
    full_df = full_df.sort_values(["raw_node_id", "time"]).drop_duplicates(
        subset=["raw_node_id", "time"], keep="last"
    )

    # Core validations for schema v2.
    required_cols = {
        "schema_version",
        "city_id",
        "node_id",
        "raw_node_id",
        "node_local_id",
        "location_id",
        "lat",
        "lon",
        "time",
    }
    missing = required_cols - set(full_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw output: {sorted(missing)}")

    node_registry = full_df[["city_id", "node_local_id", "node_id", "lat", "lon"]].drop_duplicates()
    if full_df["node_id"].nunique() != len(node_registry[["city_id", "node_local_id"]].drop_duplicates()):
        raise ValueError("node_id uniqueness check failed.")
    if node_registry["node_id"].duplicated().any():
        raise ValueError("Duplicate node_id detected in node registry.")
    if node_registry.duplicated(subset=["city_id", "lat", "lon"]).any():
        raise ValueError("Duplicate (city_id, lat, lon) detected in node registry.")

    dup_keys = full_df.duplicated(subset=["raw_node_id", "time"]).sum()
    if dup_keys > 0:
        raise ValueError(f"Duplicate (raw_node_id, time) rows found: {dup_keys}")
    dup_node_time = full_df.duplicated(subset=["node_id", "time"]).sum()
    if dup_node_time > 0:
        raise ValueError(f"Duplicate (node_id, time) rows found: {dup_node_time}")

    missing_cities, insufficient_nodes = _validate_city_coverage(
        full_df, city_centers=city_centers, min_nodes_per_city=min_nodes_per_city
    )

    print("\nData Validation:")
    print(f"Rows: {len(full_df):,}")
    print(f"Cities: {full_df['city_id'].nunique()}")
    print(f"Raw nodes: {full_df['raw_node_id'].nunique()}")
    print(full_df.groupby(["city_id", "node_local_id"]).size().head())
    print("Coverage node counts per city:")
    print(
        full_df[["city_id", "node_id"]]
        .drop_duplicates()
        .groupby("city_id")["node_id"]
        .nunique()
        .sort_index()
    )

    coverage_ok = not missing_cities and not insufficient_nodes
    if not coverage_ok:
        print(
            "WARNING: incomplete city coverage detected. "
            f"missing_cities={missing_cities}, insufficient_nodes={insufficient_nodes}"
        )

    full_df.to_parquet(output_path, index=False)
    print(f"\nSaved raw dataset (schema v{SCHEMA_VERSION}) to {output_path}")

    if strict_coverage and not coverage_ok:
        if not persist_partial and os.path.exists(output_path):
            os.remove(output_path)
        raise ValueError(
            "Raw dataset coverage validation failed: "
            f"missing_cities={missing_cities}, insufficient_nodes={insufficient_nodes}. "
            f"Each configured city must have >= {min_nodes_per_city} nodes."
        )


if __name__ == "__main__":
    main()
