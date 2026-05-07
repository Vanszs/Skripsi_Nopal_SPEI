import os
import json
import hashlib
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

from .spei import calculate_spei, calculate_water_deficit

SCHEMA_VERSION = 2
SELECTION_END_DATE = "2022-12-31"
DEFAULT_TOP_K = 5
DEFAULT_SEED = 42

WEATHER_COLS = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "soil_moisture",
    "temperature_2m_max",
    "temperature_2m_min",
]


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    )
    return 2 * radius * asin(sqrt(a))


def _validate_raw_schema(df):
    required = {
        "schema_version",
        "time",
        "city_id",
        "node_id",
        "raw_node_id",
        "location_id",
        "lat",
        "lon",
        "elevation",
        *WEATHER_COLS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw schema mismatch, missing: {sorted(missing)}")
    versions = sorted(pd.Series(df["schema_version"]).dropna().unique().tolist())
    if versions != [SCHEMA_VERSION]:
        raise ValueError(
            f"Raw schema version must be {SCHEMA_VERSION}; found {versions}. "
            "Re-run ingest.py to regenerate schema v2."
        )


def _interpolate_per_node(df):
    dfs = []
    for raw_node_id, group in df.groupby("raw_node_id", sort=True):
        group = group.sort_values("time").copy()
        # Forward-only fill to avoid leaking future values into historical rows.
        group[WEATHER_COLS] = group[WEATHER_COLS].ffill()
        dfs.append(group)
    out = pd.concat(dfs, ignore_index=True)
    return out


def _compute_similarity(df_train):
    rows = []
    for city_id, city_df in df_train.groupby("city_id", sort=True):
        city_center_lat = city_df.groupby("raw_node_id", sort=True)["lat"].first().mean()
        city_center_lon = city_df.groupby("raw_node_id", sort=True)["lon"].first().mean()

        for raw_node_id, node_df in city_df.groupby("raw_node_id", sort=True):
            node_profile = node_df.set_index("time")[WEATHER_COLS].sort_index()
            others = city_df[city_df["raw_node_id"] != raw_node_id]
            # Leave-one-node-out city profile prevents circular comparison leakage.
            if len(others) > 0:
                city_profile = others.groupby("time")[WEATHER_COLS].mean().sort_index()
            else:
                city_profile = city_df.groupby("time")[WEATHER_COLS].mean().sort_index()
            aligned = city_profile.join(
                node_profile, how="inner", lsuffix="_others", rsuffix="_node"
            ).dropna()

            corr_scores = []
            for col in WEATHER_COLS:
                a = aligned[f"{col}_others"].values
                b = aligned[f"{col}_node"].values
                if len(a) < 10:
                    continue
                if np.std(a) == 0 or np.std(b) == 0:
                    continue
                corr_scores.append(float(np.corrcoef(a, b)[0, 1]))

            behavior_score = float(np.nanmean(corr_scores)) if corr_scores else -1.0
            lat = float(node_df["lat"].iloc[0])
            lon = float(node_df["lon"].iloc[0])
            dist_km = haversine_km(city_center_lat, city_center_lon, lat, lon)
            distance_score = 1.0 / (1.0 + dist_km)
            hybrid_score = 0.7 * behavior_score + 0.3 * distance_score

            rows.append(
                {
                    "city_id": city_id,
                    "raw_node_id": raw_node_id,
                    "node_id": node_df["node_id"].iloc[0],
                    "lat": lat,
                    "lon": lon,
                    "behavior_score": behavior_score,
                    "distance_km": dist_km,
                    "distance_score": distance_score,
                    "hybrid_score": hybrid_score,
                }
            )
    return pd.DataFrame(rows)


def _select_top_k_nodes(similarity_df, top_k):
    selected = []
    for city_id, group in similarity_df.groupby("city_id", sort=True):
        group = group.copy()
        # Clamp floating noise before sorting to stabilize tie behavior across reruns.
        for c in ["hybrid_score", "behavior_score", "distance_score"]:
            group[c] = group[c].round(12)
        # Deterministic sorting with explicit tie-breakers.
        group_sorted = group.sort_values(
            by=["hybrid_score", "behavior_score", "distance_score", "raw_node_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        chosen = group_sorted.head(top_k).copy()
        chosen["selected_rank"] = np.arange(1, len(chosen) + 1)
        chosen["selected_flag"] = True
        selected.append(chosen)
    selected_df = pd.concat(selected, ignore_index=True)
    return selected_df


def preprocess_pipeline(
    input_path="data/raw/weather_history_east_java.parquet",
    output_path="data/processed/spei_dataset.parquet",
    selection_artifact_path="data/processed/node_selection_v2.parquet",
    selection_metadata_path="data/processed/node_selection_v2.meta.json",
    top_k=DEFAULT_TOP_K,
    selection_end_date=SELECTION_END_DATE,
    seed=DEFAULT_SEED,
):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("Loading raw data...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data not found at {input_path}. Run ingest.py first.")

    df = pd.read_parquet(input_path)
    df["time"] = pd.to_datetime(df["time"])
    _validate_raw_schema(df)
    df = df.sort_values(["city_id", "raw_node_id", "time"]).reset_index(drop=True)

    dup = df.duplicated(subset=["raw_node_id", "time"]).sum()
    if dup:
        raise ValueError(f"Duplicate (raw_node_id,time) rows found: {dup}")
    if df.duplicated(subset=["node_id", "time"]).sum():
        raise ValueError("Duplicate (node_id,time) rows found in raw data.")
    if df[["node_id"]].drop_duplicates()["node_id"].duplicated().any():
        raise ValueError("node_id must be globally unique.")

    print("Interpolating weather variables per raw node...")
    df = _interpolate_per_node(df)

    # Train-only node selection to avoid leakage from val/test periods.
    cutoff = pd.Timestamp(selection_end_date)
    train_slice = df[df["time"] <= cutoff].copy()
    if train_slice.empty:
        raise ValueError("Train-only slice for node selection is empty.")
    if train_slice["time"].max() > cutoff:
        raise ValueError("Train slice exceeds selection_end_date boundary.")

    print(f"Computing train-only similarity (<= {cutoff.date()})...")
    similarity_df = _compute_similarity(train_slice)
    selected_nodes = _select_top_k_nodes(similarity_df, top_k=top_k)

    # Enforce exactly top_k selected nodes per city.
    counts = selected_nodes.groupby("city_id")["raw_node_id"].nunique()
    bad = counts[counts != top_k]
    if len(bad) > 0:
        raise ValueError(
            "Node selection failed to produce exactly "
            f"{top_k} nodes for cities: {bad.to_dict()}"
        )

    os.makedirs(os.path.dirname(selection_artifact_path), exist_ok=True)
    selected_nodes["selection_end_date"] = str(cutoff.date())
    selected_nodes["top_k"] = int(top_k)
    selected_nodes["schema_version"] = SCHEMA_VERSION
    selected_nodes = selected_nodes.sort_values(["city_id", "selected_rank", "raw_node_id"]).reset_index(drop=True)
    selected_nodes.to_parquet(selection_artifact_path, index=False)
    print(f"Saved node selection artifact to {selection_artifact_path}")

    # Repro metadata with input fingerprint.
    raw_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(
            df[["time", "city_id", "raw_node_id", "node_id", *WEATHER_COLS]], index=False
        ).values.tobytes()
    ).hexdigest()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "selection_end_date": str(cutoff.date()),
        "top_k": int(top_k),
        "seed": int(seed),
        "selection_method": "hybrid_leave_one_out",
        "sort_tie_breakers": ["hybrid_score desc", "behavior_score desc", "distance_score desc", "raw_node_id asc"],
        "raw_fingerprint_sha256": raw_fingerprint,
        "selected_nodes_fingerprint_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(
                selected_nodes[
                    [
                        "city_id",
                        "raw_node_id",
                        "selected_rank",
                        "hybrid_score",
                        "behavior_score",
                        "distance_score",
                    ]
                ],
                index=False,
            ).values.tobytes()
        ).hexdigest(),
    }
    with open(selection_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved selection metadata to {selection_metadata_path}")

    print("Aggregating selected nodes into city super-nodes...")
    selected_keys = set(selected_nodes["raw_node_id"].tolist())
    df_selected = df[df["raw_node_id"].isin(selected_keys)].copy()

    # Aggregate weather signals per (city, time).
    agg_map = {col: "mean" for col in WEATHER_COLS}
    agg_map.update(
        {
            "elevation": "mean",
            "lat": "mean",
            "lon": "mean",
        }
    )
    city_daily = (
        df_selected.groupby(["city_id", "time"], as_index=False)
        .agg(agg_map)
        .sort_values(["city_id", "time"])
        .reset_index(drop=True)
    )

    city_daily["selected_node_count"] = city_daily["city_id"].map(counts).astype(int)
    city_daily["schema_version"] = SCHEMA_VERSION
    city_daily["super_node_id"] = city_daily["city_id"].apply(lambda c: f"SN_{c}")
    # Compatibility alias for legacy code paths.
    city_daily["location_id"] = city_daily["super_node_id"]

    print("Calculating SPEI features per super-node...")
    processed = []
    for super_node_id, group in city_daily.groupby("super_node_id"):
        group = group.sort_values("time").copy()
        group["water_deficit"] = calculate_water_deficit(group)
        indexed = group.set_index("time")
        group["SPEI_3"] = calculate_spei(indexed["water_deficit"], scale=3).values
        group["SPEI_6"] = calculate_spei(indexed["water_deficit"], scale=6).values
        group["SPEI_3_diff"] = group["SPEI_3"].diff().fillna(0.0)
        processed.append(group)
    df_processed = pd.concat(processed, ignore_index=True)

    print("Engineering temporal features...")
    df_processed["time_idx"] = (df_processed["time"] - df_processed["time"].min()).dt.days
    df_processed["month"] = df_processed["time"].dt.month
    df_processed["month_sin"] = np.sin(2 * np.pi * df_processed["month"] / 12)
    df_processed["month_cos"] = np.cos(2 * np.pi * df_processed["month"] / 12)
    df_processed["precipitation_log"] = np.log1p(df_processed["precipitation_sum"])

    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean = df_processed.dropna(subset=["SPEI_3", "SPEI_6", "water_deficit"]).reset_index(
        drop=True
    )

    # Final safety checks.
    dup_super = df_clean.duplicated(subset=["super_node_id", "time"]).sum()
    if dup_super:
        raise ValueError(f"Duplicate (super_node_id,time) rows found: {dup_super}")

    if not (df_clean["selected_node_count"] == top_k).all():
        raise ValueError("selected_node_count is inconsistent with requested top_k.")

    nan_counts = df_clean.isna().sum()
    if nan_counts.sum() > 0:
        print("WARNING: Remaining NaNs found, forcing dropna.")
        print(nan_counts[nan_counts > 0])
        df_clean = df_clean.dropna().reset_index(drop=True)

    print(f"Data Cleaned. Rows: {len(df_processed)} -> {len(df_clean)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    print(f"Processed data saved to {output_path}. Shape: {df_clean.shape}")
    print(
        "Super-nodes:",
        sorted(df_clean["super_node_id"].unique().tolist()),
        f"(n={df_clean['super_node_id'].nunique()})",
    )

    return df_clean


if __name__ == "__main__":
    preprocess_pipeline()
