"""
Schema-v2 smoke test:
1) build synthetic raw node-level dataset (>5 cities)
2) run preprocess_pipeline
3) validate grouping key uniqueness and cardinality
4) optional 1-epoch training run
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import preprocess_pipeline
from src.models.dataset import MODEL_GROUP_COL, create_dataset
from src.training.train import train_pipeline

TOP_K = 5  # Number of selected nodes per city in smoke fixture


def generate_synthetic_raw(path: Path, seed: int = 42):
    rng = np.random.default_rng(seed)
    cities = [f"City_{i:02d}" for i in range(8)]  # >5 super-nodes after aggregation
    node_local_ids = [f"n{i:02d}" for i in range(6)]  # candidate nodes per city
    dates = pd.date_range("2021-01-01", "2025-12-31", freq="D")

    rows = []
    for c_idx, city in enumerate(cities):
        city_lat = -7.5 + 0.1 * c_idx
        city_lon = 111.0 + 0.2 * c_idx
        for n_idx, local in enumerate(node_local_ids):
            node_id = f"{city}__{local}"
            lat = city_lat + (n_idx - 3) * 0.03
            lon = city_lon + (n_idx - 3) * 0.03
            elev = 20 + 5 * n_idx + c_idx
            for t in dates:
                day = t.dayofyear
                rows.append(
                    {
                        "schema_version": 2,
                        "time": t,
                        "city_id": city,
                        "node_local_id": local,
                        "node_id": node_id,
                        "raw_node_id": node_id,
                        "location_id": node_id,
                        "lat": lat,
                        "lon": lon,
                        "distance_km_from_city_center": float(abs(n_idx - 3) * 2.1),
                        "elevation": float(elev),
                        "precipitation_sum": float(8 + 3 * np.sin(day / 30) + rng.normal(0, 0.7)),
                        "et0_fao_evapotranspiration": float(
                            4 + 1.2 * np.cos(day / 40) + rng.normal(0, 0.3)
                        ),
                        "soil_moisture": float(0.28 + 0.04 * np.sin(day / 20) + rng.normal(0, 0.01)),
                        "temperature_2m_max": float(31 + 2 * np.sin(day / 50) + rng.normal(0, 0.5)),
                        "temperature_2m_min": float(22 + 1.5 * np.cos(day / 45) + rng.normal(0, 0.4)),
                        "relative_humidity_2m_mean": float(78 + 8 * np.sin(day / 35) + rng.normal(0, 1.0)),
                        "shortwave_radiation_sum": float(18 + 4 * np.cos(day / 38) + rng.normal(0, 0.6)),
                        "wind_speed_10m_mean": float(9 + 2 * np.sin(day / 42) + rng.normal(0, 0.4)),
                    }
                )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", action="store_true", help="Run 1-epoch training smoke")
    args = parser.parse_args()

    out_dir = Path("results/smoke_v2")
    raw_path = out_dir / "raw_schema_v2.parquet"
    proc_path = out_dir / "processed_schema_v2.parquet"
    sel_path = out_dir / "node_selection_v2.parquet"
    sel_meta = out_dir / "node_selection_v2.meta.json"
    report_path = out_dir / "smoke_report.json"

    raw = generate_synthetic_raw(raw_path, seed=args.seed)
    processed = preprocess_pipeline(
        input_path=str(raw_path),
        output_path=str(proc_path),
        selection_artifact_path=str(sel_path),
        selection_metadata_path=str(sel_meta),
        top_k=TOP_K,
        seed=args.seed,
    )

    processed["year"] = pd.to_datetime(processed["time"]).dt.year
    train_data = processed[processed["year"] < 2023].copy()
    ds = create_dataset(train_data)

    checks = {
        "raw_node_id_time_unique": int(raw.duplicated(subset=["node_id", "time"]).sum()) == 0,
        "processed_group_time_unique": int(processed.duplicated(subset=[MODEL_GROUP_COL, "time"]).sum()) == 0,
        "selected_node_count_eq_5": bool((processed["selected_node_count"] == TOP_K).all()),
        # SYNTHETIC smoke fixture uses 8 cities (distinct from production's 5) to ensure >5 entities
        "entity_count_gt_5": int(processed[MODEL_GROUP_COL].nunique()) > 5,
        "dataset_sequence_count_gt_0": len(ds) > 0,
        "selection_metadata_exists": sel_meta.exists(),
    }

    train_ckpt = None
    if args.train:
        train_ckpt = train_pipeline(
            data_path=str(proc_path),
            max_epochs=1,
            batch_size=64,
            seed=args.seed,
            run_config_path=str(out_dir / "run_config.json"),
        )
        checks["train_checkpoint_exists"] = bool(train_ckpt) and os.path.exists(train_ckpt)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "raw_rows": int(len(raw)),
        "processed_rows": int(len(processed)),
        "entities": int(processed[MODEL_GROUP_COL].nunique()),
        "cities": int(processed["city_id"].nunique()),
        "checks": checks,
        "all_pass": all(checks.values()),
        "checkpoint": train_ckpt,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
