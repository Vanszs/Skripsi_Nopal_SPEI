import os
import sys
import argparse
import json

import pandas as pd

# Ensure src is in path
sys.path.append(os.getcwd())

from src.data.ingest import main as ingest_data
from src.data.preprocess import preprocess_pipeline
from src.training.train import train_pipeline

EXPECTED_SCHEMA_VERSION = 2
RAW_PATH = "data/raw/weather_history_east_java.parquet"
PROCESSED_PATH = "data/processed/spei_dataset.parquet"


def _is_schema_v2(path, required_cols):
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:
        return False
    if "schema_version" not in df.columns:
        return False
    versions = sorted(df["schema_version"].dropna().unique().tolist())
    if versions != [EXPECTED_SCHEMA_VERSION]:
        return False
    return required_cols.issubset(set(df.columns))


def _load_city_registry(city_config_path):
    if not os.path.exists(city_config_path):
        raise FileNotFoundError(f"City config not found: {city_config_path}")
    with open(city_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("City config must be a non-empty object")
    return sorted(cfg.keys())


def _raw_has_required_coverage(path, city_config_path, min_nodes_per_city):
    if not os.path.exists(path):
        return False, {"reason": "raw_missing"}
    try:
        df = pd.read_parquet(path, columns=["city_id", "node_id"])
    except Exception as exc:
        return False, {"reason": f"raw_read_error: {exc}"}

    required_cities = set(_load_city_registry(city_config_path))
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

    ok = not missing_cities and not insufficient_nodes
    return ok, {
        "missing_cities": missing_cities,
        "insufficient_nodes": insufficient_nodes,
        "node_counts": {k: int(v) for k, v in node_counts.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Run schema-v2 pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--max-epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    parser.add_argument(
        "--city-config",
        type=str,
        default="data/config/city_centers.json",
        help="City center config path",
    )
    parser.add_argument(
        "--min-nodes-per-city",
        type=int,
        default=5,
        help="Minimum fetched nodes required for each configured city",
    )
    parser.add_argument(
        "--no-resume-ingest",
        action="store_true",
        help="Disable ingest resume mode and fetch all nodes from scratch",
    )
    args = parser.parse_args()

    print("=== STARTING PIPELINE ===")

    print("\n--- STEP 1: INGESTION ---")
    raw_required = {"schema_version", "city_id", "node_id", "raw_node_id", "lat", "lon"}
    try:
        schema_ok = _is_schema_v2(RAW_PATH, raw_required)
        coverage_ok, coverage_detail = _raw_has_required_coverage(
            RAW_PATH, args.city_config, args.min_nodes_per_city
        )
        if schema_ok and coverage_ok:
            print("Raw data schema v2 + coverage check passed. Skipping ingestion.")
        else:
            print(
                "Raw data missing/stale schema or incomplete coverage. "
                "Re-ingesting (schema v2)..."
            )
            if not coverage_ok:
                print(f"Coverage detail before re-ingest: {coverage_detail}")
            ingest_data(
                output_path=RAW_PATH,
                city_config_path=args.city_config,
                min_nodes_per_city=args.min_nodes_per_city,
                strict_coverage=True,
                resume_existing=not args.no_resume_ingest,
            )
            schema_ok = _is_schema_v2(RAW_PATH, raw_required)
            coverage_ok, coverage_detail = _raw_has_required_coverage(
                RAW_PATH, args.city_config, args.min_nodes_per_city
            )
            if not schema_ok or not coverage_ok:
                raise RuntimeError(
                    "Raw ingestion finished but validation failed: "
                    f"schema_ok={schema_ok}, coverage={coverage_detail}"
                )
    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        return

    print("\n--- STEP 2: PREPROCESSING ---")
    processed_required = {"schema_version", "city_id", "super_node_id", "selected_node_count"}
    try:
        if _is_schema_v2(PROCESSED_PATH, processed_required):
            print("Processed data schema v2 found. Rebuilding anyway for fresh selection artifact.")
        preprocess_pipeline(
            input_path=RAW_PATH,
            output_path=PROCESSED_PATH,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Preprocessing failed: {exc}")
        return

    print("\n--- STEP 3: TRAINING ---")
    if args.skip_train:
        print("Training skipped by --skip-train")
    else:
        try:
            best_model_path = train_pipeline(
                data_path=PROCESSED_PATH,
                max_epochs=args.max_epochs,
                seed=args.seed,
            )
            print(f"Training completed. Model saved at {best_model_path}")
        except Exception as exc:
            print(f"Training failed: {exc}")
            return

    print("\n=== PIPELINE FINISHED ===")
    print("Now run evaluation: python full_evaluation.py")


if __name__ == "__main__":
    main()
