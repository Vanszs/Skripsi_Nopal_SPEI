"""
End-to-end 8-variable production pipeline: ingest -> preprocess -> train -> eval -> test.

Run:  python3 scripts/run_full_pipeline.py [--wait-on-quota] [--epochs 60] [--allow-cpu]

Idempotent/resumable: ingest resumes from already-fetched nodes. If the Open-Meteo
DAILY quota is exhausted (HTTP 429 "try again tomorrow"), ingest cannot finish today.
With --wait-on-quota the script sleeps and retries until coverage is complete;
otherwise it exits cleanly so you can re-run after the quota resets.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAW = ROOT / "data/raw/weather_history_east_java.parquet"
CITY_CFG = ROOT / "data/config/city_centers.json"
PROC = ROOT / "data/processed/spei_dataset.parquet"
NEW_VARS = {"relative_humidity_2m_mean", "shortwave_radiation_sum", "wind_speed_10m_mean"}
PY = sys.executable


def _run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _quota_blocked() -> bool:
    """Return True if Open-Meteo daily quota is currently exhausted."""
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": -7.1, "longitude": 112.3,
                    "start_date": "2024-01-01", "end_date": "2024-01-02",
                    "daily": "precipitation_sum", "timezone": "Asia/Jakarta"},
            timeout=20,
        )
        return r.status_code == 429
    except requests.RequestException:
        return False  # network error != quota; let ingest handle/retry


def _coverage_complete() -> bool:
    if not RAW.exists():
        return False
    df = pd.read_parquet(RAW)
    if not NEW_VARS.issubset(df.columns):
        return False
    n_cities = len(pd.read_json(CITY_CFG, typ="series"))
    cov = df[["city_id", "raw_node_id"]].drop_duplicates().groupby("city_id")["raw_node_id"].nunique()
    return df["city_id"].nunique() == n_cities and (cov >= 5).all()


def ingest_until_complete(wait_on_quota: bool, retry_minutes: int, request_delay: float):
    from src.data.ingest import main as ingest
    while True:
        if _quota_blocked() and not _coverage_complete():
            msg = "Open-Meteo daily quota exhausted (HTTP 429)."
            if not wait_on_quota:
                print(f"{msg} Re-run after reset, or use --wait-on-quota. Exiting.")
                sys.exit(2)
            print(f"{msg} Sleeping {retry_minutes} min then retrying ...", flush=True)
            time.sleep(retry_minutes * 60)
            continue
        try:
            ingest(
                output_path=str(RAW),
                city_config_path=str(CITY_CFG),
                min_nodes_per_city=5,
                strict_coverage=True,
                resume_existing=True,
                persist_partial=True,
                max_retries=8,
                request_delay=request_delay,
            )
        except Exception as exc:  # coverage/quota failure mid-run -> persist_partial kept progress
            print(f"Ingest incomplete: {exc}")
            if not wait_on_quota:
                sys.exit(2)
            print(f"Sleeping {retry_minutes} min then retrying ...", flush=True)
            time.sleep(retry_minutes * 60)
            continue
        if _coverage_complete():
            return
        if not wait_on_quota:
            print("Coverage still incomplete. Re-run after quota reset. Exiting.")
            sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-on-quota", action="store_true",
                    help="sleep + retry until quota resets instead of exiting")
    ap.add_argument("--retry-minutes", type=int, default=60)
    ap.add_argument("--request-delay", type=float, default=3.0,
                    help="seconds to wait between node fetches (gentler on API)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--encoder", type=int, default=90)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    print("STEP 1/4: ingest (8-var, resume) ...")
    ingest_until_complete(args.wait_on_quota, args.retry_minutes, args.request_delay)
    print("Coverage complete.")

    print("STEP 2/4: preprocess (C1 train-only SPEI, 8 vars) ...")
    from src.data.preprocess import preprocess_pipeline
    preprocess_pipeline(input_path=str(RAW))
    df = pd.read_parquet(PROC)
    assert NEW_VARS.issubset(df.columns), f"processed missing new vars: {NEW_VARS - set(df.columns)}"
    assert "SPEI_6" not in df.columns, "SPEI_6 must be dropped"
    assert df["super_node_id"].nunique() == df["city_id"].nunique(), "super-node != city count"

    print("STEP 3/4: train enc=90 (GPU bf16) + evaluate + report ...")
    cmd = [PY, "run_experiment.py", "--encoder", str(args.encoder), "--epochs", str(args.epochs)]
    _run(cmd)

    print("STEP 4/4: test_pipeline.py ...")
    _run([PY, "test_pipeline.py"])

    print("\nFULL PIPELINE COMPLETE.")


if __name__ == "__main__":
    main()
