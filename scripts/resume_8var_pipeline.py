"""
Resume the 8-variable production pipeline after the Open-Meteo daily quota resets.

Run:  python3 scripts/resume_8var_pipeline.py

Steps (idempotent / resumable):
  1. Ingest remaining nodes (resume mode skips the 19 already fetched).
  2. Preprocess -> data/processed/spei_dataset.parquet (C1 train-only SPEI, 8 vars).
  3. Train enc=90 on GPU (bf16-mixed) + full evaluation + MD report.
  4. Run test_pipeline.py.

Requires CUDA. The 3 new vars are already wired into ingest/preprocess/dataset.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/weather_history_east_java.parquet"
CITY_CFG = ROOT / "data/config/city_centers.json"
PY = sys.executable


def _run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    from src.data.ingest import main as ingest

    print("STEP 1: ingest (resume) until all 5 cities have >=5 nodes ...")
    ingest(
        output_path=str(RAW),
        city_config_path=str(CITY_CFG),
        min_nodes_per_city=5,
        strict_coverage=True,
        resume_existing=True,
        max_retries=8,
    )

    df = pd.read_parquet(RAW)
    need = {"relative_humidity_2m_mean", "shortwave_radiation_sum", "wind_speed_10m_mean"}
    assert need.issubset(df.columns), f"raw missing new vars: {need - set(df.columns)}"
    cov = df[["city_id", "raw_node_id"]].drop_duplicates().groupby("city_id")["raw_node_id"].nunique()
    print(cov)
    assert (cov >= 5).all() and df["city_id"].nunique() == 5, "coverage incomplete; re-run after quota reset"

    print("STEP 2: preprocess ...")
    from src.data.preprocess import preprocess_pipeline
    preprocess_pipeline(input_path=str(RAW))

    print("STEP 3: train enc=90 + evaluate + report ...")
    _run([PY, "run_experiment.py", "--encoder", "90", "--epochs", "60"])

    print("STEP 4: test_pipeline.py ...")
    _run([PY, "test_pipeline.py"])

    print("\nRESUME PIPELINE COMPLETE.")


if __name__ == "__main__":
    main()
