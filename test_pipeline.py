"""
Pipeline Test Script â€” validates all stages without re-training.
Run: python test_pipeline.py
"""
import sys
import os
import json
import traceback
import pandas as pd
import numpy as np

sys.path.insert(0, ".")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
EXPECTED_SCHEMA_VERSION = 2
PROCESSED_DATA_PATH = None


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _is_schema_v2(path):
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_parquet(path, columns=["schema_version", "city_id", "super_node_id"])
    except Exception:
        return False
    versions = sorted(pd.Series(df["schema_version"]).dropna().unique().tolist())
    return versions == [EXPECTED_SCHEMA_VERSION]


def _resolve_processed_data_path():
    primary = "data/processed/spei_dataset.parquet"
    smoke = "results/smoke_v2/processed_schema_v2.parquet"
    if _is_schema_v2(primary):
        return primary
    if _is_schema_v2(smoke):
        print(f"{WARN} Primary processed dataset is not schema v2. Using smoke dataset: {smoke}")
        return smoke
    raise FileNotFoundError(
        "No schema-v2 processed dataset found. Run `python main.py` "
        "or `python scripts/smoke_e2e_v2.py` first."
    )


def _resolve_checkpoint_path(processed_data_path):
    run_cfg = "logs/run_config.json"
    if os.path.exists(run_cfg):
        try:
            with open(run_cfg, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ckpt = payload.get("best_model_path")
            if ckpt and os.path.exists(ckpt):
                return ckpt
        except Exception:
            pass

    smoke_report = "results/smoke_v2/smoke_report.json"
    if processed_data_path and os.path.normpath("results/smoke_v2") in os.path.normpath(processed_data_path):
        if os.path.exists(smoke_report):
            try:
                with open(smoke_report, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                ckpt = payload.get("checkpoint")
                if ckpt and os.path.exists(ckpt):
                    return ckpt
            except Exception:
                pass

    ckpt_dir = "logs/checkpoints"
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in logs/checkpoints")

    def parse_val_loss(fname):
        try:
            return float(fname.split("val_loss=")[1].replace(".ckpt", ""))
        except Exception:
            return float("inf")

    return os.path.join(ckpt_dir, min(ckpts, key=parse_val_loss))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TEST 1 â€” Imports
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("TEST 1: Imports")
try:
    from src.data.spei import calculate_water_deficit, calculate_spei, classify_spei
    from src.data.preprocess import preprocess_pipeline
    from src.models.dataset import (
        MODEL_GROUP_COL,
        MAX_ENCODER_LENGTH,
        MAX_PREDICTION_LENGTH,
        create_dataset,
    )
    from src.training.train import train_pipeline
    from src.evaluation.metrics import load_model, calculate_metrics
    print(f"{PASS} All src imports OK")
    print(f"{PASS} MAX_ENCODER_LENGTH = {MAX_ENCODER_LENGTH}  (expected 90)")
    print(f"{PASS} MAX_PREDICTION_LENGTH = {MAX_PREDICTION_LENGTH}  (expected 30)")
    assert MAX_ENCODER_LENGTH == 90, "MAX_ENCODER_LENGTH should be 90!"
    assert MAX_PREDICTION_LENGTH == 30, "MAX_PREDICTION_LENGTH should be 30!"
except Exception as e:
    print(f"{FAIL} Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TEST 2 â€” classify_spei canonical thresholds
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("TEST 2: classify_spei thresholds")
cases = [
    (-2.5, "Kekeringan Ekstrem"),
    (-1.7, "Kekeringan Parah"),
    (-1.2, "Kekeringan Sedang"),
    (-0.7, "Kekeringan Ringan"),
    ( 0.0, "Normal"),
    ( 0.7, "Basah Ringan"),
    ( 1.2, "Basah Sedang"),
    ( 1.7, "Basah Parah"),
    ( 2.5, "Basah Ekstrem"),
]
all_ok = True
for val, expected in cases:
    got = classify_spei(val)
    ok = got == expected
    if not ok:
        all_ok = False
    tag = PASS if ok else FAIL
    print(f"  {tag}  {val:+.1f} -> {got}  (expected: {expected})")
if all_ok:
    print(f"{PASS} All 9 SPEI classes correct")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TEST 3 â€” Processed data integrity
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("TEST 3: Processed data integrity")
try:
    PROCESSED_DATA_PATH = _resolve_processed_data_path()
    print(f"{PASS} Using processed dataset: {PROCESSED_DATA_PATH}")
    data = pd.read_parquet(PROCESSED_DATA_PATH)
    data["year"] = data["time"].dt.year

    # Required columns
    required_cols = [
        "schema_version",
        "time",
        "time_idx",
        "city_id",
        "super_node_id",
        "location_id",
        "selected_node_count",
        "elevation",
        "lat",
        "lon",
        "SPEI_3", "SPEI_3_diff", "water_deficit",
        "precipitation_log", "et0_fao_evapotranspiration",
        "soil_moisture", "temperature_2m_max", "temperature_2m_min",
        "month_sin", "month_cos",
    ]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        print(f"{FAIL} Missing columns: {missing}")
    else:
        print(f"{PASS} All required columns present ({len(required_cols)})")

    # NaN check
    nan_total = data.isna().sum().sum()
    tag = PASS if nan_total == 0 else FAIL
    print(f"  {tag}  NaN count = {nan_total}  (expected 0)")

    # Shape
    n_rows, n_cols = data.shape
    print(f"{PASS} Shape: {n_rows} rows Ã— {n_cols} cols")

    # Schema/cardinality checks (dynamic, not hardcoded city list)
    n_city = data["city_id"].nunique()
    n_entity = data[MODEL_GROUP_COL].nunique()
    tag_city = PASS if n_city >= 1 else FAIL
    tag_entity = PASS if n_entity >= 1 else FAIL
    print(f"  {tag_city}  city_id nunique = {n_city} (expected >=1)")
    print(f"  {tag_entity}  {MODEL_GROUP_COL} nunique = {n_entity} (expected >=1)")
    tag_card = PASS if n_entity == n_city else FAIL
    print(f"  {tag_card}  model entities == cities ({n_entity} == {n_city}) for 1 super-node per city")
    tag_sel = PASS if (data["selected_node_count"] == 5).all() else FAIL
    print(f"  {tag_sel}  selected_node_count == 5 for all rows")

    # Group uniqueness checks (post-aggregation)
    dup_entity_time = data.duplicated(subset=[MODEL_GROUP_COL, "time_idx"]).sum()
    tag_dup = PASS if dup_entity_time == 0 else FAIL
    print(f"  {tag_dup}  duplicate ({MODEL_GROUP_COL}, time_idx) = {dup_entity_time}")
    dup_city_time = data.duplicated(subset=["city_id", "time_idx"]).sum()
    tag_dup_city = PASS if dup_city_time == 0 else FAIL
    print(f"  {tag_dup_city}  duplicate (city_id, time_idx) = {dup_city_time}")
    tag_group_name = PASS if MODEL_GROUP_COL == "super_node_id" else FAIL
    print(f"  {tag_group_name}  MODEL_GROUP_COL == 'super_node_id'")

    # Optional raw schema-v2 uniqueness checks (pre-aggregation)
    raw_path = "data/raw/weather_history_east_java.parquet"
    if os.path.exists(raw_path):
        raw = pd.read_parquet(raw_path)
        if {"schema_version", "node_id", "raw_node_id", "time"}.issubset(raw.columns):
            dup_node_time = raw.duplicated(subset=["node_id", "time"]).sum()
            dup_raw_time = raw.duplicated(subset=["raw_node_id", "time"]).sum()
            tag_raw = PASS if (dup_node_time == 0 and dup_raw_time == 0) else FAIL
            print(f"  {tag_raw}  raw duplicate (node_id,time)={dup_node_time}, (raw_node_id,time)={dup_raw_time}")
        else:
            print(f"  {WARN}  raw file exists but not schema v2; uniqueness check skipped")

    # Reproducibility artifacts from preprocessing
    base_dir = os.path.dirname(PROCESSED_DATA_PATH) if PROCESSED_DATA_PATH else "data/processed"
    meta_path = os.path.join(base_dir, "node_selection_v2.meta.json")
    sel_path = os.path.join(base_dir, "node_selection_v2.parquet")
    tag_meta = PASS if os.path.exists(meta_path) else FAIL
    tag_sel_art = PASS if os.path.exists(sel_path) else FAIL
    print(f"  {tag_meta}  selection metadata exists: {meta_path}")
    print(f"  {tag_sel_art}  selection artifact exists: {sel_path}")

    # Date range
    print(f"{PASS} Date range: {data.time.min().date()} -> {data.time.max().date()}")

    # SPEI_3 distribution â€” should be approx N(0,1)
    s3 = data.SPEI_3
    mean_ok = abs(s3.mean()) < 0.15
    std_ok  = abs(s3.std() - 1.0) < 0.15
    tag = PASS if (mean_ok and std_ok) else WARN
    print(f"  {tag}  SPEI_3: mean={s3.mean():.4f} (|mean|<0.15?={mean_ok}), "
          f"std={s3.std():.4f} (|std-1|<0.15?={std_ok})")

    # Split sizes (no val leakage)
    n_train = len(data[data.year < 2023])
    n_val   = len(data[data.year == 2023])
    n_test  = len(data[data.year >= 2024])
    print(f"{PASS} Train(<2023)={n_train}  Val(2023)={n_val}  Test(>=2024)={n_test}")
    assert n_test > 0, "No test data!"

    # time_idx monotone per model entity
    for loc, g in data.groupby(MODEL_GROUP_COL):
        g_sorted = g.sort_values("time")
        diffs = g_sorted.time_idx.diff().dropna()
        if not (diffs >= 0).all():
            print(f"{FAIL} time_idx not monotone for {loc}")
        else:
            print(f"  {PASS}  time_idx monotone: {loc}")

except Exception as e:
    print(f"{FAIL} {e}")
    traceback.print_exc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TEST 4 â€” TimeSeriesDataSet creation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Dynamic cardinality check (>5 entities) for plotting utilities
section("TEST 3B: Dynamic cardinality (>5 entities)")
try:
    from full_evaluation import _grid, _palette

    entities = [f"ent_{i:02d}" for i in range(8)]
    rows, cols = _grid(len(entities), max_cols=3)
    pal = _palette(entities)
    tag_grid = PASS if (rows * cols >= len(entities)) else FAIL
    tag_pal = PASS if len(pal) == len(entities) else FAIL
    print(f"  {tag_grid}  dynamic grid capacity rows*cols={rows*cols} for n={len(entities)}")
    print(f"  {tag_pal}  dynamic palette size={len(pal)} for n={len(entities)}")
except Exception as e:
    print(f"{FAIL} {e}")
    traceback.print_exc()

section("TEST 4: TimeSeriesDataSet creation + dataloader")
try:
    from pytorch_forecasting import TimeSeriesDataSet
    import torch

    train_data = data[data.year < 2023].copy()
    train_ds = create_dataset(train_data)

    print(f"{PASS} TimeSeriesDataSet created: {len(train_ds)} sequences")
    print(f"  max_encoder_length : {train_ds.max_encoder_length}")
    print(f"  max_prediction_len : {train_ds.max_prediction_length}")
    print(f"  static_categoricals: {train_ds.static_categoricals}")
    print(f"  static_reals       : {train_ds.static_reals}")
    print(f"  time_varying_known : {train_ds.time_varying_known_reals}")
    print(f"  time_varying_unkn  : {train_ds.time_varying_unknown_reals}")

    # static_reals must contain spatial constants
    for feat in ["elevation", "lat", "lon"]:
        tag = PASS if feat in train_ds.static_reals else FAIL
        print(f"  {tag}  {feat} in static_reals")
    tag = PASS if MODEL_GROUP_COL in train_ds.static_categoricals else FAIL
    print(f"  {tag}  {MODEL_GROUP_COL} in static_categoricals")
    # S1/S2: city_id is collinear with super_node_id and was intentionally dropped
    # from static_categoricals to avoid redundant embeddings.
    tag = PASS if "city_id" not in train_ds.static_categoricals else FAIL
    print(f"  {tag}  city_id NOT in static_categoricals (S1/S2: single entity key)")

    # key features present
    for feat in ["water_deficit", "SPEI_3", "SPEI_3_diff"]:
        tag = PASS if feat in train_ds.time_varying_unknown_reals else FAIL
        print(f"  {tag}  {feat} in time_varying_unknown_reals")

    # dataloader batch check
    loader = train_ds.to_dataloader(train=False, batch_size=16, num_workers=0)
    bx, by = next(iter(loader))
    enc_shape = bx["encoder_cont"].shape
    dec_shape = bx["decoder_cont"].shape
    tgt_shape = by[0].shape
    print(f"{PASS} Batch shapes â€” enc_cont:{enc_shape}, dec_cont:{dec_shape}, target:{tgt_shape}")

    has_nan = bx["encoder_cont"].isnan().any().item()
    tag = PASS if not has_nan else FAIL
    print(f"  {tag}  No NaN in encoder_cont batch")

    # Val dataset from train schema
    val_cutoff = data[data.year == 2023]["time_idx"].max()
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds, data[data.time_idx <= val_cutoff], predict=True, stop_randomization=True
    )
    print(f"{PASS} Val TimeSeriesDataSet: {len(val_ds)} sequences")

except Exception as e:
    print(f"{FAIL} {e}")
    traceback.print_exc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TEST 5 â€” Model loading + predictions
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("TEST 5: Model loading + inference")
try:
    import os, torch
    from src.models.tft import load_tft_checkpoint

    best_ckpt = _resolve_checkpoint_path(PROCESSED_DATA_PATH)
    print(f"  Checkpoint: {best_ckpt}")

    model = load_tft_checkpoint(best_ckpt, map_location="cpu")
    model.eval()
    print(f"{PASS} Model loaded")
    print(f"  hidden_size       : {model.hparams.hidden_size}")
    print(f"  attention_heads   : {model.hparams.attention_head_size}")
    print(f"  dropout           : {model.hparams.dropout}")
    print(f"  learning_rate     : {model.hparams.learning_rate}")

    ckpt_encoder_len = int(getattr(model.hparams, "max_encoder_length", 90))
    ckpt_pred_len = int(getattr(model.hparams, "max_prediction_length", 30))
    print(f"  ckpt encoder_len  : {ckpt_encoder_len}")
    print(f"  ckpt pred_len     : {ckpt_pred_len}")

    # Quick forward pass on test data â€” use checkpoint's encoder length
    test_data_local = data[data.year >= 2024].copy()
    train_data_local = data[data.year < 2024].copy()
    train_ds_eval = create_dataset(train_data_local,
                                   max_encoder_length=ckpt_encoder_len,
                                   max_prediction_length=ckpt_pred_len)
    test_ds = TimeSeriesDataSet.from_dataset(
        train_ds_eval, test_data_local, predict=False, stop_randomization=True
    )
    test_loader = test_ds.to_dataloader(train=False, batch_size=32, num_workers=0)
    print(f"  Test sequences    : {len(test_ds)}")

    with torch.no_grad():
        preds = model.predict(
            test_loader,
            mode="raw",
            return_x=True,
            trainer_kwargs={"accelerator": "cpu", "devices": 1},
        )

    p = preds.output.prediction.cpu()
    print(f"{PASS} Predictions shape: {p.shape}  (expected: [N, 30, 3])")

    tag = PASS if p.shape[2] == 3 else FAIL
    print(f"  {tag}  Quantile dim = {p.shape[2]}  (expected 3)")

    p50 = p[:, :, 1].numpy()
    p50_min, p50_max = p50.min(), p50.max()
    in_range = (-5 < p50_min) and (p50_max < 5)
    tag = PASS if in_range else WARN
    print(f"  {tag}  P50 range: [{p50_min:.3f}, {p50_max:.3f}]  (expected within -5..5)")

    p10 = p[:, :, 0].numpy()
    p90 = p[:, :, 2].numpy()
    crossing = ((p10 > p50) | (p50 > p90)).mean()
    tag = PASS if crossing < 0.05 else WARN
    print(f"  {tag}  Quantile crossing rate P10>P50 or P50>P90: {crossing:.3%}  (expected <5%)")

    # Prediction metrics
    from sklearn.metrics import mean_squared_error
    actuals_normalized = preds.x["decoder_target"].cpu().numpy()  # shape [N, 30]
    p50_flat = p50.flatten()
    act_flat  = actuals_normalized.flatten()
    assert len(p50_flat) == len(act_flat), f"Length mismatch: pred={len(p50_flat)} actual={len(act_flat)}"
    rmse = np.sqrt(mean_squared_error(act_flat, p50_flat))
    mae  = np.mean(np.abs(act_flat - p50_flat))
    corr = np.corrcoef(act_flat, p50_flat)[0, 1]
    print(f"{PASS} Quick test metrics (P50, raw):")
    print(f"  RMSE       : {rmse:.4f}")
    print(f"  MAE        : {mae:.4f}")
    print(f"  Pearson r  : {corr:.4f}")

    # Flag anomaly if RMSE > 2.0 (SPEI scale ~[-3,3])
    tag = PASS if rmse < 2.0 else WARN
    print(f"  {tag}  RMSE < 2.0 (sane for SPEI Z-score scale)")

except Exception as e:
    print(f"{FAIL} {e}")
    traceback.print_exc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TEST 6 â€” evaluate.py end-to-end (run_evaluation.py logic)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("TEST 6: evaluate.py end-to-end")
try:
    from evaluate import evaluate_model
    metrics = evaluate_model(
        checkpoint_path=best_ckpt,
        test_year_start=2024,
        data_path=PROCESSED_DATA_PATH,
    )

    ov = metrics["overall_raw"]
    print()
    print(f"{PASS} evaluate_model() returned metrics successfully")
    print(f"  RMSE      : {ov['rmse']:.4f}")
    print(f"  MAE       : {ov['mae']:.4f}")
    print(f"  R2        : {ov['r2']:.4f}")
    print(f"  Pearson r : {ov['pearson_r']:.4f}")
    print(f"  Bias      : {ov['bias']:.4f}")
    print(f"  Samples   : {ov['samples']}")

    # Sanity checks
    tag = PASS if ov["rmse"] < 2.0 else WARN
    print(f"  {tag}  RMSE < 2.0")
    tag = PASS if abs(ov["bias"]) < 0.5 else WARN
    print(f"  {tag}  |Bias| < 0.5  (got {abs(ov['bias']):.4f})")
    tag = PASS if ov["pearson_r"] > 0 else WARN
    print(f"  {tag}  Pearson r > 0  (positive correlation)")
    tag = PASS if ov["samples"] > 0 else FAIL
    print(f"  {tag}  Samples > 0  (got {ov['samples']})")

    # Per-location
    print()
    print("  Per-location RMSE (raw):")
    for loc, m in metrics["per_location"].items():
        r = m["raw"]
        print(f"    {loc:12s}  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  r={r['pearson_r']:.4f}")

except Exception as e:
    print(f"{FAIL} {e}")
    traceback.print_exc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("PIPELINE TEST COMPLETE")
print("Check output files in results/ for saved plots and CSVs.")

