"""
Pipeline Test Script — validates all stages without re-training.
Run: python test_pipeline.py
"""
import sys
import traceback
import pandas as pd
import numpy as np

sys.path.insert(0, ".")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# TEST 1 — Imports
# ─────────────────────────────────────────────────────────────
section("TEST 1: Imports")
try:
    from src.data.spei import calculate_water_deficit, calculate_spei, classify_spei
    from src.data.preprocess import preprocess_pipeline
    from src.models.dataset import create_dataset, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH
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

# ─────────────────────────────────────────────────────────────
# TEST 2 — classify_spei canonical thresholds
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# TEST 3 — Processed data integrity
# ─────────────────────────────────────────────────────────────
section("TEST 3: Processed data integrity")
try:
    data = pd.read_parquet("data/processed/spei_dataset.parquet")
    data["year"] = data["time"].dt.year

    # Required columns
    required_cols = [
        "time", "time_idx", "location_id", "elevation",
        "SPEI_3", "SPEI_6", "SPEI_3_diff", "water_deficit",
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
    print(f"{PASS} Shape: {n_rows} rows × {n_cols} cols")

    # Locations
    locs = sorted(data.location_id.unique())
    expected_locs = ["Bojonegoro", "Lamongan", "Nganjuk", "Ngawi", "Tuban"]
    tag = PASS if locs == expected_locs else FAIL
    print(f"  {tag}  Locations: {locs}")

    # Date range
    print(f"{PASS} Date range: {data.time.min().date()} → {data.time.max().date()}")

    # SPEI_3 distribution — should be approx N(0,1)
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

    # time_idx monotone per location
    for loc, g in data.groupby("location_id"):
        g_sorted = g.sort_values("time")
        diffs = g_sorted.time_idx.diff().dropna()
        if not (diffs >= 0).all():
            print(f"{FAIL} time_idx not monotone for {loc}")
        else:
            print(f"  {PASS}  time_idx monotone: {loc}")

except Exception as e:
    print(f"{FAIL} {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# TEST 4 — TimeSeriesDataSet creation
# ─────────────────────────────────────────────────────────────
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

    # static_reals must contain elevation
    tag = PASS if "elevation" in train_ds.static_reals else FAIL
    print(f"  {tag}  elevation in static_reals")
    tag = PASS if "elevation" not in train_ds.time_varying_known_reals else FAIL
    print(f"  {tag}  elevation NOT in time_varying_known")

    # key features present
    for feat in ["SPEI_6", "water_deficit", "SPEI_3", "SPEI_3_diff"]:
        tag = PASS if feat in train_ds.time_varying_unknown_reals else FAIL
        print(f"  {tag}  {feat} in time_varying_unknown_reals")

    # dataloader batch check
    loader = train_ds.to_dataloader(train=False, batch_size=16, num_workers=0)
    bx, by = next(iter(loader))
    enc_shape = bx["encoder_cont"].shape
    dec_shape = bx["decoder_cont"].shape
    tgt_shape = by[0].shape
    print(f"{PASS} Batch shapes — enc_cont:{enc_shape}, dec_cont:{dec_shape}, target:{tgt_shape}")

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

# ─────────────────────────────────────────────────────────────
# TEST 5 — Model loading + predictions
# ─────────────────────────────────────────────────────────────
section("TEST 5: Model loading + inference")
try:
    import os, torch
    from pytorch_forecasting import TemporalFusionTransformer

    ckpt_dir = "logs/checkpoints"
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

    def parse_val_loss(fname):
        try:
            return float(fname.split("val_loss=")[1].replace(".ckpt", ""))
        except:
            return float("inf")

    best_ckpt = os.path.join(ckpt_dir, min(ckpts, key=parse_val_loss))
    print(f"  Checkpoint: {best_ckpt}")

    model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt, map_location="cpu")
    model.eval()
    print(f"{PASS} Model loaded")
    print(f"  hidden_size       : {model.hparams.hidden_size}")
    print(f"  attention_heads   : {model.hparams.attention_head_size}")
    print(f"  dropout           : {model.hparams.dropout}")
    print(f"  learning_rate     : {model.hparams.learning_rate}")

    ckpt_encoder_len = int(getattr(model.hparams, "max_encoder_length", 90))
    ckpt_pred_len    = int(getattr(model.hparams, "max_prediction_length", 30))
    print(f"  ckpt encoder_len  : {ckpt_encoder_len}")
    print(f"  ckpt pred_len     : {ckpt_pred_len}")

    # Quick forward pass on test data — use checkpoint's encoder length
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
        preds = model.predict(test_loader, mode="raw", return_x=True)

    p = preds.output.prediction.cpu()
    print(f"{PASS} Predictions shape: {p.shape}  (expected: [N, 30, 7])")

    # Check quantile dimension = 7
    tag = PASS if p.shape[2] == 7 else FAIL
    print(f"  {tag}  Quantile dim = {p.shape[2]}  (expected 7)")

    # Check P50 index=3 range sensible for SPEI
    p50 = p[:, :, 3].numpy()
    p50_min, p50_max = p50.min(), p50.max()
    in_range = (-5 < p50_min) and (p50_max < 5)
    tag = PASS if in_range else WARN
    print(f"  {tag}  P50 range: [{p50_min:.3f}, {p50_max:.3f}]  (expected within -5..5)")

    # Monotone quantiles: P10 < P50 < P90 should hold on average
    p10 = p[:, :, 1].numpy()
    p90 = p[:, :, 5].numpy()
    crossing = ((p10 > p50) | (p50 > p90)).mean()
    tag = PASS if crossing < 0.05 else WARN
    print(f"  {tag}  Quantile crossing rate P10>P50 or P50>P90: {crossing:.3%}  (expected <5%)")

    # Prediction metrics
    from sklearn.metrics import mean_squared_error
    actuals_raw = preds.x["decoder_target"].cpu().numpy()  # shape [N, 30]
    p50_flat = p50.flatten()
    act_flat  = actuals_raw.flatten()
    min_len   = min(len(p50_flat), len(act_flat))
    rmse = np.sqrt(mean_squared_error(act_flat[:min_len], p50_flat[:min_len]))
    mae  = np.mean(np.abs(act_flat[:min_len] - p50_flat[:min_len]))
    corr = np.corrcoef(act_flat[:min_len], p50_flat[:min_len])[0, 1]
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

# ─────────────────────────────────────────────────────────────
# TEST 6 — evaluate.py end-to-end (run_evaluation.py logic)
# ─────────────────────────────────────────────────────────────
section("TEST 6: evaluate.py end-to-end")
try:
    from evaluate import evaluate_model
    metrics = evaluate_model(checkpoint_path=best_ckpt, test_year_start=2024)

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

# ─────────────────────────────────────────────────────────────
section("PIPELINE TEST COMPLETE")
print("Check output files in results/ for saved plots and CSVs.")
