"""
full_evaluation.py
==================
One-shot comprehensive evaluation script for TFT SPEI Forecasting Model.

Outputs (saved to results/full_eval_<TIMESTAMP>/):
  Metrics:
    - metrics_summary.json       : Overall + per-location + per-horizon metrics
                                   + PICP (interval coverage) + naive baseline
    - metrics_report.txt         : Human-readable full report
    - horizon_metrics.csv        : RMSE/MAE/Bias/Corr per forecast day (1-30)
    - classification_report.csv  : SPEI drought-class accuracy per location
    - classification_summary.csv : 3-class broad accuracy summary
    - predictions_full.csv       : Aligned actual/pred table with in_interval flag

  Plots:
    01_scatter_overall.png       : Actual vs Predicted scatter (all locations)
    02_scatter_per_location.png  : 5-panel scatter per location
    03_timeseries_per_location.png : Time-series overlay per location (sample)
    04_error_distribution.png    : Error histogram + KDE per location
    05_variable_importance.png   : VSN encoder & decoder importance
    06_horizon_metrics.png       : RMSE / MAE / Bias / Corr vs forecast horizon
    07_location_comparison.png   : Grouped bar: metrics across locations
    08_quantile_fan.png          : P10/P50/P90 fan chart per location
    09_spei_classification.png   : Confusion heatmap (actual class vs pred class)
    10_bias_over_time.png        : Monthly rolling bias per location
    11_model_vs_naive_picp.png   : Model vs Naive RMSE comparison + PICP coverage

Run:
    python full_evaluation.py
    python full_evaluation.py --checkpoint logs/checkpoints/epoch=8-val_loss=0.37.ckpt
"""

import sys
import json
import re
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress only routine convergence / Lightning boilerplate warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
torch.set_float32_matmul_precision("medium")

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.models.dataset import create_dataset
from src.data.spei import classify_spei

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = {
    "Bojonegoro": "#e63946",
    "Lamongan":   "#f4a261",
    "Nganjuk":    "#2a9d8f",
    "Ngawi":      "#457b9d",
    "Tuban":      "#7b2d8b",
}
sns.set_theme(style="whitegrid", font_scale=1.0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    mask = np.isfinite(actual) & np.isfinite(pred)
    a, p = actual[mask], pred[mask]
    if len(a) < 2:
        return dict(rmse=None, mae=None, r2=None, bias=None, pearson_r=None, n=len(a))
    return dict(
        rmse=float(np.sqrt(mean_squared_error(a, p))),
        mae=float(mean_absolute_error(a, p)),
        r2=float(r2_score(a, p)),
        bias=float(np.mean(p - a)),
        pearson_r=float(np.corrcoef(a, p)[0, 1]),
        n=int(len(a)),
    )


def _log(msg: str = "", fp=None):
    print(msg)
    if fp:
        fp.write(msg + "\n")
        fp.flush()


def _broad(c: str) -> str:
    """Map 9-class SPEI label to 3-class (Kekeringan / Normal / Basah)."""
    return "Kekeringan" if "Kekeringan" in c else ("Basah" if "Basah" in c else "Normal")


def _best_checkpoint(ckpt_dir: Path) -> Path:
    """
    Return the checkpoint with the lowest val_loss parsed from filename.
    Pattern expected: epoch=N-val_loss=X.ckpt
    Falls back to alphabetical last if no val_loss can be parsed.
    """
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
    scored = []
    for p in ckpts:
        m = re.search(r"val_loss=([\d.]+)", p.name)
        if m:
            scored.append((float(m.group(1)), p))
    if scored:
        scored.sort(key=lambda x: x[0])   # lowest val_loss first
        return scored[0][1]
    return ckpts[-1]   # fallback: alphabetical last


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def run(checkpoint_path: str, out_dir: Path, log_fp):

    _log("=" * 72, log_fp)
    _log("  TFT SPEI-3 FORECASTING — COMPREHENSIVE EVALUATION", log_fp)
    _log(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_fp)
    _log("=" * 72, log_fp)

    # ── 1. LOAD DATA ─────────────────────────────────────────────────────
    _log("\n[1/6] Loading dataset …", log_fp)
    data_path = ROOT / "data/processed/spei_dataset.parquet"
    if not data_path.exists():
        _log(f"ERROR: {data_path} not found.", log_fp); return

    data = pd.read_parquet(data_path)
    data["year"] = data["time"].dt.year

    _log(f"  Rows      : {len(data):,}", log_fp)
    _log(f"  Period    : {data['time'].min().date()} → {data['time'].max().date()}", log_fp)
    _log(f"  Locations : {sorted(data['location_id'].unique())}", log_fp)
    _log(f"  SPEI_3    : mean={data['SPEI_3'].mean():.3f}  std={data['SPEI_3'].std():.3f}"
         f"  min={data['SPEI_3'].min():.3f}  max={data['SPEI_3'].max():.3f}", log_fp)

    # ── 2. LOAD MODEL ─────────────────────────────────────────────────────
    _log("\n[2/6] Loading model checkpoint …", log_fp)
    _log(f"  Path: {checkpoint_path}", log_fp)

    model = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint_path, map_location="cpu")
    model.eval()

    enc_len  = int(getattr(model.hparams, "max_encoder_length",  90))
    pred_len = int(getattr(model.hparams, "max_prediction_length", 30))
    _log(f"  Encoder length   : {enc_len}", log_fp)
    _log(f"  Prediction length: {pred_len}", log_fp)

    # ── 3. BUILD DATASETS ─────────────────────────────────────────────────
    # Split consistent with train.py: train<2023, val=2023, test>=2024
    _log("\n[3/6] Building datasets …", log_fp)
    train_data = data[data.year < 2023].copy()
    val_data   = data[data.year == 2023].copy()
    test_data  = data[data.year >= 2024].copy()

    _log(f"  Train : year < 2023   → {len(train_data):,} rows", log_fp)
    _log(f"  Val   : year == 2023  → {len(val_data):,} rows", log_fp)
    _log(f"  Test  : year >= 2024  → {len(test_data):,} rows  ← evaluation target", log_fp)

    # Guard: empty test set
    if test_data.empty:
        _log("ERROR: test_data is empty (no rows with year >= 2024). Aborting.", log_fp)
        return

    locations = sorted(test_data["location_id"].unique())
    if not locations:
        _log("ERROR: No locations found in test_data. Aborting.", log_fp)
        return

    train_ds = create_dataset(train_data,
                              max_encoder_length=enc_len,
                              max_prediction_length=pred_len)

    # ── 4. GENERATE PREDICTIONS (single pass — ensemble + horizon) ─────────
    _log("\n[4/6] Generating predictions …", log_fp)

    ensemble_rows = []
    # True per-horizon storage: keys 0..pred_len-1
    horizon_bank = {h: {"actual": [], "pred": []} for h in range(pred_len)}

    for loc in locations:
        _log(f"  Processing {loc} …", log_fp)
        loc_data = test_data[test_data.location_id == loc].copy()

        loc_ds = TimeSeriesDataSet.from_dataset(
            train_ds, loc_data, predict=False, stop_randomization=True)
        loader = loc_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

        raw = model.predict(loader, mode="raw", return_x=True)
        pv_norm = raw.output.prediction.cpu().numpy()      # (B, T, 7)  normalized
        tv  = raw.x["decoder_time_idx"].cpu().numpy()      # (B, T)

        # Denormalise predictions: GroupNormalizer computes
        #   norm = (x - center) / scale  so  raw = norm * scale + center
        # target_scale shape: (B, 2)  [:, 0]=center  [:, 1]=scale_factor
        if "target_scale" in raw.x:
            ts      = raw.x["target_scale"].cpu().numpy()  # (B, 2)
            _center = ts[:, 0].reshape(-1, 1, 1)           # (B, 1, 1)
            _scale  = ts[:, 1].reshape(-1, 1, 1)           # (B, 1, 1)
            pv = pv_norm * _scale + _center                # (B, T, 7) in SPEI units
        else:
            pv = pv_norm  # fallback: assume already in raw scale

        # Actual target values in decoder window (for horizon metrics)
        av = None
        if "decoder_target" in raw.x:
            av_norm = raw.x["decoder_target"].cpu().numpy()  # (B, T) normalized
            if "target_scale" in raw.x:
                ts2     = raw.x["target_scale"].cpu().numpy()  # (B, 2)
                _center2 = ts2[:, 0].reshape(-1, 1)             # (B, 1)
                _scale2  = ts2[:, 1].reshape(-1, 1)             # (B, 1)
                av = av_norm * _scale2 + _center2               # (B, T) in SPEI units
            else:
                av = av_norm
            # Diagnostic: verify decoder_target and actual SPEI are on same scale
            _log(f"    decoder_target  [{loc}]: "
                 f"mean={av.mean():.3f}  std={av.std():.3f}", log_fp)
            _log(f"    SPEI_3 actual   [{loc}]: "
                 f"mean={loc_data.SPEI_3.mean():.3f}  std={loc_data.SPEI_3.std():.3f}",
                 log_fp)

        # ── Step-0-only predictions (no ensemble averaging) ────────────────────
        # For each window i, tv[i, 0] is the time_idx at step=0.
        # Using step-0 only ensures each timestep is predicted from the
        # freshest encoder context, eliminating the smoothing bias introduced
        # by averaging predictions where the same timestep is at step 1..29.
        step0_preds: dict = {}          # time_idx -> {p10, p50, p90}
        for i in range(pv.shape[0]):
            t = int(tv[i, 0])           # time_idx of the first decoder step
            if t not in step0_preds:    # deduplicate (safety)
                step0_preds[t] = {
                    "pred_p10": float(pv[i, 0, 1]),
                    "pred_p50": float(pv[i, 0, 3]),
                    "pred_p90": float(pv[i, 0, 5]),
                }

        # ── Horizon bank: collect every (step, actual, pred) for horizon metrics ─
        # This uses ALL steps and ALL windows — each (i, step) pair represents
        # a forecasting event at a specific horizon depth.  Still uses
        # denormalized values so horizon metrics are in SPEI units.
        for i in range(pv.shape[0]):
            for step in range(pred_len):
                if av is not None:
                    horizon_bank[step]["actual"].append(float(av[i, step]))
                    horizon_bank[step]["pred"].append(float(pv[i, step, 3]))

        for t in sorted(step0_preds):
            ensemble_rows.append({
                "time_idx":    t,
                "location_id": loc,
                **step0_preds[t],
            })

    df_preds = pd.DataFrame(ensemble_rows)
    df_actual = (test_data[["time_idx", "time", "location_id", "SPEI_3"]]
                 .rename(columns={"SPEI_3": "actual"}))
    df = pd.merge(df_actual, df_preds, on=["time_idx", "location_id"], how="inner")
    df["error"] = df["pred_p50"] - df["actual"]
    df["month"] = pd.to_datetime(df["time"]).dt.to_period("M").astype(str)
    df["actual_class"] = df["actual"].apply(classify_spei)
    df["pred_class"]   = df["pred_p50"].apply(classify_spei)
    # PICP: 1 if actual falls inside the P10–P90 prediction interval, else 0
    # Nominal coverage for P10–P90 = 80%.
    df["in_interval"] = (
        (df["actual"] >= df["pred_p10"]) &
        (df["actual"] <= df["pred_p90"])
    ).astype(int)

    _log(f"  Merged prediction rows: {len(df):,}", log_fp)

    # ── 5. COMPUTE METRICS ────────────────────────────────────────────────
    _log("\n[5/6] Computing metrics …", log_fp)

    overall = _metrics(df["actual"].values, df["pred_p50"].values)

    per_loc = {}
    for loc in locations:
        sub = df[df.location_id == loc]
        per_loc[loc] = _metrics(sub["actual"].values, sub["pred_p50"].values)

    # True per-horizon metrics — from horizon_bank collected in prediction loop
    horizon_agg = []
    for h in range(pred_len):
        hd = horizon_bank[h]
        if len(hd["actual"]) >= 2:
            m = _metrics(np.array(hd["actual"]), np.array(hd["pred"]))
        else:
            m = dict(rmse=None, mae=None, r2=None, bias=None, pearson_r=None, n=0)
        m["horizon"] = h + 1
        horizon_agg.append(m)

    df_horizon = pd.DataFrame(horizon_agg)

    # ── PICP (Prediction Interval Coverage Probability) ────────────────────
    picp_overall = float(df["in_interval"].mean())
    picp_per_loc = {
        loc: float(df[df.location_id == loc]["in_interval"].mean())
        for loc in locations
    }

    # ── Naive persistence baseline ──────────────────────────────────────
    # Predict SPEI(t) = SPEI(t-1).  If the model cannot beat this,
    # the training configuration is fundamentally flawed.
    test_sorted_naive = (
        test_data.sort_values(["location_id", "time_idx"])
        .assign(naive_pred=lambda x:
                x.groupby("location_id")["SPEI_3"].transform(lambda s: s.shift(1)))
        [["time_idx", "location_id", "naive_pred"]]
        .dropna()
    )
    df_naive = pd.merge(
        df[["time_idx", "location_id", "actual"]],
        test_sorted_naive, on=["time_idx", "location_id"], how="inner"
    )
    naive_overall = _metrics(df_naive["actual"].values, df_naive["naive_pred"].values)
    naive_per_loc = {
        loc: _metrics(
            df_naive[df_naive.location_id == loc]["actual"].values,
            df_naive[df_naive.location_id == loc]["naive_pred"].values,
        )
        for loc in locations
    }

    # Classification — per-class stats (9 canonical SPEI classes)
    all_classes = [
        "Kekeringan Ekstrem", "Kekeringan Parah", "Kekeringan Sedang", "Kekeringan Ringan",
        "Normal",
        "Basah Ringan", "Basah Sedang", "Basah Parah", "Basah Ekstrem",
    ]
    clf_detail_rows = []
    for loc in locations:
        sub = df[df.location_id == loc]
        for cls in all_classes:
            actual_count = int((sub["actual_class"] == cls).sum())
            pred_count   = int((sub["pred_class"]   == cls).sum())
            correct      = int(((sub["actual_class"] == cls) & (sub["pred_class"] == cls)).sum())
            precision    = correct / pred_count   if pred_count   > 0 else 0.0
            recall       = correct / actual_count if actual_count > 0 else 0.0
            f1           = (2 * precision * recall / (precision + recall)
                            if (precision + recall) > 0 else 0.0)
            clf_detail_rows.append({
                "location":     loc,      "class":        cls,
                "actual_count": actual_count, "pred_count": pred_count,
                "correct":      correct,  "precision":    round(precision, 4),
                "recall":       round(recall, 4), "f1": round(f1, 4),
            })
    df_clf = pd.DataFrame(clf_detail_rows)

    # Broad accuracy summary (3-class)
    clf_summary_rows = []
    for loc in locations:
        sub       = df[df.location_id == loc].copy()
        exact_acc = float((sub["actual_class"] == sub["pred_class"]).mean())
        broad_acc = float(
            (sub["actual_class"].apply(_broad) == sub["pred_class"].apply(_broad)).mean())
        clf_summary_rows.append({
            "location": loc, "exact_acc": round(exact_acc, 4),
            "broad_acc": round(broad_acc, 4), "total": len(sub),
        })
    df_clf_summary = pd.DataFrame(clf_summary_rows)

    # Print overall metrics
    _log("\n  ─── OVERALL METRICS (test year ≥ 2024) ───", log_fp)
    for k, v in overall.items():
        if v is None:
            _log(f"    {k.upper():<12}: N/A", log_fp)
        else:
            _log(f"    {k.upper():<12}: {v:.4f}", log_fp)
    _log(f"    {'PICP':<12}: {picp_overall:.4f}  (nominal 0.80 for P10–P90)", log_fp)

    _log("\n  ─── NAIVE PERSISTENCE BASELINE ───", log_fp)
    for k, v in naive_overall.items():
        if v is None:
            _log(f"    {k.upper():<12}: N/A", log_fp)
        else:
            _log(f"    {k.upper():<12}: {v:.4f}", log_fp)
    _log("  (model must beat naive RMSE to demonstrate predictive skill)", log_fp)

    _log("\n  ─── PER-LOCATION METRICS ───", log_fp)
    for loc, m in per_loc.items():
        _log(f"    {loc:<15} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}"
             f"  R²={m['r2']:.4f}  Bias={m['bias']:.4f}  r={m['pearson_r']:.4f}"
             f"  PICP={picp_per_loc.get(loc, float('nan')):.4f}", log_fp)

    _log("\n  ─── HORIZON METRICS (first and last 5 days) ───", log_fp)
    for row in horizon_agg:
        h = int(row["horizon"])
        if h <= 5 or h >= pred_len - 4:
            rmse = row['rmse']; bias = row['bias']; r = row['pearson_r']
            _log(f"    Day {h:>2}  RMSE={rmse:.4f}  Bias={bias:.4f}  r={r:.4f}"
                 if rmse is not None else f"    Day {h:>2}  (no data)", log_fp)
        elif h == 6:
            _log("    ...   (days 6–25 omitted)", log_fp)

    # Save CSVs
    df_horizon.to_csv(out_dir / "horizon_metrics.csv", index=False)
    df_clf.to_csv(out_dir / "classification_report.csv", index=False)         # 9-class detail
    df_clf_summary.to_csv(out_dir / "classification_summary.csv", index=False) # 3-class summary
    df.to_csv(out_dir / "predictions_full.csv", index=False)

    # JSON — now includes per_horizon, picp, naive baseline
    metrics_payload = {
        "generated": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "encoder_length": enc_len,
        "prediction_length": pred_len,
        "test_period": "year >= 2024",
        "train_period": "year < 2023",
        "val_period": "year == 2023",
        "aggregation": "step-0-only (no ensemble averaging)",
        "overall": overall,
        "picp_overall": round(picp_overall, 6),
        "picp_nominal": 0.80,
        "picp_per_location": {k: round(v, 6) for k, v in picp_per_loc.items()},
        "naive_persistence": naive_overall,
        "naive_per_location": naive_per_loc,
        "per_location": per_loc,
        "per_horizon": [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
            for row in horizon_agg
        ],
    }
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    _log("\n  Saved: metrics_summary.json  horizon_metrics.csv"
         "  classification_report.csv  classification_summary.csv  predictions_full.csv", log_fp)

    # ── 6. PLOTS ─────────────────────────────────────────────────────────
    _log("\n[6/6] Generating plots …", log_fp)

    # ── Plot 01: Overall scatter ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    for loc in locations:
        sub = df[df.location_id == loc]
        ax.scatter(sub["actual"], sub["pred_p50"],
                   alpha=0.25, s=8, color=PALETTE[loc], label=loc)
    lim = [-3.5, 3.5]
    ax.plot(lim, lim, "k--", lw=1.5, label="Perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual SPEI-3"); ax.set_ylabel("Predicted P50 SPEI-3")
    ax.set_title(f"Actual vs Predicted — All Locations\n"
                 f"RMSE={overall['rmse']:.3f}  MAE={overall['mae']:.3f}"
                 f"  R²={overall['r2']:.3f}  r={overall['pearson_r']:.3f}")
    ax.legend(markerscale=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "01_scatter_overall.png", dpi=150)
    plt.close(fig)
    _log("  01_scatter_overall.png", log_fp)

    # ── Plot 02: Per-location scatter (2×3 grid) ──────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, loc in enumerate(locations):
        sub = df[df.location_id == loc]
        m   = per_loc[loc]
        ax  = axes[i]
        ax.scatter(sub["actual"], sub["pred_p50"],
                   alpha=0.3, s=8, color=PALETTE[loc])
        ax.plot([-3.5, 3.5], [-3.5, 3.5], "k--", lw=1.2)
        ax.set_xlim([-3.5, 3.5]); ax.set_ylim([-3.5, 3.5])
        ax.set_title(f"{loc}\nRMSE={m['rmse']:.3f}  R²={m['r2']:.3f}  Bias={m['bias']:.3f}")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted P50")
        ax.grid(True, alpha=0.3)
    axes[-1].axis("off")
    fig.suptitle("Per-Location Actual vs Predicted Scatter", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "02_scatter_per_location.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  02_scatter_per_location.png", log_fp)

    # ── Plot 03: Time series per location ─────────────────────────────────
    fig, axes = plt.subplots(len(locations), 1,
                             figsize=(16, 4 * len(locations)), sharex=False)
    if len(locations) == 1:
        axes = [axes]
    for i, loc in enumerate(locations):
        sub = df[df.location_id == loc].sort_values("time").reset_index(drop=True)
        n   = min(365, len(sub))
        sub = sub.iloc[:n]
        ax  = axes[i]
        ax.fill_between(range(n), sub["pred_p10"], sub["pred_p90"],
                        alpha=0.2, color=PALETTE[loc], label="P10–P90")
        ax.plot(range(n), sub["actual"],   "k-",  lw=1.2, label="Actual", alpha=0.85)
        ax.plot(range(n), sub["pred_p50"], "--",   lw=1.2,
                color=PALETTE[loc], label="Predicted P50")
        ax.axhline(-1.5, color="orange", ls=":", lw=0.9, alpha=0.7)
        ax.axhline( 1.5, color="steelblue", ls=":", lw=0.9, alpha=0.7)
        ax.set_ylim([-4, 4])
        ax.set_ylabel("SPEI-3", fontsize=9)
        ax.set_title(f"{loc} — First {n} test days", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Time Series: Actual vs Predicted P50 (test 2024+)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "03_timeseries_per_location.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  03_timeseries_per_location.png", log_fp)

    # ── Plot 04: Error distribution per location ──────────────────────────
    from scipy.stats import gaussian_kde
    n_cols = 3
    n_rows = int(np.ceil(len(locations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, loc in enumerate(locations):
        sub = df[df.location_id == loc]["error"].dropna()
        ax  = axes[i]
        ax.hist(sub, bins=40, color=PALETTE.get(loc, "gray"),
                alpha=0.7, edgecolor="white", density=True)
        kde = gaussian_kde(sub)
        xs  = np.linspace(sub.min(), sub.max(), 200)
        ax.plot(xs, kde(xs), "k-", lw=1.5)
        ax.axvline(0,          color="red",    ls="--", lw=1.2, label="Zero")
        ax.axvline(sub.mean(), color="orange",  ls="-",  lw=1.2,
                   label=f"Mean={sub.mean():.3f}")
        ax.set_title(f"{loc}\nμ={sub.mean():.3f}  σ={sub.std():.3f}")
        ax.set_xlabel("Error (Pred − Actual)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(len(locations), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Prediction Error Distribution per Location", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "04_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  04_error_distribution.png", log_fp)

    # ── Plot 05: Variable importance ──────────────────────────────────────
    try:
        test_ds_vi = TimeSeriesDataSet.from_dataset(
            train_ds, test_data, predict=False, stop_randomization=True)
        vi_loader  = test_ds_vi.to_dataloader(train=False, batch_size=64, num_workers=0)
        raw_vi     = model.predict(vi_loader, mode="raw")
        interp     = model.interpret_output(raw_vi, reduction="sum")

        def _to_dict(imp, names):
            if isinstance(imp, torch.Tensor):
                return {n: imp[j].item() for j, n in enumerate(names)}
            return {k: (v.item() if hasattr(v, "item") else v) for k, v in imp.items()}

        enc_imp = _to_dict(interp["encoder_variables"], model.encoder_variables)
        dec_imp = _to_dict(interp["decoder_variables"], model.decoder_variables)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, imp_dict, title, color in [
            (axes[0], enc_imp, "Encoder (Past Inputs)",   "steelblue"),
            (axes[1], dec_imp, "Decoder (Future Inputs)", "darkorange"),
        ]:
            names = list(imp_dict.keys())
            vals  = [imp_dict[n] for n in names]
            idx   = np.argsort(vals)
            ax.barh([names[j] for j in idx], [vals[j] for j in idx], color=color)
            ax.set_xlabel("Importance Score (VSN)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="x")
        fig.suptitle("Variable Importance (TFT Attention)", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "05_variable_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log("  05_variable_importance.png", log_fp)
    except Exception as e:
        _log(f"  05_variable_importance.png SKIPPED: {e}", log_fp)

    # ── Plot 06: Metrics per horizon ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = [
        ("rmse",      "RMSE",             "steelblue"),
        ("mae",       "MAE",              "darkorange"),
        ("bias",      "Bias (Pred−Actual)","tomato"),
        ("pearson_r", "Pearson r",         "seagreen"),
    ]
    for ax, (col, label, color) in zip(axes.flatten(), metrics_to_plot):
        valid = df_horizon.dropna(subset=[col])
        ax.bar(valid["horizon"], valid[col], color=color, alpha=0.75, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Forecast Horizon (days)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Forecast Horizon")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Metric Degradation — True Per-Step Horizon (test 2024+)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "06_horizon_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  06_horizon_metrics.png", log_fp)

    # ── Plot 07: Per-location comparison (grouped bars) ───────────────────
    metrics_bar = ["rmse", "mae", "pearson_r"]
    bar_labels  = ["RMSE", "MAE", "Pearson r"]
    x = np.arange(len(locations))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for j, (met, lbl) in enumerate(zip(metrics_bar, bar_labels)):
        vals = [per_loc[loc][met] for loc in locations]
        ax.bar(x + j * width, vals, width, label=lbl, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(locations, rotation=15)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metric Comparison Across Locations")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "07_location_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  07_location_comparison.png", log_fp)

    # ── Plot 08: Quantile fan chart — all locations ───────────────────────
    fig, axes = plt.subplots(len(locations), 1,
                             figsize=(14, 4 * len(locations)), sharex=False)
    if len(locations) == 1:
        axes = [axes]
    for i, loc in enumerate(locations):
        sub   = df[df.location_id == loc].sort_values("time").reset_index(drop=True)
        n_fan = min(180, len(sub))
        sub   = sub.iloc[:n_fan]
        dates = pd.to_datetime(sub["time"])
        ax    = axes[i]
        ax.fill_between(dates, sub["pred_p10"], sub["pred_p90"],
                        alpha=0.25, color=PALETTE.get(loc, "gray"),
                        label="P10–P90 Interval")
        ax.plot(dates, sub["pred_p50"], "-",  color=PALETTE.get(loc, "gray"),
                lw=1.8, label="Predicted P50")
        ax.plot(dates, sub["actual"],   "k-", lw=1.4, label="Actual")
        ax.axhline(-1.5, color="orange",    ls="--", lw=1, alpha=0.7)
        ax.axhline( 1.5, color="steelblue", ls="--", lw=1, alpha=0.7)
        ax.set_ylim([-4, 4])
        ax.set_ylabel("SPEI-3")
        ax.set_title(f"{loc} — first {n_fan} test days")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Quantile Fan Chart P10/P50/P90 (test 2024+)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "08_quantile_fan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  08_quantile_fan.png", log_fp)

    # ── Plot 09: SPEI classification confusion heatmap ────────────────────
    df["ab"] = df["actual_class"].apply(_broad)
    df["pb"] = df["pred_class"].apply(_broad)
    cats     = ["Kekeringan", "Normal", "Basah"]
    conf     = pd.crosstab(df["ab"], df["pb"],
                           rownames=["Actual"], colnames=["Predicted"])
    conf     = conf.reindex(index=cats, columns=cats, fill_value=0)
    conf_pct = conf.div(conf.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, ax=axes[0])
    axes[0].set_title("Confusion Matrix — Count\n(3-class: Kekeringan / Normal / Basah)")
    sns.heatmap(conf_pct, annot=True, fmt=".1f", cmap="Greens",
                linewidths=0.5, ax=axes[1], vmin=0, vmax=100)
    axes[1].set_title("Confusion Matrix — Row % (Recall)\n(3-class)")
    fig.tight_layout()
    fig.savefig(out_dir / "09_spei_classification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  09_spei_classification.png", log_fp)

    # ── Plot 10: Monthly bias per location ────────────────────────────────
    month_bias = (df.groupby(["location_id", "month"])["error"]
                  .mean().reset_index().rename(columns={"error": "bias"}))
    month_bias["month_dt"] = pd.to_datetime(month_bias["month"])
    month_bias = month_bias.sort_values("month_dt")

    fig, ax = plt.subplots(figsize=(14, 5))
    for loc in locations:
        sub_m = month_bias[month_bias.location_id == loc]
        ax.plot(sub_m["month_dt"], sub_m["bias"],
                marker="o", markersize=3, lw=1.5,
                color=PALETTE[loc], label=loc)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Bias (Pred − Actual)")
    ax.set_title("Monthly Bias Over Test Period per Location")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "10_bias_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  10_bias_over_time.png", log_fp)
    # ── Plot 11: Model vs Naive baseline RMSE + PICP per location ──────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: RMSE comparison (model vs naive) per location
    x      = np.arange(len(locations))
    width  = 0.35
    rmse_model = [per_loc[loc]["rmse"] or 0.0 for loc in locations]
    rmse_naive = [naive_per_loc[loc]["rmse"] or 0.0 for loc in locations]
    axes[0].bar(x - width / 2, rmse_model, width, label="TFT Model",
                color="steelblue", alpha=0.85)
    axes[0].bar(x + width / 2, rmse_naive, width, label="Naive Persistence",
                color="tomato",    alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(locations, rotation=15, ha="right")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE: TFT Model vs Naive Persistence Baseline")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right panel: PICP per location (bar) with nominal 80% line
    picp_vals = [picp_per_loc.get(loc, 0.0) for loc in locations]
    bar_colors = [
        "seagreen" if v >= 0.75 else "darkorange" for v in picp_vals
    ]
    axes[1].bar(x, picp_vals, color=bar_colors, alpha=0.85)
    axes[1].axhline(0.80, color="black", lw=1.5, ls="--",
                    label="Nominal 80% (P10–P90)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(locations, rotation=15, ha="right")
    axes[1].set_ylim([0, 1.05])
    axes[1].set_ylabel("Coverage Probability")
    axes[1].set_title("PICP — P10–P90 Interval Coverage per Location")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Model Skill vs Naive Baseline and Prediction Interval Coverage",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "11_model_vs_naive_picp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  11_model_vs_naive_picp.png", log_fp)
    # ── Final summary ─────────────────────────────────────────────────────
    _log("\n" + "=" * 72, log_fp)
    _log("  ALL OUTPUTS SAVED", log_fp)
    _log(f"  Directory : {out_dir}", log_fp)
    _log("=" * 72, log_fp)

    _log("\n  FILES GENERATED:", log_fp)
    for f in sorted(out_dir.iterdir()):
        size = f.stat().st_size
        _log(f"    {f.name:<45}  {size/1024:>7.1f} KB", log_fp)

    _log("\n  CLASSIFICATION ACCURACY (broad 3-class):", log_fp)
    for _, row in df_clf_summary.iterrows():
        _log(f"    {row['location']:<15}  exact={row['exact_acc']*100:.1f}%"
             f"  broad={row['broad_acc']*100:.1f}%", log_fp)

    _log("\n  PICP — P10–P90 Interval Coverage (nominal = 80%):", log_fp)
    for loc in locations:
        _log(f"    {loc:<15}  PICP={picp_per_loc.get(loc, float('nan')):.4f}", log_fp)
    _log(f"    {'OVERALL':<15}  PICP={picp_overall:.4f}", log_fp)

    _log("\n  MODEL vs NAIVE PERSISTENCE (overall):", log_fp)
    m_rmse = overall.get("rmse") or float("nan")
    n_rmse = naive_overall.get("rmse") or float("nan")
    skill  = (1.0 - m_rmse / n_rmse) * 100 if n_rmse > 0 else float("nan")
    _log(f"    Model  RMSE = {m_rmse:.4f}", log_fp)
    _log(f"    Naive  RMSE = {n_rmse:.4f}", log_fp)
    _log(f"    Skill Score = {skill:.1f}%  (positive = model beats naive)", log_fp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full TFT SPEI Evaluation")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .ckpt file. Defaults to latest in logs/checkpoints/")
    args = parser.parse_args()

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = str(_best_checkpoint(ROOT / "logs/checkpoints"))

    print(f"Checkpoint : {ckpt_path}")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"results/full_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "metrics_report.txt"
    with open(log_path, "w", encoding="utf-8") as log_fp:
        run(ckpt_path, out_dir, log_fp)

    print(f"\nDone. Results → {out_dir}")


if __name__ == "__main__":
    main()
