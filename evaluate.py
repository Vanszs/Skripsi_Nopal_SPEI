"""
Evaluation script for TFT SPEI Forecasting Model
"""
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.models.dataset import create_dataset

def evaluate_model(checkpoint_path="logs/checkpoints/enc30-epoch=1-val_loss=0.2892.ckpt", test_year_start=2024):
    """
    Evaluate model on test set.
    NOTE: test_year_start=2024 to avoid overlap with validation (2023).
    Train: <2023, Val: 2023, Test: >=2024
    """
    print("="*60)
    print("TFT SPEI FORECASTING - EVALUATION")
    print("="*60)
    
    # Load data
    data = pd.read_parquet("data/processed/spei_dataset.parquet")
    data["year"] = data["time"].dt.year

    print(f"\nTotal Dataset Shape: {data.shape}")
    print(f"Locations: {sorted(data['location_id'].unique())}")
    print(f"Date Range: {data['time'].min()} to {data['time'].max()}")

    # ── Load model FIRST so we can read its trained hparams ──────────────
    print(f"\nLoading model: {checkpoint_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    model.to("cpu")

    # Read encoder/prediction lengths from the checkpoint — critical for
    # interpret_output() to not crash with a size mismatch.
    ckpt_encoder_len = int(getattr(model.hparams, "max_encoder_length", 90))
    ckpt_pred_len    = int(getattr(model.hparams, "max_prediction_length", 30))
    print(f"  Checkpoint encoder length : {ckpt_encoder_len}")
    print(f"  Checkpoint prediction len : {ckpt_pred_len}")

    # Create training dataset using the TRAINING split: year < 2023.
    # IMPORTANT: this must NOT use test_year_start (2024) because the scaler
    # is fit on train_data — using year < 2024 would include the 2023 val set
    # and produce a scaler that does not match the one used during training.
    train_data = data[data.year < 2023].copy()
    train_ds = create_dataset(train_data,
                              max_encoder_length=ckpt_encoder_len,
                              max_prediction_length=ckpt_pred_len)

    # Test data (>= test_year_start)
    test_data = data[data.year >= test_year_start].copy()
    print(f"\nTest Data Shape: {test_data.shape}")
    print(f"Test Period: {test_data['time'].min()} to {test_data['time'].max()}")

    pred_len = ckpt_pred_len

    def generate_predictions(model, test_data, train_ds, pred_len):
        """
        Generate predictions using STEP-0-ONLY aggregation.

        Rationale: for each time_idx T, many overlapping windows contain T at
        different forecast steps (s=0, 1, ..., pred_len-1).  Averaging all of
        them introduces systematic smoothing bias — predictions from windows
        where T is at step 29 carry stale encoder context and compress towards
        the mean.  Instead we use ONLY the window that predicts T at step s=0
        (the freshest encoder context), which gives the least-biased estimate
        and eliminates the averaging-induced systematic offset.
        """
        results = []
        locations = test_data["location_id"].unique()

        for loc in locations:
            print(f"Processing {loc}...")
            loc_data = test_data[test_data.location_id == loc].copy()

            loc_ds = TimeSeriesDataSet.from_dataset(
                train_ds, loc_data, predict=False, stop_randomization=True
            )
            loc_loader = loc_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

            raw_preds = model.predict(loc_loader, mode="raw", return_x=True)
            p_values = raw_preds.output.prediction.cpu().numpy()  # (B, T, 7)
            t_values = raw_preds.x["decoder_time_idx"].cpu().numpy()  # (B, T)

            # Step-0-only: take each window's first-step prediction only.
            # tv[i, 0] is the time_idx that window i predicts at step 0.
            step0_preds = {}   # time_idx -> {p10, p50, p90}
            for i in range(p_values.shape[0]):
                t_idx = int(t_values[i, 0])   # step = 0
                if t_idx not in step0_preds:  # deduplicate (safety)
                    step0_preds[t_idx] = {
                        "pred_p10": float(p_values[i, 0, 1]),
                        "pred_p50": float(p_values[i, 0, 3]),
                        "pred_p90": float(p_values[i, 0, 5]),
                    }

            for t_idx in sorted(step0_preds):
                results.append({
                    "time_idx":    t_idx,
                    "location_id": loc,
                    **step0_preds[t_idx],
                })

        return pd.DataFrame(results)

    print("\nGenerating predictions...")
    df_preds = generate_predictions(model, test_data, train_ds, pred_len)

    df_actual = test_data[["time_idx", "time", "location_id", "SPEI_3"]].rename(
        columns={"SPEI_3": "actual"}
    )
    df_final = pd.merge(df_actual, df_preds, on=["time_idx", "location_id"], how="inner")

    print(f"Predictions rows: {len(df_final)}")

    def calculate_metrics(df, actual_col="actual", pred_col="pred_p50"):
        actual = df[actual_col].values
        pred = df[pred_col].values
        return {
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "mae": float(mean_absolute_error(actual, pred)),
            "r2": float(r2_score(actual, pred)),
            "bias": float(np.mean(pred - actual)),
            "pearson_r": float(np.corrcoef(actual, pred)[0, 1]),
            "samples": int(len(actual)),
        }

    # ========================================================================
    # NOTE: Post-hoc calibration removed to prevent data leakage.
    # Previously, bias/scale were computed FROM test-set statistics and
    # applied back TO the same test set — this inflates metrics artificially.
    # Only RAW model predictions are used for fair evaluation.
    # ========================================================================

    overall_raw = calculate_metrics(df_final, pred_col="pred_p50")

    # ── PICP: Prediction Interval Coverage Probability ───────────────────
    # Nominal coverage for a P10–P90 interval = 80%.
    # Values well below 0.80 signal under-coverage (interval too narrow);
    # values above 0.80 signal over-coverage (interval too wide / conservative).
    df_final["in_interval"] = (
        (df_final["actual"] >= df_final["pred_p10"]) &
        (df_final["actual"] <= df_final["pred_p90"])
    )
    picp_overall = float(df_final["in_interval"].mean())
    picp_per_loc = {
        loc: float(df_final[df_final.location_id == loc]["in_interval"].mean())
        for loc in df_final["location_id"].unique()
    }

    # ── Naive persistence baseline ────────────────────────────────────────
    # Naive: predict SPEI(t) = SPEI(t-1)  (yesterday's value = today's forecast).
    # If the model cannot beat naive RMSE, the training configuration is suspect.
    test_sorted = test_data.sort_values(["location_id", "time_idx"]).copy()
    test_sorted["naive_pred"] = (
        test_sorted.groupby("location_id")["SPEI_3"].shift(1)
    )
    df_naive = test_sorted[["time_idx", "location_id", "naive_pred"]].dropna()
    df_naive_merged = pd.merge(
        df_final[["time_idx", "location_id", "actual"]],
        df_naive, on=["time_idx", "location_id"], how="inner"
    )
    naive_raw = calculate_metrics(df_naive_merged, actual_col="actual", pred_col="naive_pred")

    print("\n" + "="*60)
    print(f"TEST SET METRICS ({test_year_start}-2025)")
    print("="*60)
    print("RAW MODEL (unbiased, step-0-only predictions):")
    for k, v in overall_raw.items():
        print(f"  {k.upper():10}: {v}")
    print(f"  {'PICP':10}: {picp_overall:.4f}  (nominal 0.80 for P10-P90)")
    print("\nNAIVE PERSISTENCE BASELINE:")
    for k, v in naive_raw.items():
        print(f"  {k.upper():10}: {v}")

    # Per-location metrics (raw only — no leaky calibration)
    per_location = {}
    for loc in df_final["location_id"].unique():
        loc_df = df_final[df_final.location_id == loc]
        naive_loc = df_naive_merged[df_naive_merged.location_id == loc]
        per_location[loc] = {
            "raw":     calculate_metrics(loc_df, pred_col="pred_p50"),
            "naive":   calculate_metrics(naive_loc, actual_col="actual", pred_col="naive_pred"),
            "picp":    picp_per_loc.get(loc, None),
        }
    
    # Get interpretation using raw mode
    print("\nExtracting variable importance...")
    test_ds = TimeSeriesDataSet.from_dataset(train_ds, test_data, predict=False, stop_randomization=True)
    test_dataloader = test_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    raw_preds = model.predict(test_dataloader, mode="raw")
    interpretation = model.interpret_output(raw_preds, reduction="sum")
    
    encoder_importance = interpretation["encoder_variables"]
    decoder_importance = interpretation["decoder_variables"]
    
    print("\n" + "="*60)
    print("VARIABLE IMPORTANCE (VSN)")
    print("="*60)
    
    def process_importance(importance_data, variable_names):
        if isinstance(importance_data, torch.Tensor):
            # If Tensor, map to names
            import_dict = {name: importance_data[i].item() for i, name in enumerate(variable_names)}
        elif isinstance(importance_data, dict):
            # If dict, use as is (ensure values are floats)
            import_dict = {k: v.item() if hasattr(v, "item") else v for k, v in importance_data.items()}
        else:
            return {}
        return import_dict

    encoder_importance_dict = process_importance(encoder_importance, model.encoder_variables)
    decoder_importance_dict = process_importance(decoder_importance, model.decoder_variables)
    
    print("\nEncoder Variables (Past Inputs):")
    for var, score in sorted(encoder_importance_dict.items(), key=lambda x: -x[1]):
        print(f"  {var}: {score:.4f}")
    
    print("\nDecoder Variables (Future Inputs):")
    for var, score in sorted(decoder_importance_dict.items(), key=lambda x: -x[1]):
        print(f"  {var}: {score:.4f}")
    
    # Save plots
    os.makedirs("results", exist_ok=True)
    
    # 1. Variable Importance Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    encoder_vars = list(encoder_importance_dict.keys())
    encoder_vals = list(encoder_importance_dict.values())
    
    ax1 = axes[0]
    ax1.barh(encoder_vars, encoder_vals, color="steelblue")
    ax1.set_xlabel("Importance Score")
    ax1.set_title("Encoder Variable Importance (Past Inputs)")
    ax1.invert_yaxis()
    
    decoder_vars = list(decoder_importance_dict.keys())
    decoder_vals = list(decoder_importance_dict.values())
    
    ax2 = axes[1]
    ax2.barh(decoder_vars, decoder_vals, color="darkorange")
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Decoder Variable Importance (Future Inputs)")
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("results/variable_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: results/variable_importance.png")
    
    # 2. Prediction vs Actual scatter
    pred_flat = df_final["pred_p50"].values
    actual_flat = df_final["actual"].values
    rmse = overall_raw["rmse"]
    correlation = overall_raw["pearson_r"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual_flat, pred_flat, alpha=0.3, s=10)
    lims = [min(actual_flat.min(), pred_flat.min()), max(actual_flat.max(), pred_flat.max())]
    ax.plot(lims, lims, "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual SPEI-3")
    ax.set_ylabel("Predicted SPEI-3")
    ax.set_title(f"TFT SPEI Prediction\nRMSE={rmse:.4f}, Corr={correlation:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/prediction_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/prediction_scatter.png")
    
    # 3. Time series sample
    sample_size = min(365, len(actual_flat))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(sample_size), actual_flat[:sample_size], label="Actual", alpha=0.8)
    ax.plot(range(sample_size), pred_flat[:sample_size], label="Predicted", alpha=0.8)
    ax.fill_between(range(sample_size), -1.5, 1.5, alpha=0.1, color="green", label="Normal Range")
    ax.axhline(y=-1.5, color="orange", linestyle="--", alpha=0.5, label="Drought Threshold")
    ax.set_xlabel("Days")
    ax.set_ylabel("SPEI-3")
    ax.set_title("TFT SPEI-3 Forecast Sample")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/timeseries_sample.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/timeseries_sample.png")
    
    # 4. Error distribution histogram
    errors = pred_flat - actual_flat
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="r", linestyle="--", label="Zero Error")
    ax.set_xlabel("Prediction Error (Predicted - Actual)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Error Distribution\nMean={np.mean(errors):.4f}, Std={np.std(errors):.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/error_distribution.png")
    
    # Save evaluation artifacts
    metrics_payload = {
        "overall_raw":    overall_raw,
        "overall_picp":   picp_overall,
        "overall_naive":  naive_raw,
        "per_location":   per_location,
        "notes": {
            "train_split":   "year < 2023",
            "val_split":     "year == 2023",
            "test_split":    f"year >= {test_year_start}",
            "aggregation":   "step-0-only (no ensemble averaging)",
            "picp_nominal":  "0.80 for P10-P90 interval",
        },
    }

    df_final.to_csv("results/predictions_eval.csv", index=False)
    with open("results/evaluation_metrics_detailed.json", "w") as _f:
        json.dump(metrics_payload, _f, indent=2)

    print("\nSaved: results/predictions_eval.csv")
    print("Saved: results/evaluation_metrics_detailed.json")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

    return metrics_payload

if __name__ == "__main__":
    evaluate_model()
