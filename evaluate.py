"""
Evaluation script for TFT SPEI Forecasting Model (schema v2 aware).
"""
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.dataset import MODEL_GROUP_COL, create_dataset

EXPECTED_SCHEMA_VERSION = 2

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but StandardScaler was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=UserWarning,
)
torch.set_float32_matmul_precision("medium")


def _metrics(actual, pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mae": float(mean_absolute_error(actual, pred)),
        "r2": float(r2_score(actual, pred)),
        "bias": float(np.mean(pred - actual)),
        "pearson_r": float(np.corrcoef(actual, pred)[0, 1]),
        "samples": int(len(actual)),
    }


def _entity_col(data: pd.DataFrame):
    if MODEL_GROUP_COL in data.columns:
        return MODEL_GROUP_COL
    if "location_id" in data.columns:
        return "location_id"
    raise ValueError("No valid grouping column found (expected super_node_id/location_id).")


def _quantile_index_map(model):
    quantiles = np.array([float(q) for q in getattr(model.loss, "quantiles", [0.1, 0.5, 0.9])])
    idx_p10 = int(np.argmin(np.abs(quantiles - 0.10)))
    idx_p50 = int(np.argmin(np.abs(quantiles - 0.50)))
    idx_p90 = int(np.argmin(np.abs(quantiles - 0.90)))
    return quantiles.tolist(), {"p10": idx_p10, "p50": idx_p50, "p90": idx_p90}


def _checkpoint_from_run_config():
    cfg_path = Path("logs/run_config.json")
    if not cfg_path.exists():
        return None
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        ckpt = payload.get("best_model_path")
        if ckpt and Path(ckpt).exists():
            return ckpt
    except Exception:
        return None
    return None


def evaluate_model(
    checkpoint_path="logs/checkpoints/enc90-epoch=3-val_loss=0.1956.ckpt",
    test_year_start=2024,
    data_path="data/processed/spei_dataset.parquet",
):
    print("=" * 60)
    print("TFT SPEI FORECASTING - EVALUATION")
    print("=" * 60)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")
    data = pd.read_parquet(data_path)
    data["time"] = pd.to_datetime(data["time"])
    data["year"] = data["time"].dt.year
    entity_col = _entity_col(data)

    if "schema_version" in data.columns:
        versions = sorted(data["schema_version"].dropna().unique().tolist())
        if versions != [EXPECTED_SCHEMA_VERSION]:
            raise ValueError(
                f"Processed schema version is {versions}, expected [{EXPECTED_SCHEMA_VERSION}]."
            )

    print(f"\nTotal Dataset Shape: {data.shape}")
    print(f"Entity key: {entity_col}")
    print(f"Entities: {sorted(data[entity_col].astype(str).unique().tolist())}")
    print(f"Cities: {sorted(data['city_id'].astype(str).unique().tolist())}")
    print(f"Date Range: {data['time'].min()} to {data['time'].max()}")

    print(f"\nLoading model: {checkpoint_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    model.to("cpu")
    quantiles, qidx = _quantile_index_map(model)

    ckpt_encoder_len = int(getattr(model.hparams, "max_encoder_length", 90))
    ckpt_pred_len = int(getattr(model.hparams, "max_prediction_length", 30))
    print(f"  Checkpoint encoder length : {ckpt_encoder_len}")
    print(f"  Checkpoint prediction len : {ckpt_pred_len}")

    train_data = data[data.year < 2023].copy()
    train_ds = create_dataset(
        train_data,
        max_encoder_length=ckpt_encoder_len,
        max_prediction_length=ckpt_pred_len,
    )

    test_data = data[data.year >= test_year_start].copy()
    print(f"\nTest Data Shape: {test_data.shape}")
    print(f"Test Period: {test_data['time'].min()} to {test_data['time'].max()}")
    pred_len = ckpt_pred_len

    def generate_predictions():
        results = []
        for ent in sorted(test_data[entity_col].astype(str).unique()):
            print(f"Processing {ent}...")
            loc_data = test_data[test_data[entity_col].astype(str) == ent].copy()
            loc_ds = TimeSeriesDataSet.from_dataset(
                train_ds, loc_data, predict=False, stop_randomization=True
            )
            loc_loader = loc_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
            raw_preds = model.predict(
                loc_loader,
                mode="raw",
                return_x=True,
                trainer_kwargs={"accelerator": "cpu", "devices": 1},
            )
            p_values = raw_preds.output.prediction.cpu().numpy()
            t_values = raw_preds.x["decoder_time_idx"].cpu().numpy()
            step0_preds = {}
            for i in range(p_values.shape[0]):
                t_idx = int(t_values[i, 0])
                if t_idx not in step0_preds:
                    step0_preds[t_idx] = {
                        "pred_p10": float(p_values[i, 0, qidx["p10"]]),
                        "pred_p50": float(p_values[i, 0, qidx["p50"]]),
                        "pred_p90": float(p_values[i, 0, qidx["p90"]]),
                    }
            city_id = str(loc_data["city_id"].iloc[0])
            for t_idx in sorted(step0_preds):
                results.append(
                    {
                        "time_idx": t_idx,
                        entity_col: ent,
                        "city_id": city_id,
                        **step0_preds[t_idx],
                    }
                )
        return pd.DataFrame(results)

    print("\nGenerating predictions...")
    df_preds = generate_predictions()
    df_actual = test_data[[entity_col, "city_id", "time_idx", "time", "SPEI_3"]].rename(
        columns={"SPEI_3": "actual"}
    )
    df_final = pd.merge(df_actual, df_preds, on=[entity_col, "city_id", "time_idx"], how="inner")
    print(f"Predictions rows: {len(df_final)}")

    overall_raw = _metrics(df_final["actual"].values, df_final["pred_p50"].values)

    df_final["in_interval"] = (
        (df_final["actual"] >= df_final["pred_p10"])
        & (df_final["actual"] <= df_final["pred_p90"])
    )
    picp_overall = float(df_final["in_interval"].mean())
    picp_per_entity = {
        ent: float(df_final[df_final[entity_col].astype(str) == str(ent)]["in_interval"].mean())
        for ent in sorted(df_final[entity_col].astype(str).unique())
    }

    test_sorted = test_data.sort_values([entity_col, "time_idx"]).copy()
    test_sorted["naive_pred"] = test_sorted.groupby(entity_col)["SPEI_3"].shift(1)
    df_naive = test_sorted[[entity_col, "time_idx", "naive_pred"]].dropna()
    df_naive_merged = pd.merge(
        df_final[[entity_col, "time_idx", "actual"]],
        df_naive,
        on=[entity_col, "time_idx"],
        how="inner",
    )
    naive_raw = _metrics(df_naive_merged["actual"].values, df_naive_merged["naive_pred"].values)

    print("\n" + "=" * 60)
    print(f"TEST SET METRICS ({test_year_start}+) | key={entity_col}")
    print("=" * 60)
    print("RAW MODEL:")
    for k, v in overall_raw.items():
        print(f"  {k.upper():10}: {v}")
    print(f"  {'PICP':10}: {picp_overall:.4f}  (nominal 0.80 for P10-P90)")
    print("\nNAIVE PERSISTENCE BASELINE:")
    for k, v in naive_raw.items():
        print(f"  {k.upper():10}: {v}")

    # Horizon metrics
    actual_lookup = {
        (int(r.time_idx), str(getattr(r, entity_col))): float(r.SPEI_3)
        for r in test_data.itertuples(index=False)
    }
    horizon_preds = {h: {} for h in range(pred_len)}
    for ent in sorted(test_data[entity_col].astype(str).unique()):
        loc_data = test_data[test_data[entity_col].astype(str) == ent].copy()
        loc_ds = TimeSeriesDataSet.from_dataset(train_ds, loc_data, predict=False, stop_randomization=True)
        loader = loc_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        raw = model.predict(
            loader,
            mode="raw",
            return_x=True,
            trainer_kwargs={"accelerator": "cpu", "devices": 1},
        )
        pv = raw.output.prediction.cpu().numpy()
        tv = raw.x["decoder_time_idx"].cpu().numpy()
        for i in range(pv.shape[0]):
            for h in range(pred_len):
                key = (int(tv[i, h]), ent)
                if key not in horizon_preds[h]:
                    horizon_preds[h][key] = float(pv[i, h, qidx["p50"]])

    # Fair naive horizon on the same timestamp/entity subset as model horizon predictions.
    naive_by_h = {}
    for h in range(pred_len):
        pairs = []
        for (t_idx, ent), _pred_val in horizon_preds[h].items():
            y_t = actual_lookup.get((t_idx, ent))
            y_prev = actual_lookup.get((t_idx - (h + 1), ent))
            if y_t is not None and y_prev is not None:
                pairs.append((y_t, y_prev))
        if len(pairs) >= 2:
            a = np.array([x[0] for x in pairs], dtype=float)
            p = np.array([x[1] for x in pairs], dtype=float)
            naive_by_h[h + 1] = float(np.sqrt(mean_squared_error(a, p)))
        else:
            naive_by_h[h + 1] = None

    beat_count = 0
    print("\nMULTI-HORIZON EVALUATION (step-h-only)")
    print(f"{'Day':>5} {'TFT_RMSE':>10} {'Naive_RMSE':>12} {'Ratio':>8} {'Beats?':>8}")
    print("-" * 50)
    for h in range(pred_len):
        actuals_h = []
        preds_h = []
        for key, pval in horizon_preds[h].items():
            aval = actual_lookup.get(key)
            if aval is not None:
                actuals_h.append(aval)
                preds_h.append(pval)
        if len(actuals_h) < 2:
            continue
        rmse_h = float(np.sqrt(mean_squared_error(actuals_h, preds_h)))
        naive_h = naive_by_h.get(h + 1)
        beats = rmse_h < naive_h if naive_h else False
        if beats:
            beat_count += 1
        ratio = rmse_h / naive_h if naive_h else float("nan")
        marker = "  BEATS" if beats else ""
        print(f"  {h+1:>3} {rmse_h:>10.4f} {naive_h:>12.4f} {ratio:>7.2f}x {'YES' if beats else 'no':>6}{marker}")
    print(f"\nModel beats naive at {beat_count}/{pred_len} horizons")

    # Per-entity and per-city metrics
    per_entity = {}
    for ent in sorted(df_final[entity_col].astype(str).unique()):
        ent_df = df_final[df_final[entity_col].astype(str) == ent]
        naive_ent = df_naive_merged[df_naive_merged[entity_col].astype(str) == ent]
        per_entity[ent] = {
            "raw": _metrics(ent_df["actual"].values, ent_df["pred_p50"].values),
            "naive": _metrics(naive_ent["actual"].values, naive_ent["naive_pred"].values),
            "picp": picp_per_entity.get(ent),
            "city_id": str(ent_df["city_id"].iloc[0]),
        }

    per_city = {}
    for city in sorted(df_final["city_id"].astype(str).unique()):
        city_df = df_final[df_final["city_id"].astype(str) == city]
        city_naive = pd.merge(
            city_df[[entity_col, "time_idx", "actual"]],
            df_naive[[entity_col, "time_idx", "naive_pred"]],
            on=[entity_col, "time_idx"],
            how="inner",
        )
        per_city[city] = {
            "raw": _metrics(city_df["actual"].values, city_df["pred_p50"].values),
            "naive": _metrics(city_naive["actual"].values, city_naive["naive_pred"].values)
            if len(city_naive) > 1
            else None,
            "picp": float(city_df["in_interval"].mean()),
            "entities": sorted(city_df[entity_col].astype(str).unique().tolist()),
        }

    print("\nExtracting variable importance...")
    test_ds = TimeSeriesDataSet.from_dataset(train_ds, test_data, predict=False, stop_randomization=True)
    test_loader = test_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    raw_preds = model.predict(
        test_loader,
        mode="raw",
        trainer_kwargs={"accelerator": "cpu", "devices": 1},
    )
    interpretation = model.interpret_output(raw_preds, reduction="sum")

    def _to_dict(imp_data, names):
        if isinstance(imp_data, torch.Tensor):
            return {name: imp_data[i].item() for i, name in enumerate(names)}
        if isinstance(imp_data, dict):
            return {k: (v.item() if hasattr(v, "item") else v) for k, v in imp_data.items()}
        return {}

    encoder_importance_dict = _to_dict(interpretation["encoder_variables"], model.encoder_variables)
    decoder_importance_dict = _to_dict(interpretation["decoder_variables"], model.decoder_variables)

    print("\n" + "=" * 60)
    print("VARIABLE IMPORTANCE (VSN)")
    print("=" * 60)
    print("\nEncoder Variables:")
    for var, score in sorted(encoder_importance_dict.items(), key=lambda x: -x[1]):
        print(f"  {var}: {score:.4f}")
    print("\nDecoder Variables:")
    for var, score in sorted(decoder_importance_dict.items(), key=lambda x: -x[1]):
        print(f"  {var}: {score:.4f}")

    os.makedirs("results", exist_ok=True)

    # Plot 1: variable importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, imp, title, color in [
        (axes[0], encoder_importance_dict, "Encoder Variable Importance", "steelblue"),
        (axes[1], decoder_importance_dict, "Decoder Variable Importance", "darkorange"),
    ]:
        names = list(imp.keys())
        values = list(imp.values())
        ax.barh(names, values, color=color)
        ax.set_xlabel("Importance Score")
        ax.set_title(title)
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/variable_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: global scatter
    pred_flat = df_final["pred_p50"].values
    actual_flat = df_final["actual"].values
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual_flat, pred_flat, alpha=0.3, s=10)
    lims = [
        min(actual_flat.min(), pred_flat.min()),
        max(actual_flat.max(), pred_flat.max()),
    ]
    ax.plot(lims, lims, "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual SPEI-3")
    ax.set_ylabel("Predicted SPEI-3")
    ax.set_title(
        f"TFT SPEI Prediction\nRMSE={overall_raw['rmse']:.4f}, Corr={overall_raw['pearson_r']:.4f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/prediction_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: timeseries sample
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

    # Plot 4: error distribution
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

    metrics_payload = {
        "checkpoint": checkpoint_path,
        "quantiles": quantiles,
        "quantile_index_map": qidx,
        "overall_raw": overall_raw,
        "overall_picp": picp_overall,
        "overall_naive": naive_raw,
        # Keep legacy name for compatibility with existing scripts.
        "per_location": per_entity,
        "per_entity": per_entity,
        "per_city": per_city,
        "entity_key": entity_col,
        "notes": {
            "train_split": "year < 2023",
            "val_split": "year == 2023",
            "test_split": f"year >= {test_year_start}",
            "aggregation": "step-0-only",
            "picp_nominal": "0.80 for P10-P90 interval",
            "schema_version_expected": EXPECTED_SCHEMA_VERSION,
        },
    }

    df_final.to_csv("results/predictions_eval.csv", index=False)
    with open("results/evaluation_metrics_detailed.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\nSaved: results/predictions_eval.csv")
    print("Saved: results/evaluation_metrics_detailed.json")
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    return metrics_payload


if __name__ == "__main__":
    import argparse

    default_ckpt = _checkpoint_from_run_config() or "logs/checkpoints/enc90-epoch=0-val_loss=0.2000.ckpt"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_ckpt,
    )
    parser.add_argument("--test_year", type=int, default=2024)
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/spei_dataset.parquet",
    )
    args = parser.parse_args()
    evaluate_model(
        checkpoint_path=args.checkpoint,
        test_year_start=args.test_year,
        data_path=args.data_path,
    )
