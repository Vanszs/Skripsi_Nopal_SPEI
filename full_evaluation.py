"""
full_evaluation.py
==================
Comprehensive TFT evaluation with schema-v2 support and dynamic plotting.
"""

import argparse
import json
import math
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but StandardScaler was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=UserWarning,
)
torch.set_float32_matmul_precision("medium")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.data.spei import classify_spei
from src.models.dataset import MODEL_GROUP_COL, create_dataset

EXPECTED_SCHEMA_VERSION = 2
sns.set_theme(style="whitegrid", font_scale=1.0)


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
    return "Kekeringan" if "Kekeringan" in c else ("Basah" if "Basah" in c else "Normal")


def _entity_col(df: pd.DataFrame) -> str:
    if MODEL_GROUP_COL in df.columns:
        return MODEL_GROUP_COL
    if "location_id" in df.columns:
        return "location_id"
    raise ValueError("No valid entity key found in dataset.")


def _palette(keys):
    colors = sns.color_palette("tab20", n_colors=max(3, len(keys)))
    return {key: colors[i % len(colors)] for i, key in enumerate(keys)}


def _grid(n, max_cols=3):
    cols = min(max_cols, max(1, n))
    rows = math.ceil(n / cols)
    return rows, cols


def _best_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
    scored = []
    for p in ckpts:
        m = re.search(r"val_loss=(\d+\.\d+)", p.name)
        if m:
            scored.append((float(m.group(1)), p))
    if scored:
        scored.sort(key=lambda x: x[0])
        return scored[0][1]
    return ckpts[-1]


def _checkpoint_from_run_config() -> Path | None:
    cfg_path = ROOT / "logs" / "run_config.json"
    if not cfg_path.exists():
        return None
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        ckpt = payload.get("best_model_path")
        if ckpt and Path(ckpt).exists():
            return Path(ckpt)
    except Exception:
        return None
    return None


def _quantile_index_map(model):
    quantiles = np.array([float(q) for q in getattr(model.loss, "quantiles", [0.1, 0.5, 0.9])])
    idx_p10 = int(np.argmin(np.abs(quantiles - 0.10)))
    idx_p50 = int(np.argmin(np.abs(quantiles - 0.50)))
    idx_p90 = int(np.argmin(np.abs(quantiles - 0.90)))
    return quantiles.tolist(), {"p10": idx_p10, "p50": idx_p50, "p90": idx_p90}


def run(checkpoint_path: str, out_dir: Path, log_fp):
    _log("=" * 72, log_fp)
    _log("  TFT SPEI-3 FORECASTING - COMPREHENSIVE EVALUATION", log_fp)
    _log(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_fp)
    _log("=" * 72, log_fp)

    data_path = ROOT / "data/processed/spei_dataset.parquet"
    if not data_path.exists():
        _log(f"ERROR: {data_path} not found.", log_fp)
        return

    data = pd.read_parquet(data_path)
    data["time"] = pd.to_datetime(data["time"])
    data["year"] = data["time"].dt.year
    entity_col = _entity_col(data)

    if "schema_version" in data.columns:
        versions = sorted(data["schema_version"].dropna().unique().tolist())
        if versions != [EXPECTED_SCHEMA_VERSION]:
            raise ValueError(
                f"Processed schema version {versions}, expected [{EXPECTED_SCHEMA_VERSION}]"
            )

    _log(f"Rows      : {len(data):,}", log_fp)
    _log(f"Period    : {data['time'].min().date()} -> {data['time'].max().date()}", log_fp)
    _log(f"Entity key: {entity_col}", log_fp)
    _log(f"Entities  : {data[entity_col].nunique()}", log_fp)
    _log(f"Cities    : {data['city_id'].nunique()}", log_fp)

    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    quantiles, qidx = _quantile_index_map(model)

    enc_len = int(getattr(model.hparams, "max_encoder_length", 90))
    pred_len = int(getattr(model.hparams, "max_prediction_length", 30))
    _log(f"Checkpoint: {checkpoint_path}", log_fp)
    _log(f"enc_len={enc_len} pred_len={pred_len}", log_fp)

    train_data = data[data.year < 2023].copy()
    test_data = data[data.year >= 2024].copy()
    if test_data.empty:
        _log("ERROR: test_data empty (year>=2024).", log_fp)
        return

    entities = sorted(test_data[entity_col].astype(str).unique().tolist())
    cities = sorted(test_data["city_id"].astype(str).unique().tolist())
    entity_palette = _palette(entities)
    city_palette = _palette(cities)

    train_ds = create_dataset(train_data, max_encoder_length=enc_len, max_prediction_length=pred_len)

    # Build ground truth lookup for horizon metrics.
    actual_lookup = {
        (int(r.time_idx), str(getattr(r, entity_col))): float(r.SPEI_3)
        for r in test_data.itertuples(index=False)
    }

    ensemble_rows = []
    horizon_preds = {h: {} for h in range(pred_len)}
    for ent in entities:
        _log(f"Processing {ent} ...", log_fp)
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

        step0 = {}
        for i in range(pv.shape[0]):
            t_idx = int(tv[i, 0])
            if t_idx not in step0:
                step0[t_idx] = {
                    "pred_p10": float(pv[i, 0, qidx["p10"]]),
                    "pred_p50": float(pv[i, 0, qidx["p50"]]),
                    "pred_p90": float(pv[i, 0, qidx["p90"]]),
                }
            for h in range(pred_len):
                key = (int(tv[i, h]), ent)
                if key not in horizon_preds[h]:
                    horizon_preds[h][key] = float(pv[i, h, qidx["p50"]])

        city_id = str(loc_data["city_id"].iloc[0])
        for t in sorted(step0):
            ensemble_rows.append({"time_idx": t, entity_col: ent, "city_id": city_id, **step0[t]})

    df_preds = pd.DataFrame(ensemble_rows)
    df_actual = test_data[[entity_col, "city_id", "time_idx", "time", "SPEI_3"]].rename(
        columns={"SPEI_3": "actual"}
    )
    df = pd.merge(df_actual, df_preds, on=[entity_col, "city_id", "time_idx"], how="inner")
    df["error"] = df["pred_p50"] - df["actual"]
    df["month"] = pd.to_datetime(df["time"]).dt.to_period("M").astype(str)
    df["actual_class"] = df["actual"].apply(classify_spei)
    df["pred_class"] = df["pred_p50"].apply(classify_spei)
    df["in_interval"] = (
        (df["actual"] >= df["pred_p10"]) & (df["actual"] <= df["pred_p90"])
    ).astype(int)
    _log(f"Merged rows: {len(df):,}", log_fp)

    overall = _metrics(df["actual"].values, df["pred_p50"].values)
    picp_overall = float(df["in_interval"].mean())
    picp_per_entity = {
        ent: float(df[df[entity_col].astype(str) == ent]["in_interval"].mean()) for ent in entities
    }
    picp_per_city = {
        city: float(df[df["city_id"].astype(str) == city]["in_interval"].mean()) for city in cities
    }

    per_entity = {}
    for ent in entities:
        sub = df[df[entity_col].astype(str) == ent]
        per_entity[ent] = _metrics(sub["actual"].values, sub["pred_p50"].values)

    per_city = {}
    for city in cities:
        sub = df[df["city_id"].astype(str) == city]
        per_city[city] = _metrics(sub["actual"].values, sub["pred_p50"].values)

    test_sorted = test_data.sort_values([entity_col, "time_idx"]).copy()
    test_sorted["naive_pred"] = test_sorted.groupby(entity_col)["SPEI_3"].shift(1)
    df_naive = pd.merge(
        df[[entity_col, "city_id", "time_idx", "actual"]],
        test_sorted[[entity_col, "time_idx", "naive_pred"]].dropna(),
        on=[entity_col, "time_idx"],
        how="inner",
    )
    naive_overall = _metrics(df_naive["actual"].values, df_naive["naive_pred"].values)
    naive_per_entity = {}
    for ent in entities:
        sub = df_naive[df_naive[entity_col].astype(str) == ent]
        naive_per_entity[ent] = _metrics(sub["actual"].values, sub["naive_pred"].values)
    naive_per_city = {}
    for city in cities:
        sub = df_naive[df_naive["city_id"].astype(str) == city]
        naive_per_city[city] = _metrics(sub["actual"].values, sub["naive_pred"].values)

    # Horizon metrics + fair naive horizon (same timestamp/entity subset as model horizon).
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
    horizon_rows = []
    for h in range(pred_len):
        actuals_h = []
        preds_h = []
        for key, pred_val in horizon_preds[h].items():
            actual_val = actual_lookup.get(key)
            if actual_val is not None:
                actuals_h.append(actual_val)
                preds_h.append(pred_val)
        if len(actuals_h) >= 2:
            m = _metrics(np.array(actuals_h), np.array(preds_h))
        else:
            m = dict(rmse=None, mae=None, r2=None, bias=None, pearson_r=None, n=0)
        m["horizon"] = h + 1
        m["naive_rmse"] = naive_by_h.get(h + 1)
        m["beats_naive"] = (
            m["rmse"] < m["naive_rmse"]
            if m["rmse"] is not None and m["naive_rmse"] is not None
            else None
        )
        horizon_rows.append(m)
    df_horizon = pd.DataFrame(horizon_rows)

    # Classification reports.
    all_classes = [
        "Kekeringan Ekstrem",
        "Kekeringan Parah",
        "Kekeringan Sedang",
        "Kekeringan Ringan",
        "Normal",
        "Basah Ringan",
        "Basah Sedang",
        "Basah Parah",
        "Basah Ekstrem",
    ]
    clf_rows = []
    for ent in entities:
        sub = df[df[entity_col].astype(str) == ent]
        for cls in all_classes:
            actual_count = int((sub["actual_class"] == cls).sum())
            pred_count = int((sub["pred_class"] == cls).sum())
            correct = int(((sub["actual_class"] == cls) & (sub["pred_class"] == cls)).sum())
            precision = correct / pred_count if pred_count > 0 else 0.0
            recall = correct / actual_count if actual_count > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            clf_rows.append(
                {
                    "entity": ent,
                    "city_id": str(sub["city_id"].iloc[0]) if len(sub) else None,
                    "class": cls,
                    "actual_count": actual_count,
                    "pred_count": pred_count,
                    "correct": correct,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                }
            )
    df_clf = pd.DataFrame(clf_rows)

    clf_summary_rows = []
    for city in cities:
        sub = df[df["city_id"].astype(str) == city].copy()
        exact_acc = float((sub["actual_class"] == sub["pred_class"]).mean())
        broad_acc = float((sub["actual_class"].apply(_broad) == sub["pred_class"].apply(_broad)).mean())
        clf_summary_rows.append(
            {"city_id": city, "exact_acc": round(exact_acc, 4), "broad_acc": round(broad_acc, 4), "total": len(sub)}
        )
    df_clf_summary = pd.DataFrame(clf_summary_rows)

    # Persist outputs.
    df_horizon.to_csv(out_dir / "horizon_metrics.csv", index=False)
    df_clf.to_csv(out_dir / "classification_report.csv", index=False)
    df_clf_summary.to_csv(out_dir / "classification_summary.csv", index=False)
    df.to_csv(out_dir / "predictions_full.csv", index=False)

    metrics_payload = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "entity_key": entity_col,
        "checkpoint": str(checkpoint_path),
        "quantiles": quantiles,
        "quantile_index_map": qidx,
        "train_period": "year < 2023",
        "val_period": "year == 2023",
        "test_period": "year >= 2024",
        "prediction_length": pred_len,
        "overall": overall,
        "picp_overall": picp_overall,
        "picp_per_entity": picp_per_entity,
        "picp_per_city": picp_per_city,
        "picp_per_location": picp_per_city,
        "naive_persistence": naive_overall,
        # Backward compatibility key
        "per_location": per_city,
        "per_entity": per_entity,
        "per_city": per_city,
        "naive_per_entity": naive_per_entity,
        "naive_per_city": naive_per_city,
        "per_horizon": horizon_rows,
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    _log("Saved core CSV/JSON artifacts.", log_fp)

    # -------------------- Plots --------------------
    # 01 Overall scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    for city in cities:
        sub = df[df["city_id"].astype(str) == city]
        ax.scatter(sub["actual"], sub["pred_p50"], alpha=0.3, s=8, color=city_palette[city], label=city)
    ax.plot([-3.5, 3.5], [-3.5, 3.5], "k--", lw=1.2)
    ax.set_xlim([-3.5, 3.5])
    ax.set_ylim([-3.5, 3.5])
    ax.set_xlabel("Actual SPEI-3")
    ax.set_ylabel("Predicted P50")
    ax.set_title(
        f"Actual vs Predicted - All Cities\n"
        f"RMSE={overall['rmse']:.3f} MAE={overall['mae']:.3f} R2={overall['r2']:.3f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "01_scatter_overall.png", dpi=150)
    plt.close(fig)

    # 02 Per-entity scatter (dynamic grid)
    rows, cols = _grid(len(entities), max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i, ent in enumerate(entities):
        sub = df[df[entity_col].astype(str) == ent]
        m = per_entity[ent]
        ax = axes[i]
        ax.scatter(sub["actual"], sub["pred_p50"], alpha=0.3, s=8, color=entity_palette[ent])
        ax.plot([-3.5, 3.5], [-3.5, 3.5], "k--", lw=1)
        ax.set_xlim([-3.5, 3.5])
        ax.set_ylim([-3.5, 3.5])
        ax.set_title(f"{ent}\nRMSE={m['rmse']:.3f} R2={m['r2']:.3f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Pred P50")
        ax.grid(True, alpha=0.3)
    for j in range(len(entities), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Per-Entity Scatter", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "02_scatter_per_location.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 03 Time-series per city
    rows, cols = _grid(len(cities), max_cols=2)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)
    for i, city in enumerate(cities):
        sub = (
            df[df["city_id"].astype(str) == city]
            .groupby("time", as_index=False)[["actual", "pred_p10", "pred_p50", "pred_p90"]]
            .mean()
            .sort_values("time")
        )
        n = min(365, len(sub))
        sub = sub.iloc[:n]
        ax = axes[i]
        ax.fill_between(range(n), sub["pred_p10"], sub["pred_p90"], alpha=0.2, color=city_palette[city])
        ax.plot(range(n), sub["actual"], "k-", lw=1.2, label="Actual")
        ax.plot(range(n), sub["pred_p50"], "--", lw=1.2, color=city_palette[city], label="Pred P50")
        ax.set_title(f"{city} - first {n} test days")
        ax.set_ylabel("SPEI-3")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for j in range(len(cities), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Time Series per City", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "03_timeseries_per_location.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 04 Error distribution per city
    rows, cols = _grid(len(cities), max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)
    for i, city in enumerate(cities):
        sub = df[df["city_id"].astype(str) == city]["error"].dropna()
        ax = axes[i]
        ax.hist(sub, bins=40, color=city_palette[city], alpha=0.7, edgecolor="white", density=True)
        ax.axvline(0, color="red", ls="--", lw=1)
        ax.axvline(sub.mean(), color="black", ls="-", lw=1)
        ax.set_title(f"{city} | mean={sub.mean():.3f} std={sub.std():.3f}")
        ax.grid(True, alpha=0.3)
    for j in range(len(cities), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Error Distribution per City", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "04_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 05 Variable importance
    try:
        test_ds_vi = TimeSeriesDataSet.from_dataset(train_ds, test_data, predict=False, stop_randomization=True)
        vi_loader = test_ds_vi.to_dataloader(train=False, batch_size=64, num_workers=0)
        raw_vi = model.predict(
            vi_loader,
            mode="raw",
            trainer_kwargs={"accelerator": "cpu", "devices": 1},
        )
        interp = model.interpret_output(raw_vi, reduction="sum")

        def _to_dict(imp, names):
            if isinstance(imp, torch.Tensor):
                return {n: imp[j].item() for j, n in enumerate(names)}
            return {k: (v.item() if hasattr(v, "item") else v) for k, v in imp.items()}

        enc_imp = _to_dict(interp["encoder_variables"], model.encoder_variables)
        dec_imp = _to_dict(interp["decoder_variables"], model.decoder_variables)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, imp_dict, title, color in [
            (axes[0], enc_imp, "Encoder", "steelblue"),
            (axes[1], dec_imp, "Decoder", "darkorange"),
        ]:
            names = list(imp_dict.keys())
            vals = [imp_dict[n] for n in names]
            order = np.argsort(vals)
            ax.barh([names[j] for j in order], [vals[j] for j in order], color=color)
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="x")
        fig.suptitle("Variable Importance")
        fig.tight_layout()
        fig.savefig(out_dir / "05_variable_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        _log(f"05_variable_importance skipped: {exc}", log_fp)

    # 06 Horizon metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_metrics = [
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("bias", "Bias"),
        ("pearson_r", "Pearson r"),
    ]
    for ax, (col, title) in zip(axes.reshape(-1), plot_metrics):
        valid = df_horizon.dropna(subset=[col])
        ax.plot(valid["horizon"], valid[col], marker="o", lw=1.5, label="TFT")
        if col == "rmse":
            naive_valid = df_horizon.dropna(subset=["naive_rmse"])
            ax.plot(naive_valid["horizon"], naive_valid["naive_rmse"], "k--", lw=1.5, label="Naive")
        ax.axhline(0, color="black", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Horizon")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Horizon Metrics")
    fig.tight_layout()
    fig.savefig(out_dir / "06_horizon_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 07 City comparison
    x = np.arange(len(cities))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for j, met in enumerate(["rmse", "mae", "pearson_r"]):
        vals = [per_city[c][met] if per_city[c][met] is not None else 0.0 for c in cities]
        ax.bar(x + j * width, vals, width, label=met.upper())
    ax.set_xticks(x + width)
    ax.set_xticklabels(cities, rotation=15)
    ax.set_title("Per-City Metric Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "07_location_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 08 Quantile fan per city
    rows, cols = _grid(len(cities), max_cols=2)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)
    for i, city in enumerate(cities):
        sub = (
            df[df["city_id"].astype(str) == city]
            .groupby("time", as_index=False)[["actual", "pred_p10", "pred_p50", "pred_p90"]]
            .mean()
            .sort_values("time")
        )
        n = min(180, len(sub))
        sub = sub.iloc[:n]
        dates = pd.to_datetime(sub["time"])
        ax = axes[i]
        ax.fill_between(dates, sub["pred_p10"], sub["pred_p90"], alpha=0.25, color=city_palette[city], label="P10-P90")
        ax.plot(dates, sub["pred_p50"], "-", color=city_palette[city], label="Pred P50")
        ax.plot(dates, sub["actual"], "k-", lw=1.2, label="Actual")
        ax.set_title(city)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(len(cities), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Quantile Fan (Per City)")
    fig.tight_layout()
    fig.savefig(out_dir / "08_quantile_fan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 09 Broad confusion matrix
    df["ab"] = df["actual_class"].apply(_broad)
    df["pb"] = df["pred_class"].apply(_broad)
    cats = ["Kekeringan", "Normal", "Basah"]
    conf = pd.crosstab(df["ab"], df["pb"], rownames=["Actual"], colnames=["Predicted"])
    conf = conf.reindex(index=cats, columns=cats, fill_value=0)
    conf_pct = conf.div(conf.sum(axis=1), axis=0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=axes[0])
    axes[0].set_title("Confusion Count (3-class)")
    sns.heatmap(conf_pct, annot=True, fmt=".1f", cmap="Greens", linewidths=0.5, ax=axes[1], vmin=0, vmax=100)
    axes[1].set_title("Confusion Row %")
    fig.tight_layout()
    fig.savefig(out_dir / "09_spei_classification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 10 Bias over time per city
    month_bias = (
        df.groupby(["city_id", "month"])["error"]
        .mean()
        .reset_index()
        .rename(columns={"error": "bias"})
    )
    month_bias["month_dt"] = pd.to_datetime(month_bias["month"])
    fig, ax = plt.subplots(figsize=(14, 5))
    for city in cities:
        sub = month_bias[month_bias["city_id"].astype(str) == city].sort_values("month_dt")
        ax.plot(sub["month_dt"], sub["bias"], marker="o", markersize=3, lw=1.5, label=city, color=city_palette[city])
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_title("Monthly Bias by City")
    ax.set_ylabel("Mean Bias")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "10_bias_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 11 Model vs Naive + PICP per city
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rmse_model = [per_city[c]["rmse"] if per_city[c]["rmse"] is not None else 0.0 for c in cities]
    rmse_naive = [
        naive_per_city[c]["rmse"] if naive_per_city[c]["rmse"] is not None else 0.0 for c in cities
    ]
    axes[0].bar(x - 0.18, rmse_model, 0.36, label="TFT", color="steelblue", alpha=0.85)
    axes[0].bar(x + 0.18, rmse_naive, 0.36, label="Naive", color="tomato", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cities, rotation=15, ha="right")
    axes[0].set_title("RMSE: Model vs Naive")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend()

    picp_vals = [picp_per_city.get(city, 0.0) for city in cities]
    axes[1].bar(x, picp_vals, color="seagreen", alpha=0.85)
    axes[1].axhline(0.80, color="black", ls="--", lw=1.3, label="Nominal 80%")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cities, rotation=15, ha="right")
    axes[1].set_ylim([0, 1.05])
    axes[1].set_title("PICP per City")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "11_model_vs_naive_picp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    _log("\nSUMMARY:", log_fp)
    _log(f"Overall RMSE={overall['rmse']:.4f} | Naive RMSE={naive_overall['rmse']:.4f}", log_fp)
    _log(f"Overall PICP={picp_overall:.4f}", log_fp)
    beat_count = int(sum(1 for row in horizon_rows if row.get("beats_naive")))
    _log(f"Horizon beats naive: {beat_count}/{pred_len}", log_fp)


def main():
    parser = argparse.ArgumentParser(description="Full TFT SPEI Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .ckpt file. Defaults to best in logs/checkpoints/",
    )
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = str(_checkpoint_from_run_config() or _best_checkpoint(ROOT / "logs/checkpoints"))

    print(f"Checkpoint : {ckpt_path}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"results/full_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics_report.txt"
    with open(log_path, "w", encoding="utf-8") as log_fp:
        run(ckpt_path, out_dir, log_fp)

    print(f"\nDone. Results -> {out_dir}")


if __name__ == "__main__":
    main()
