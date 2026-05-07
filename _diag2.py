"""Deep diagnostic: distribution shift + calibration + target_scale checks."""

import warnings

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from sklearn.metrics import mean_squared_error

from src.models.dataset import MODEL_GROUP_COL, create_dataset

torch.set_float32_matmul_precision("medium")
warnings.filterwarnings("ignore")

data = pd.read_parquet("data/processed/spei_dataset.parquet")
data["year"] = pd.to_datetime(data["time"]).dt.year
entity_col = MODEL_GROUP_COL if MODEL_GROUP_COL in data.columns else "location_id"

print("=" * 70)
print("1. TRAIN vs VAL vs TEST per-entity SPEI-3 distribution")
print("=" * 70)
for ent in sorted(data[entity_col].astype(str).unique()):
    for split, mask in [
        ("Train <2023", data.year < 2023),
        ("Val  =2023", data.year == 2023),
        ("Test >=2024", data.year >= 2024),
    ]:
        s = data[mask & (data[entity_col].astype(str) == ent)]["SPEI_3"]
        print(
            f"  {ent:16} {split} mean={s.mean():.4f} std={s.std():.4f} "
            f"min={s.min():.3f} max={s.max():.3f} n={len(s)}"
        )
    print()

print("=" * 70)
print("2. MODEL PERFORMANCE ON VALIDATION/TEST (step-0-only)")
print("=" * 70)

ckpt = "logs/checkpoints/enc90-epoch=1-val_loss=0.1839.ckpt"
model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu")
model.eval()
enc_len = int(getattr(model.hparams, "max_encoder_length", 90))
pred_len = int(getattr(model.hparams, "max_prediction_length", 30))

train_data = data[data.year < 2023].copy()
val_data = data[data.year == 2023].copy()
test_data = data[data.year >= 2024].copy()
train_ds = create_dataset(train_data, max_encoder_length=enc_len, max_prediction_length=pred_len)


def eval_split(split_data, split_name):
    all_rows = []
    for ent in sorted(split_data[entity_col].astype(str).unique()):
        loc_data = split_data[split_data[entity_col].astype(str) == ent].copy()
        loc_ds = TimeSeriesDataSet.from_dataset(train_ds, loc_data, predict=False, stop_randomization=True)
        loader = loc_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        raw = model.predict(loader, mode="raw", return_x=True)
        pv = raw.output.prediction.cpu().numpy()
        tv = raw.x["decoder_time_idx"].cpu().numpy()
        step0 = {}
        for i in range(pv.shape[0]):
            t = int(tv[i, 0])
            if t not in step0:
                step0[t] = float(pv[i, 0, 3])
        for t in sorted(step0):
            all_rows.append({"time_idx": t, entity_col: ent, "pred_p50": step0[t]})

    df_preds = pd.DataFrame(all_rows)
    df_actual = split_data[["time_idx", entity_col, "SPEI_3"]].rename(columns={"SPEI_3": "actual"})
    df_eval = pd.merge(df_actual, df_preds, on=["time_idx", entity_col], how="inner")

    print(f"\n  {split_name}: {len(df_eval)} samples")
    overall_rmse = np.sqrt(mean_squared_error(df_eval["actual"], df_eval["pred_p50"]))
    overall_bias = (df_eval["pred_p50"] - df_eval["actual"]).mean()
    print(f"  Overall RMSE={overall_rmse:.4f}  Bias={overall_bias:.4f}")

    biases = {}
    for ent in sorted(df_eval[entity_col].astype(str).unique()):
        sub = df_eval[df_eval[entity_col].astype(str) == ent]
        rmse = np.sqrt(mean_squared_error(sub["actual"], sub["pred_p50"]))
        bias = (sub["pred_p50"] - sub["actual"]).mean()
        biases[ent] = bias
        print(f"    {ent:16} RMSE={rmse:.4f}  Bias={bias:.4f}")

    return df_eval, biases


print("\n--- VAL SET (2023) ---")
df_val, val_biases = eval_split(val_data, "Validation 2023")
print("\n--- TEST SET (2024+) ---")
df_test, test_biases = eval_split(test_data, "Test 2024+")

print("\n" + "=" * 70)
print("3. POST-HOC CALIBRATION (bias from val, applied to test)")
print("=" * 70)
print("  Per-entity bias on VAL:", {k: round(v, 4) for k, v in val_biases.items()})

df_test_cal = df_test.copy()
for ent, bias in val_biases.items():
    mask = df_test_cal[entity_col].astype(str) == ent
    df_test_cal.loc[mask, "pred_p50_cal"] = df_test_cal.loc[mask, "pred_p50"] - bias

ts = test_data.sort_values([entity_col, "time_idx"]).copy()
ts["naive"] = ts.groupby(entity_col)["SPEI_3"].shift(1)
df_n = pd.merge(
    df_test[["time_idx", entity_col, "actual"]],
    ts[["time_idx", entity_col, "naive"]].dropna(),
    on=["time_idx", entity_col],
    how="inner",
)

cal_rmse = np.sqrt(mean_squared_error(df_test_cal["actual"], df_test_cal["pred_p50_cal"]))
naive_rmse = np.sqrt(mean_squared_error(df_n["actual"], df_n["naive"]))
print(f"\n  Raw model  RMSE={np.sqrt(mean_squared_error(df_test['actual'], df_test['pred_p50'])):.4f}")
print(f"  Calibrated RMSE={cal_rmse:.4f}")
print(f"  Naive      RMSE={naive_rmse:.4f}")
print(f"  Calibrated beats naive? {cal_rmse < naive_rmse}")

print("\n" + "=" * 70)
print("4. TARGET_SCALE ANALYSIS")
print("=" * 70)
target_entity = sorted(test_data[entity_col].astype(str).unique().tolist())[0]
loc_data = test_data[test_data[entity_col].astype(str) == target_entity].copy()
loc_ds = TimeSeriesDataSet.from_dataset(train_ds, loc_data, predict=False, stop_randomization=True)
loader = loc_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
raw = model.predict(loader, mode="raw", return_x=True)
tscale = raw.x["target_scale"].cpu().numpy()
pv = raw.output.prediction.cpu().numpy()
print(f"Entity: {target_entity}")
print(f"  target_scale shape: {tscale.shape}")
print(
    f"  center mean/std: {tscale[:,0].mean():.4f}/{tscale[:,0].std():.4f} | "
    f"scale mean/std: {tscale[:,1].mean():.4f}/{tscale[:,1].std():.4f}"
)
normalized_pred = (pv[:, 0, 3] - tscale[:, 0]) / tscale[:, 1]
print(f"  pred_p50 denorm mean/std: {pv[:,0,3].mean():.4f}/{pv[:,0,3].std():.4f}")
print(f"  pred_p50 norm   mean/std: {normalized_pred.mean():.4f}/{normalized_pred.std():.4f}")
