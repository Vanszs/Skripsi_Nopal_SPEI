"""Deep diagnostic: distribution shift + calibration test + root cause analysis."""
import torch, warnings
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from sklearn.metrics import mean_squared_error
from src.models.dataset import create_dataset

data = pd.read_parquet("data/processed/spei_dataset.parquet")
data["year"] = data["time"].dt.year

# ============================================================
# 1) Distribution shift analysis
# ============================================================
print("=" * 70)
print("1. TRAIN vs VAL vs TEST per-location SPEI-3 distribution")
print("=" * 70)
for loc in sorted(data["location_id"].unique()):
    for split, mask in [("Train <2023", data.year < 2023),
                        ("Val  =2023", data.year == 2023),
                        ("Test >=2024", data.year >= 2024)]:
        s = data[mask & (data.location_id == loc)]["SPEI_3"]
        print(f"  {loc:12} {split}  mean={s.mean():.4f}  std={s.std():.4f}  min={s.min():.3f}  max={s.max():.3f}  n={len(s)}")
    print()

# ============================================================
# 2) Evaluate enc90-epoch=1 on VALIDATION set (2023)
# ============================================================
print("=" * 70)
print("2. MODEL PERFORMANCE ON VALIDATION SET (2023) — enc90-epoch=1")
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

def eval_split(model, split_data, train_ds, split_name):
    """Evaluate step-0-only on a data split."""
    all_rows = []
    for loc in sorted(split_data["location_id"].unique()):
        loc_data = split_data[split_data.location_id == loc].copy()
        try:
            loc_ds = TimeSeriesDataSet.from_dataset(train_ds, loc_data, predict=False, stop_randomization=True)
            loader = loc_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
            raw = model.predict(loader, mode="raw", return_x=True)
            pv = raw.output.prediction.cpu().numpy()
            tv = raw.x["decoder_time_idx"].cpu().numpy()
            
            step0 = {}
            for i in range(pv.shape[0]):
                t = int(tv[i, 0])
                if t not in step0:
                    step0[t] = float(pv[i, 0, 3])  # p50
            
            for t in sorted(step0):
                all_rows.append({"time_idx": t, "location_id": loc, "pred_p50": step0[t]})
        except Exception as e:
            print(f"  {loc}: Error - {e}")
    
    df_preds = pd.DataFrame(all_rows)
    df_actual = split_data[["time_idx", "location_id", "SPEI_3"]].rename(columns={"SPEI_3": "actual"})
    df = pd.merge(df_actual, df_preds, on=["time_idx", "location_id"], how="inner")
    
    print(f"\n  {split_name}: {len(df)} samples")
    overall_rmse = np.sqrt(mean_squared_error(df["actual"], df["pred_p50"]))
    overall_bias = (df["pred_p50"] - df["actual"]).mean()
    print(f"  Overall RMSE={overall_rmse:.4f}  Bias={overall_bias:.4f}")
    
    biases = {}
    for loc in sorted(df["location_id"].unique()):
        sub = df[df.location_id == loc]
        rmse = np.sqrt(mean_squared_error(sub["actual"], sub["pred_p50"]))
        bias = (sub["pred_p50"] - sub["actual"]).mean()
        biases[loc] = bias
        print(f"    {loc:12} RMSE={rmse:.4f}  Bias={bias:.4f}")
    
    return df, biases

print("\n--- VAL SET (2023) ---")
df_val, val_biases = eval_split(model, val_data, train_ds, "Validation 2023")

print("\n--- TEST SET (2024+) ---")
df_test, test_biases = eval_split(model, test_data, train_ds, "Test 2024+")

# ============================================================
# 3) Calibration: remove bias learned from validation
# ============================================================
print("\n" + "=" * 70)
print("3. POST-HOC CALIBRATION (bias from val, applied to test)")
print("=" * 70)
print("  Per-location bias on VAL:", {k: round(v, 4) for k, v in val_biases.items()})

# Apply calibration
df_test_cal = df_test.copy()
for loc, bias in val_biases.items():
    mask = df_test_cal.location_id == loc
    df_test_cal.loc[mask, "pred_p50_cal"] = df_test_cal.loc[mask, "pred_p50"] - bias

cal_rmse = np.sqrt(mean_squared_error(df_test_cal["actual"], df_test_cal["pred_p50_cal"]))
cal_bias = (df_test_cal["pred_p50_cal"] - df_test_cal["actual"]).mean()

# Naive
ts = test_data.sort_values(["location_id", "time_idx"]).copy()
ts["naive"] = ts.groupby("location_id")["SPEI_3"].shift(1)
df_n = pd.merge(df_test[["time_idx", "location_id", "actual"]],
                ts[["time_idx", "location_id", "naive"]].dropna(),
                on=["time_idx", "location_id"], how="inner")
naive_rmse = np.sqrt(mean_squared_error(df_n["actual"], df_n["naive"]))

print(f"\n  Raw model  RMSE={np.sqrt(mean_squared_error(df_test['actual'], df_test['pred_p50'])):.4f}")
print(f"  Calibrated RMSE={cal_rmse:.4f}  Bias={cal_bias:.4f}")
print(f"  Naive      RMSE={naive_rmse:.4f}")
print(f"  Calibrated beats naive? {cal_rmse < naive_rmse}")

for loc in sorted(df_test_cal["location_id"].unique()):
    sub = df_test_cal[df_test_cal.location_id == loc]
    sub_n = df_n[df_n.location_id == loc]
    mr = np.sqrt(mean_squared_error(sub["actual"], sub["pred_p50_cal"]))
    nr = np.sqrt(mean_squared_error(sub_n["actual"], sub_n["naive"]))
    print(f"    {loc:12} Cal_RMSE={mr:.4f}  Naive_RMSE={nr:.4f}  beats={'YES' if mr<nr else 'NO '}")

# ============================================================
# 4) Check target_scale from GroupNormalizer 
# ============================================================
print("\n" + "=" * 70)
print("4. TARGET_SCALE ANALYSIS (GroupNormalizer internals)")
print("=" * 70)

loc_data = test_data[test_data.location_id == "Bojonegoro"].copy()
loc_ds = TimeSeriesDataSet.from_dataset(train_ds, loc_data, predict=False, stop_randomization=True)
loader = loc_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
raw = model.predict(loader, mode="raw", return_x=True)

ts = raw.x["target_scale"].cpu().numpy()  # (B, 2) - center, scale
print(f"  target_scale shape: {ts.shape}")
print(f"  center (col 0): mean={ts[:, 0].mean():.4f}, std={ts[:, 0].std():.4f}, min={ts[:, 0].min():.4f}, max={ts[:, 0].max():.4f}")
print(f"  scale  (col 1): mean={ts[:, 1].mean():.4f}, std={ts[:, 1].std():.4f}, min={ts[:, 1].min():.4f}, max={ts[:, 1].max():.4f}")

# Check prediction in normalized vs denormalized space
pv = raw.output.prediction.cpu().numpy()  # (B, T, 7) - denormalized
print(f"\n  pred_p50 (denorm): mean={pv[:, 0, 3].mean():.4f}, std={pv[:, 0, 3].std():.4f}")

# What would the model predict if center/scale were (0, 1)?
# The model's raw output before denorm is: normalized_pred = (denorm_pred - center) / scale
normalized_pred = (pv[:, 0, 3] - ts[:, 0]) / ts[:, 1]
print(f"  pred_p50 (norm):   mean={normalized_pred.mean():.4f}, std={normalized_pred.std():.4f}")
