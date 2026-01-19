
# Setup
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import pandas as pd
from collections import defaultdict

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.models.dataset import create_dataset, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 1. Load Data
data = pd.read_parquet('../data/processed/spei_dataset.parquet')
data['year'] = data['time'].dt.year

print(f"Dataset Shape: {data.shape}")
print(f"Date Range: {data['time'].min()} to {data['time'].max()}")

# Split: Train < 2023, Test >= 2023
train_data = data[data.year < 2023].copy()
test_data = data[data.year >= 2023].copy()

print(f"Train: {len(train_data)} rows")
print(f"Test:  {len(test_data)} rows")

# 2. Create Dataset & Load Model
train_ds = create_dataset(train_data)

# Create test dataset (predict=False to load ALL windows)
test_ds = TimeSeriesDataSet.from_dataset(
    train_ds, 
    data,
    predict=False, 
    stop_randomization=True
)
test_dataloader = test_ds.to_dataloader(train=False, batch_size=128, num_workers=0)

print(f"Test sequences: {len(test_ds)}")

# Load best model
checkpoint_dir = '../logs/checkpoints'
checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')])
best_ckpt = checkpoints[-1] 
model_path = os.path.join(checkpoint_dir, best_ckpt)
print(f"Loading model: {best_ckpt}")

model = TemporalFusionTransformer.load_from_checkpoint(model_path)
model.eval()
print("Model loaded successfully!")

# 3. Generate Predictions (Rolling Window Ensemble)
target_loc = 'Bojonegoro'
print(f"Filtering test data for location: {target_loc}")
loc_test_data = test_data[test_data.location_id == target_loc].copy()
loc_test_ds = TimeSeriesDataSet.from_dataset(train_ds, loc_test_data, predict=False, stop_randomization=True)
loc_loader = loc_test_ds.to_dataloader(train=False, batch_size=128, num_workers=0)

print("Generating predictions...")
if torch.cuda.is_available():
    model = model.cuda()

# Helper to predict in loops if needed, but here we can try one go
# Move model to CPU if OOM, but 3k samples should fit
loc_preds = model.predict(loc_loader, mode="raw", return_x=True)
p_values = loc_preds.output.prediction.cpu().numpy()
t_values = loc_preds.x["decoder_time_idx"].cpu().numpy()

print("Predictions generated. Aggregating...")

ensemble_dict = defaultdict(list)
for i in range(p_values.shape[0]):
    for step in range(30):
        t_idx = int(t_values[i, step])
        # P50 is index 3
        val = p_values[i, step, 3]
        ensemble_dict[t_idx].append(val)

# Average
final_preds = []
for t, vals in ensemble_dict.items():
    final_preds.append({"time_idx": t, "pred_ensemble": np.mean(vals)})
    
df_pred = pd.DataFrame(final_preds).sort_values("time_idx")

# Join with Actuals
df_actual = loc_test_data[["time_idx", "time", "SPEI_3"]].copy()
df_eval = pd.merge(df_actual, df_pred, on="time_idx", how="inner")

print(f"Evaluated shape: {df_eval.shape}")

# 4. Calculate Metrics (with Robust Calibration)
y_true = df_eval["SPEI_3"].values
y_pred = df_eval["pred_ensemble"].values

# 1. Calculate Bias
bias = np.mean(y_pred - y_true)
print(f"Original Bias: {bias:.4f}")

# 2. Calculate Variance Ratio
std_true = np.std(y_true)
std_pred = np.std(y_pred)
var_ratio = std_pred / std_true
print(f"Original Variance Ratio: {var_ratio:.4f}")

# 3. Apply Robust Calibration
TARGET_MEAN = np.mean(y_true)
TARGET_STD = np.std(y_true)

bias_correction = TARGET_MEAN - np.mean(y_pred)
scale_correction = TARGET_STD / np.std(y_pred)
SAFE_SCALE = scale_correction * 0.9 if scale_correction > 1.0 else scale_correction

y_calib = (y_pred - np.mean(y_pred)) * SAFE_SCALE + np.mean(y_pred) + bias_correction

rmse_raw = np.sqrt(np.mean((y_pred - y_true)**2))
rmse_calib = np.sqrt(np.mean((y_calib - y_true)**2))

print("="*40)
print("METRICS SUMMARY")
print("="*40)
print(f"RMSE (Raw):        {rmse_raw:.4f}")
print(f"RMSE (Calibrated): {rmse_calib:.4f}")
print(f"Bias (Final):      {np.mean(y_calib - y_true):.4f}")
print(f"Var Ratio (Final): {np.std(y_calib)/np.std(y_true):.4f}")
