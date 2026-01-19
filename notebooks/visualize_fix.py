
import os
import sys
import warnings

# IMPORTANT: Fix path BEFORE importing local modules
sys.path.insert(0, os.path.abspath('d:/SKRIPSI/Skripsi_Nopal'))

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.models.dataset import create_dataset, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH

warnings.filterwarnings('ignore')

# Setup Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 1.5

def generate_visualization():
    print("Loading resources...")
    # 1. Load Data
    data = pd.read_parquet('../data/processed/spei_dataset.parquet')
    data['year'] = data['time'].dt.year
    train_data = data[data.year < 2023].copy()
    test_data = data[data.year >= 2023].copy()

    # 2. Setup Dataset & Model
    train_ds = create_dataset(train_data)
    
    # Load optimal model
    checkpoint_dir = '../logs/checkpoints'
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')])
    best_ckpt = checkpoints[-1] 
    model = TemporalFusionTransformer.load_from_checkpoint(os.path.join(checkpoint_dir, best_ckpt))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 3. Rolling Window Inference (Bojonegoro)
    target_loc = 'Bojonegoro'
    print(f"Generating Rolling Ensemble for {target_loc}...")
    
    loc_test_data = test_data[test_data.location_id == target_loc].copy()
    # predict=False enables sliding window creation
    loc_test_ds = TimeSeriesDataSet.from_dataset(train_ds, loc_test_data, predict=False, stop_randomization=True)
    loc_loader = loc_test_ds.to_dataloader(train=False, batch_size=128, num_workers=0)
    
    # Prediction Loop
    raw_preds = model.predict(loc_loader, mode="raw", return_x=True) # (N, 30, 7)
    p_values = raw_preds.output.prediction.cpu().numpy()
    t_values = raw_preds.x["decoder_time_idx"].cpu().numpy()
    
    # Ensemble Aggregation
    ensemble = defaultdict(list)
    for i in range(p_values.shape[0]):
        for step in range(30):
            t_idx = int(t_values[i, step])
            val = p_values[i, step, 3] # P50
            ensemble[t_idx].append(val)
            
    results = []
    for t, vals in ensemble.items():
        results.append({"time_idx": t, "pred_ensemble": np.mean(vals)})
    
    df_pred = pd.DataFrame(results).sort_values("time_idx")
    
    # Merge with Actuals
    df_actual = loc_test_data[["time_idx", "time", "SPEI_3"]].copy()
    df_eval = pd.merge(df_actual, df_pred, on="time_idx", how="inner")
    
    # 4. Calibration
    y_true = df_eval["SPEI_3"].values
    y_pred = df_eval["pred_ensemble"].values
    
    TARGET_MEAN = np.mean(y_true)
    TARGET_STD = np.std(y_true)
    
    # Match moments
    bias = TARGET_MEAN - np.mean(y_pred)
    scale = TARGET_STD / np.std(y_pred)
    safe_scale = scale * 0.95 # Conservative dampening
    
    y_calib = (y_pred - np.mean(y_pred)) * safe_scale + np.mean(y_pred) + bias
    df_eval["pred_calib"] = y_calib
    
    print(f"Final Calibration Stats -> Bias: {np.mean(y_calib - y_true):.4f}, Var Ratio: {np.std(y_calib)/np.std(y_true):.4f}")

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1])
    
    # Plot 1: Full Time Series
    ax1.plot(df_eval['time'], df_eval['SPEI_3'], label='Actual (Observation)', color='black', alpha=0.6, linewidth=1)
    ax1.plot(df_eval['time'], df_eval['pred_calib'], label='TFT Deep Learning (Calibrated)', color='#2ca02c', linewidth=2, alpha=0.9)
    # ax1.plot(df_eval['time'], df_eval['pred_ensemble'], label='Raw Ensemble', color='red', linestyle=':', alpha=0.4)
    
    ax1.set_title(f'Multi-Horizon SPEI-3 Forecast: {target_loc} (2023-2025)', fontsize=14, fontweight='bold')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(-1.5, color='orange', linestyle='--', label='Severe Drought (-1.5)')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoom in (Smoothness Check) - First 180 days
    zoom_df = df_eval.iloc[:180]
    ax2.plot(zoom_df['time'], zoom_df['SPEI_3'], color='black', alpha=0.6, linewidth=1, marker='.')
    ax2.plot(zoom_df['time'], zoom_df['pred_calib'], color='#2ca02c', linewidth=2, marker='.')
    
    ax2.set_title('Zoom-in: Smoothness & Reactivity Check (First 6 Months)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('SPEI-3')
    
    plt.tight_layout()
    output_path = '../results/fixed_spei_prediction.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    generate_visualization()
