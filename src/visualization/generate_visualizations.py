
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.dataset import create_dataset

# Configuration
DATA_PATH = "data/processed/spei_dataset.parquet"
CHECKPOINT_DIR = "logs/checkpoints"
RESULTS_DIR = "results"
FIG_SIZE = (12, 6)
sns.set_style("whitegrid")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    if not os.path.exists(DATA_PATH):
        # Try adjusting path relative to script execution
        alt_path = os.path.join("..", "..", DATA_PATH)
        if os.path.exists(alt_path):
            return pd.read_parquet(alt_path)
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    return pd.read_parquet(DATA_PATH)

def find_checkpoint():
    """Dynamically find the best checkpoint (lowest val_loss) in CHECKPOINT_DIR."""
    if not os.path.exists(CHECKPOINT_DIR):
        raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".ckpt")]
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found in {CHECKPOINT_DIR}")

    # Sort by val_loss value parsed from filename (format: epoch=N-val_loss=X.XX.ckpt)
    def parse_val_loss(fname):
        try:
            return float(fname.split("val_loss=")[1].replace(".ckpt", ""))
        except (IndexError, ValueError):
            return float("inf")

    best = min(checkpoints, key=parse_val_loss)
    return os.path.join(CHECKPOINT_DIR, best)

def generate_visualizations():
    ensure_dir(RESULTS_DIR)
    
    print("1. Loading Data...")
    data = load_data()
    print(f"Data Loaded. Shape: {data.shape}")
    
    # Define Test Period (2024-2025)
    # MAX_ENCODER_LENGTH imported from dataset.py (90 days) to stay consistent
    from src.models.dataset import MAX_ENCODER_LENGTH as MAX_ENCODER
    data_2024 = data[data["time"].dt.year >= 2024]
    
    if len(data_2024) == 0:
        print("WARNING: No data for 2024. Using last 365 days.")
        test_start_idx = data.time_idx.max() - 365
    else:
        test_start_idx = data_2024["time_idx"].min()
        
    subset_data = data[data.time_idx >= (test_start_idx - MAX_ENCODER)].copy()
    
    print("2. Preparing Dataset...")
    # Dummy training dataset for encoders
    train_ds = create_dataset(data[data["time"].dt.year < 2023])
    
    # Test Dataset
    test_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        subset_data,
        predict=False,
        stop_randomization=True
    )
    
    test_dataloader = test_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    print("3. Loading Model...")
    ckpt_path = find_checkpoint()
    print(f"Loading from: {ckpt_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    
    print("4. Generating Predictions...")
    try:
        # Standard predictions
        raw_predictions = model.predict(test_dataloader, mode="prediction", return_x=True)
        pred_values = raw_predictions.output.cpu().numpy() # [N, 30]
        actual_values = raw_predictions.x["decoder_target"].cpu().numpy() # [N, 30]
        
        # Flatten for aggregate analysis (taking step 1)
        y_true = actual_values[:, 0]
        y_pred = pred_values[:, 0]
        
        # --- Plot 1: Prediction Scatter ---
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.3, color='teal')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
        plt.title(f"Prediction Scatter Plot", fontsize=14)
        plt.xlabel("Actual SPEI")
        plt.ylabel("Predicted SPEI")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "prediction_scatter.png"))
        plt.close()
        print("Saved: prediction_scatter.png")
        
        # --- Plot 2: Error Distribution ---
        residuals = y_true - y_pred
        plt.figure(figsize=FIG_SIZE)
        sns.histplot(residuals, kde=True, color='purple')
        plt.title("Error Distribution (Residuals)", fontsize=14)
        plt.xlabel("Residual (Actual - Predicted)")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"))
        plt.close()
        print("Saved: error_distribution.png")
        
        # --- Plot 3: Time Series Sample ---
        # Plot a segment
        window_size = min(365, len(y_true))
        plt.figure(figsize=FIG_SIZE)
        plt.plot(y_true[:window_size], label='Actual', color='blue', alpha=0.7)
        plt.plot(y_pred[:window_size], label='Predicted', color='orange', linestyle='--', alpha=0.7)
        plt.title("Time Series Sample (Test Set)", fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "timeseries_sample.png"))
        plt.close()
        print("Saved: timeseries_sample.png")
        
        # --- Plot 4: Variable Importance ---
        print("Generating Interpretability...")
        raw_output = model.predict(test_dataloader, mode="raw", return_x=True)
        interpretation = model.interpret_output(raw_output.output, reduction="sum")
        
        # Plot variable importance
        model.plot_interpretation(interpretation)
        plt.savefig(os.path.join(RESULTS_DIR, "variable_importance.png"))
        plt.close()
        print("Saved: variable_importance.png")
        
        # --- Plot 5: Cross Correlation (Lag) ---
        if len(y_true) > 10:
            lags = np.arange(0, 11)
            correlations = []
            for lag in lags:
                if len(y_true) <= lag + 5:
                    correlations.append(np.nan)
                    continue
                if lag == 0:
                    corr, _ = pearsonr(y_true, y_pred)
                else:
                    corr, _ = pearsonr(y_true[lag:], y_pred[:-lag])
                correlations.append(corr)
                
            plt.figure(figsize=FIG_SIZE)
            plt.plot(lags, correlations, marker='o', linewidth=2, color='purple')
            best_lag = np.argmax(np.nan_to_num(correlations, nan=-1.0))
            plt.axvline(x=best_lag, color='red', linestyle='--', label=f'Optimal Lag: {best_lag}')
            plt.title("Cross-Correlation Analysis", fontsize=14)
            plt.xlabel("Lag (Days)")
            plt.ylabel("Pearson Correlation")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(RESULTS_DIR, "lag_analysis.png"))
            plt.close()
            print("Saved: lag_analysis.png")

        # --- Plot 6: Shock Detection ---
        if len(y_true) > 60:
            diffs = np.diff(y_true)
            shock_idx = np.argmax(np.abs(diffs))
            start = max(0, shock_idx - 30)
            end = min(len(y_true), shock_idx + 30)
            
            plt.figure(figsize=FIG_SIZE)
            plt.plot(range(start, end), y_true[start:end], label='Actual', color='blue', linewidth=2)
            plt.plot(range(start, end), y_pred[start:end], label='Predicted', color='orange', linestyle='--', linewidth=2)
            plt.axvline(x=shock_idx, color='red', linestyle=':', label='Shock Event')
            plt.title("Shock Detection Response", fontsize=14)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(RESULTS_DIR, "shock_detection.png"))
            plt.close()
            print("Saved: shock_detection.png")

    except Exception as e:
        print(f"Error during visualization generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_visualizations()
