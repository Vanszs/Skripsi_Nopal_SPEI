"""
Evaluation script for TFT SPEI Forecasting Model
"""
import pandas as pd
import torch
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.models.dataset import create_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

def evaluate_model(checkpoint_path="logs/checkpoints/epoch=0-val_loss=0.35.ckpt"):
    print("="*60)
    print("TFT SPEI FORECASTING - EVALUATION")
    print("="*60)
    
    # Load data
    data = pd.read_parquet("data/processed/spei_dataset.parquet")
    data["year"] = data["time"].dt.year
    
    print(f"\nTotal Dataset Shape: {data.shape}")
    print(f"Locations: {sorted(data['location_id'].unique())}")
    print(f"Date Range: {data['time'].min()} to {data['time'].max()}")
    
    # Load best model
    print(f"\nLoading model: {checkpoint_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    model.to("cpu")  # Ensure model is on CPU
    
    # Create training dataset for reference
    training_cutoff = data[data.year < 2023]["time_idx"].max()
    train_ds = create_dataset(data[data.time_idx <= training_cutoff])
    
    # Test data (2024-2025)
    test_data = data[data.year >= 2024]
    print(f"\nTest Data Shape: {test_data.shape}")
    print(f"Test Period: {test_data['time'].min()} to {test_data['time'].max()}")
    
    # Create test dataset
    test_ds = TimeSeriesDataSet.from_dataset(train_ds, data, predict=True, stop_randomization=True)
    test_dataloader = test_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    print("\nGenerating predictions...")
    
    # Collect predictions and actuals
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            # Get predictions
            pred = model(batch_x)
            
            # For quantile output, take the median (middle quantile)
            if isinstance(pred, dict):
                pred_values = pred["prediction"]
            elif hasattr(pred, "prediction"):
                pred_values = pred.prediction
            else:
                pred_values = pred
            
            # If 3D (batch, horizon, quantiles), take middle quantile
            if len(pred_values.shape) == 3:
                mid_idx = pred_values.shape[2] // 2
                pred_values = pred_values[:, :, mid_idx]
            
            all_preds.append(pred_values.cpu())
            all_actuals.append(batch_y[0].cpu())  # batch_y is tuple (target, ...)
    
    # Concatenate all batches
    preds_tensor = torch.cat(all_preds, dim=0)
    actuals_tensor = torch.cat(all_actuals, dim=0)
    
    print(f"Predictions shape: {preds_tensor.shape}")
    print(f"Actuals shape: {actuals_tensor.shape}")
    
    # Flatten for metrics
    pred_flat = preds_tensor.flatten().numpy()
    actual_flat = actuals_tensor.flatten().numpy()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual_flat, pred_flat))
    mae = mean_absolute_error(actual_flat, pred_flat)
    correlation = np.corrcoef(actual_flat, pred_flat)[0, 1]
    
    print("\n" + "="*60)
    print("TEST SET METRICS (2024-2025)")
    print("="*60)
    print(f"RMSE:        {rmse:.4f}")
    print(f"MAE:         {mae:.4f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Samples:     {len(pred_flat)}")
    
    # Get interpretation using raw mode
    print("\nExtracting variable importance...")
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
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
        "encoder_importance": encoder_importance,
        "decoder_importance": decoder_importance
    }

if __name__ == "__main__":
    evaluate_model()
