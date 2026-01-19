"""
Detailed Actual vs Predict Analysis Script
===========================================
Script untuk menganalisis performa model TFT secara detail dengan
output teks yang lengkap untuk review akademik.

Output:
- Console log dengan analisis lengkap
- File TXT dengan metrik detail per horizon
- File CSV dengan data actual vs predict (per sample)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.models.dataset import create_dataset, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH

# Configuration
OUTPUT_DIR = "results/actual_vs_predict_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"detailed_analysis_{TIMESTAMP}.txt")

def log(message:""):
    """Print and save to file"""
    print(message)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")

def calculate_quantile_coverage(actuals, predictions, quantile_idx_low=1, quantile_idx_high=5):
    """
    Calculate quantile coverage (Prediction Interval Coverage Probability - PICP)
    predictions: shape (batch, horizon, quantiles)
    Default: P10 (idx=1) and P90 (idx=5) for 80% interval
    """
    p_low = predictions[:, :, quantile_idx_low]
    p_high = predictions[:, :, quantile_idx_high]
    
    within_interval = ((actuals >= p_low) & (actuals <= p_high)).float()
    coverage = within_interval.mean().item()
    return coverage

def calculate_interval_sharpness(predictions, quantile_idx_low=1, quantile_idx_high=5):
    """
    Calculate interval sharpness (average interval width)
    Narrower intervals are sharper (better if coverage is maintained)
    """
    p_low = predictions[:, :, quantile_idx_low]
    p_high = predictions[:, :, quantile_idx_high]
    
    interval_width = (p_high - p_low).mean().item()
    return interval_width

def calculate_calibration_error(actuals, predictions, quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]):
    """
    Calculate calibration error for each quantile
    A well-calibrated model should have X% of observations below the X-th quantile prediction
    """
    calibration = {}
    for i, q in enumerate(quantiles):
        q_pred = predictions[:, :, i]
        below = (actuals < q_pred).float().mean().item()
        calibration[f"Q{int(q*100):02d}"] = {
            "expected": q,
            "observed": below,
            "error": abs(below - q)
        }
    return calibration

def calculate_metrics_per_horizon(actuals, predictions, horizons):
    """
    Calculate metrics for each forecast horizon separately
    """
    horizon_metrics = []
    
    # Use P50 (median) for point predictions
    p50 = predictions[:, :, 3]  # Default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    
    for h in range(horizons):
        actual_h = actuals[:, h].cpu().numpy()
        pred_h = p50[:, h].cpu().numpy()
        
        rmse = np.sqrt(np.mean((pred_h - actual_h) ** 2))
        mae = np.mean(np.abs(pred_h - actual_h))
        bias = np.mean(pred_h - actual_h)
        corr = np.corrcoef(pred_h, actual_h)[0, 1] if len(actual_h) > 1 else 0
        
        horizon_metrics.append({
            "horizon": h + 1,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "correlation": corr
        })
    
    return pd.DataFrame(horizon_metrics)

def analyze_drought_classification(actuals, predictions):
    """
    Analyze drought event classification accuracy
    SPEI < -1.5 -> Kekeringan Parah
    SPEI > 1.5 -> Basah Ekstrem
    """
    p50 = predictions[:, :, 3].cpu().numpy().flatten()
    actual = actuals.cpu().numpy().flatten()
    
    # Drought classification
    def classify_spei(value):
        if value < -1.5:
            return "Kekeringan Parah"
        elif value < -1.0:
            return "Kekeringan Sedang"
        elif value < -0.5:
            return "Kekeringan Ringan"
        elif value < 0.5:
            return "Normal"
        elif value < 1.0:
            return "Basah Ringan"
        elif value < 1.5:
            return "Basah Sedang"
        else:
            return "Basah Ekstrem"
    
    actual_class = np.array([classify_spei(v) for v in actual])
    pred_class = np.array([classify_spei(v) for v in p50])
    
    # Confusion analysis
    correct = (actual_class == pred_class).sum()
    total = len(actual_class)
    accuracy = correct / total
    
    # Per-class analysis
    classes = ["Kekeringan Parah", "Kekeringan Sedang", "Kekeringan Ringan", 
               "Normal", "Basah Ringan", "Basah Sedang", "Basah Ekstrem"]
    
    class_stats = {}
    for cls in classes:
        actual_count = (actual_class == cls).sum()
        pred_count = (pred_class == cls).sum()
        correct_cls = ((actual_class == cls) & (pred_class == cls)).sum()
        
        precision = correct_cls / pred_count if pred_count > 0 else 0
        recall = correct_cls / actual_count if actual_count > 0 else 0
        
        class_stats[cls] = {
            "actual_count": int(actual_count),
            "pred_count": int(pred_count),
            "correct": int(correct_cls),
            "precision": precision,
            "recall": recall
        }
    
    return accuracy, class_stats

def main():
    log("=" * 70)
    log("ANALISIS DETAIL ACTUAL VS PREDICT - MODEL TFT SPEI")
    log("=" * 70)
    log(f"Waktu Analisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    
    # 1. Load Data
    log("=" * 70)
    log("STEP 1: LOADING DATA")
    log("=" * 70)
    
    data_path = "data/processed/spei_dataset.parquet"
    if not os.path.exists(data_path):
        log(f"ERROR: File tidak ditemukan: {data_path}")
        return
    
    data = pd.read_parquet(data_path)
    data['year'] = data['time'].dt.year
    
    log(f"Dataset Shape: {data.shape}")
    log(f"Period: {data['time'].min()} to {data['time'].max()}")
    log(f"Locations: {data['location_id'].unique().tolist()}")
    log(f"Target (SPEI_3) Stats:")
    log(f"  - Mean: {data['SPEI_3'].mean():.4f}")
    log(f"  - Std: {data['SPEI_3'].std():.4f}")
    log(f"  - Min: {data['SPEI_3'].min():.4f}")
    log(f"  - Max: {data['SPEI_3'].max():.4f}")
    log("")
    
    # 2. Create Test Dataset with proper splitting
    log("=" * 70)
    log("STEP 2: CREATING TEST DATASET")
    log("=" * 70)
    
    # Split: Train < 2023, Test >= 2023
    train_data = data[data.year < 2023].copy()
    test_data = data[data.year >= 2023].copy()
    
    log(f"Train Period: {train_data['time'].min()} to {train_data['time'].max()}")
    log(f"Test Period: {test_data['time'].min()} to {test_data['time'].max()}")
    log(f"Train Samples: {len(train_data)}")
    log(f"Test Samples: {len(test_data)}")
    
    train_ds = create_dataset(train_data)
    # Use all data for test to get proper test sequences
    test_ds = TimeSeriesDataSet.from_dataset(
        train_ds, 
        data,  # Use full data to create test sequences
        predict=True,  # Use only sequences starting in test period
        stop_randomization=True
    )
    test_dataloader = test_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    log(f"Test Dataset Size: {len(test_ds)} sequences")
    log(f"Encoder Length: {MAX_ENCODER_LENGTH} days")
    log(f"Prediction Length: {MAX_PREDICTION_LENGTH} days")
    log("")
    
    # 3. Load Model
    log("=" * 70)
    log("STEP 3: LOADING MODEL")
    log("=" * 70)
    
    checkpoint_dir = "logs/checkpoints"
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')])
    
    if not checkpoints:
        log("ERROR: No checkpoints found!")
        return
    
    # Get best model (lowest val_loss)
    best_ckpt = checkpoints[0]
    for ckpt in checkpoints:
        if "val_loss" in ckpt:
            best_ckpt = ckpt  # Last one is typically the best
    
    model_path = os.path.join(checkpoint_dir, best_ckpt)
    log(f"Loading: {model_path}")
    
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    model.eval()
    log("Model loaded successfully.")
    log("")
    
    # 4. Run Predictions
    log("=" * 70)
    log("STEP 4: RUNNING PREDICTIONS")
    log("=" * 70)
    
    # Get predictions with raw mode for quantiles
    predictions_obj = model.predict(test_dataloader, mode="raw", return_x=True, return_y=True)
    
    raw_predictions = predictions_obj.output
    x = predictions_obj.x
    y = predictions_obj.y
    
    # Extract prediction tensor
    if hasattr(raw_predictions, 'prediction'):
        preds = raw_predictions.prediction
    elif isinstance(raw_predictions, dict):
        preds = raw_predictions['prediction']
    else:
        preds = raw_predictions
    
    # Extract actuals
    if isinstance(y, tuple):
        actuals = y[0]
    else:
        actuals = y
    
    log(f"Predictions Shape: {preds.shape}")
    log(f"  - Samples: {preds.shape[0]}")
    log(f"  - Horizons: {preds.shape[1]}")
    log(f"  - Quantiles: {preds.shape[2]}")
    
    if actuals is None:
        log("WARNING: Actuals is None! Attempting alternative extraction...")
        # Try to get actuals from dataloader directly
        all_actuals = []
        for batch_x, batch_y in test_dataloader:
            if isinstance(batch_y, tuple):
                all_actuals.append(batch_y[0])
            else:
                all_actuals.append(batch_y)
        actuals = torch.cat(all_actuals, dim=0)
        log(f"Actuals Shape (from dataloader): {actuals.shape}")
    else:
        log(f"Actuals Shape: {actuals.shape}")
    
    log("")
    
    # 5. Overall Metrics
    log("=" * 70)
    log("STEP 5: OVERALL METRICS")
    log("=" * 70)
    
    p50 = preds[:, :, 3]  # Median prediction
    
    # Flatten for overall metrics
    actual_flat = actuals.flatten().cpu().numpy()
    pred_flat = p50.flatten().cpu().numpy()
    
    # Ensure same length
    min_len = min(len(actual_flat), len(pred_flat))
    actual_flat = actual_flat[:min_len]
    pred_flat = pred_flat[:min_len]
    
    rmse = np.sqrt(np.mean((pred_flat - actual_flat) ** 2))
    mae = np.mean(np.abs(pred_flat - actual_flat))
    bias = np.mean(pred_flat - actual_flat)
    corr = np.corrcoef(pred_flat, actual_flat)[0, 1]
    
    log("Point Metrics (using P50 - Median):")
    log(f"  RMSE: {rmse:.4f}")
    log(f"  MAE: {mae:.4f}")
    log(f"  Bias: {bias:.4f}")
    log(f"  Correlation: {corr:.4f}")
    log("")
    log("Range Comparison:")
    log(f"  Actual Range: [{actual_flat.min():.4f}, {actual_flat.max():.4f}]")
    log(f"  Predicted Range: [{pred_flat.min():.4f}, {pred_flat.max():.4f}]")
    log(f"  Variance Ratio (Pred/Actual): {np.var(pred_flat) / np.var(actual_flat):.4f}")
    log("")
    
    # 6. Quantile Metrics
    log("=" * 70)
    log("STEP 6: QUANTILE & PROBABILISTIC METRICS")
    log("=" * 70)
    
    # Prediction Interval Coverage
    coverage_80 = calculate_quantile_coverage(actuals, preds, 1, 5)  # P10-P90
    coverage_50 = calculate_quantile_coverage(actuals, preds, 2, 4)  # P25-P75
    
    sharpness_80 = calculate_interval_sharpness(preds, 1, 5)
    sharpness_50 = calculate_interval_sharpness(preds, 2, 4)
    
    log("Prediction Interval Coverage Probability (PICP):")
    log(f"  80% Interval (P10-P90): {coverage_80:.4f} (Expected: 0.80)")
    log(f"  50% Interval (P25-P75): {coverage_50:.4f} (Expected: 0.50)")
    log("")
    log("Interval Sharpness (Average Width):")
    log(f"  80% Interval: {sharpness_80:.4f}")
    log(f"  50% Interval: {sharpness_50:.4f}")
    log("")
    
    # Calibration
    calibration = calculate_calibration_error(actuals, preds)
    log("Quantile Calibration Analysis:")
    log("-" * 50)
    log(f"{'Quantile':<10} {'Expected':<10} {'Observed':<10} {'Error':<10}")
    log("-" * 50)
    for q, stats in calibration.items():
        log(f"{q:<10} {stats['expected']:<10.2f} {stats['observed']:<10.4f} {stats['error']:<10.4f}")
    
    avg_calibration_error = np.mean([v['error'] for v in calibration.values()])
    log("-" * 50)
    log(f"Average Calibration Error: {avg_calibration_error:.4f}")
    log("")
    
    # 7. Metrics per Horizon
    log("=" * 70)
    log("STEP 7: METRICS PER FORECAST HORIZON")
    log("=" * 70)
    
    horizon_df = calculate_metrics_per_horizon(actuals, preds, preds.shape[1])
    
    log("-" * 60)
    log(f"{'Horizon':<10} {'RMSE':<12} {'MAE':<12} {'Bias':<12} {'Corr':<12}")
    log("-" * 60)
    for _, row in horizon_df.iterrows():
        log(f"{row['horizon']:<10} {row['rmse']:<12.4f} {row['mae']:<12.4f} {row['bias']:<12.4f} {row['correlation']:<12.4f}")
    log("-" * 60)
    log("")
    
    # Save horizon metrics to CSV
    horizon_csv = os.path.join(OUTPUT_DIR, f"horizon_metrics_{TIMESTAMP}.csv")
    horizon_df.to_csv(horizon_csv, index=False)
    log(f"Horizon metrics saved to: {horizon_csv}")
    log("")
    
    # 8. Drought Classification Analysis
    log("=" * 70)
    log("STEP 8: DROUGHT CLASSIFICATION ANALYSIS")
    log("=" * 70)
    
    accuracy, class_stats = analyze_drought_classification(actuals, preds)
    log(f"Overall Classification Accuracy: {accuracy:.4f}")
    log("")
    log("-" * 80)
    log(f"{'Class':<20} {'Actual':<10} {'Predicted':<10} {'Correct':<10} {'Precision':<10} {'Recall':<10}")
    log("-" * 80)
    for cls, stats in class_stats.items():
        log(f"{cls:<20} {stats['actual_count']:<10} {stats['pred_count']:<10} {stats['correct']:<10} {stats['precision']:<10.4f} {stats['recall']:<10.4f}")
    log("-" * 80)
    log("")
    
    # 9. Sample-by-Sample Analysis
    log("=" * 70)
    log("STEP 9: SAMPLE-BY-SAMPLE ANALYSIS (First 10 Samples)")
    log("=" * 70)
    
    n_samples = min(10, preds.shape[0])
    sample_analysis = []
    
    for i in range(n_samples):
        sample_actual = actuals[i].cpu().numpy()
        sample_p10 = preds[i, :, 1].cpu().numpy()
        sample_p50 = preds[i, :, 3].cpu().numpy()
        sample_p90 = preds[i, :, 5].cpu().numpy()
        
        sample_rmse = np.sqrt(np.mean((sample_p50 - sample_actual) ** 2))
        sample_mae = np.mean(np.abs(sample_p50 - sample_actual))
        sample_coverage = np.mean((sample_actual >= sample_p10) & (sample_actual <= sample_p90))
        
        sample_analysis.append({
            "sample_id": i + 1,
            "rmse": sample_rmse,
            "mae": sample_mae,
            "coverage_80": sample_coverage,
            "actual_mean": sample_actual.mean(),
            "pred_mean": sample_p50.mean()
        })
        
        log(f"\nSample {i + 1}:")
        log(f"  RMSE: {sample_rmse:.4f}, MAE: {sample_mae:.4f}")
        log(f"  80% Coverage: {sample_coverage:.4f}")
        log(f"  Actual Mean: {sample_actual.mean():.4f}, Pred Mean: {sample_p50.mean():.4f}")
        log(f"  Horizons 1-5 Actual: {sample_actual[:5]}")
        log(f"  Horizons 1-5 P50:    {sample_p50[:5]}")
    
    # Save sample analysis
    sample_df = pd.DataFrame(sample_analysis)
    sample_csv = os.path.join(OUTPUT_DIR, f"sample_analysis_{TIMESTAMP}.csv")
    sample_df.to_csv(sample_csv, index=False)
    log(f"\nSample analysis saved to: {sample_csv}")
    log("")
    
    # 10. Full Actual vs Predict Data Export
    log("=" * 70)
    log("STEP 10: EXPORTING FULL ACTUAL VS PREDICT DATA")
    log("=" * 70)
    
    # Create detailed export
    export_data = []
    for i in range(preds.shape[0]):
        for h in range(preds.shape[1]):
            export_data.append({
                "sample_id": i,
                "horizon": h + 1,
                "actual": actuals[i, h].item(),
                "pred_p02": preds[i, h, 0].item(),
                "pred_p10": preds[i, h, 1].item(),
                "pred_p25": preds[i, h, 2].item(),
                "pred_p50": preds[i, h, 3].item(),
                "pred_p75": preds[i, h, 4].item(),
                "pred_p90": preds[i, h, 5].item(),
                "pred_p98": preds[i, h, 6].item(),
            })
    
    export_df = pd.DataFrame(export_data)
    export_csv = os.path.join(OUTPUT_DIR, f"full_actual_vs_predict_{TIMESTAMP}.csv")
    export_df.to_csv(export_csv, index=False)
    log(f"Full data exported to: {export_csv}")
    log(f"Total rows: {len(export_df)}")
    log("")
    
    # Summary
    log("=" * 70)
    log("ANALYSIS SUMMARY")
    log("=" * 70)
    log(f"1. Overall RMSE: {rmse:.4f}")
    log(f"2. Overall MAE: {mae:.4f}")
    log(f"3. Correlation: {corr:.4f}")
    log(f"4. Bias: {bias:.4f}")
    log(f"5. 80% PI Coverage: {coverage_80:.4f} (expected 0.80)")
    log(f"6. Variance Ratio: {np.var(pred_flat) / np.var(actual_flat):.4f}")
    log(f"7. Classification Accuracy: {accuracy:.4f}")
    log("")
    log("=" * 70)
    log("ANALISIS SELESAI")
    log(f"Output tersimpan di: {OUTPUT_DIR}")
    log("=" * 70)

if __name__ == "__main__":
    main()
