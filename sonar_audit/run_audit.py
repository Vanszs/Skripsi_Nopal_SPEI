"""
TFT SPEI Model - Comprehensive Re-Audit Script
All-in-One Sonar Audit for Post-Optimization Model v3
Author: Auto-generated for Skripsi Audit
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy import stats
from scipy.signal import correlate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.models.dataset import create_dataset, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH

# ============================================================================
# CONFIGURATION
# ============================================================================
CHECKPOINT_PATH = PROJECT_ROOT / "logs/checkpoints/epoch=0-val_loss=0.38.ckpt"
DATA_PATH = PROJECT_ROOT / "data/processed/spei_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "sonar_audit/reports"
PLOTS_DIR = PROJECT_ROOT / "sonar_audit/plots"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class AuditResult:
    """Container for audit section results"""
    def __init__(self, section_name):
        self.section_name = section_name
        self.status = "NOT_RUN"  # NOT_RUN, PASS, WARN, FAIL
        self.metrics = {}
        self.notes = []
        self.timestamp = None
    
    def complete(self, status, metrics=None, notes=None):
        self.status = status
        self.metrics = metrics or {}
        self.notes = notes or []
        self.timestamp = datetime.now().isoformat()


class SonarAudit:
    """Comprehensive Model Audit Framework"""
    
    def __init__(self):
        self.results = {}
        self.actual_test = None
        self.predicted_test = None
        self.model = None
        self.data = None
        
    def load_model_and_data(self):
        """Load model checkpoint and test data"""
        print("="*60)
        print("LOADING MODEL AND DATA")
        print("="*60)
        
        # Load data
        self.data = pd.read_parquet(DATA_PATH)
        self.data["year"] = self.data["time"].dt.year
        print(f"Total Dataset Shape: {self.data.shape}")
        print(f"Date Range: {self.data['time'].min()} to {self.data['time'].max()}")
        
        # Load model
        print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
        self.model = TemporalFusionTransformer.load_from_checkpoint(
            str(CHECKPOINT_PATH), map_location="cpu"
        )
        self.model.eval()
        
        # Create datasets
        training_cutoff = self.data[self.data.year < 2023]["time_idx"].max()
        train_ds = create_dataset(self.data[self.data.time_idx <= training_cutoff])
        
        test_ds = TimeSeriesDataSet.from_dataset(
            train_ds, self.data, predict=True, stop_randomization=True
        )
        test_dataloader = test_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        
        # Generate predictions
        print("\nGenerating predictions...")
        all_preds, all_actuals = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in test_dataloader:
                pred = self.model(batch_x)
                if isinstance(pred, dict):
                    pred_values = pred["prediction"]
                elif hasattr(pred, "prediction"):
                    pred_values = pred.prediction
                else:
                    pred_values = pred
                
                if len(pred_values.shape) == 3:
                    mid_idx = pred_values.shape[2] // 2
                    pred_values = pred_values[:, :, mid_idx]
                
                all_preds.append(pred_values.cpu())
                all_actuals.append(batch_y[0].cpu())
        
        preds_tensor = torch.cat(all_preds, dim=0)
        actuals_tensor = torch.cat(all_actuals, dim=0)
        
        self.predicted_test = preds_tensor.flatten().numpy()
        self.actual_test = actuals_tensor.flatten().numpy()
        
        print(f"Predictions shape: {self.predicted_test.shape}")
        print(f"Actuals shape: {self.actual_test.shape}")
        
        return True

    # ========================================================================
    # SECTION 1: MODEL CONFIGURATION VALIDATION
    # ========================================================================
    def audit_section_1_config(self):
        """Audit model configuration and hyperparameters"""
        result = AuditResult("Section 1: Model Configuration")
        
        try:
            metrics = {
                "max_encoder_length": MAX_ENCODER_LENGTH,
                "max_prediction_length": MAX_PREDICTION_LENGTH,
                "learning_rate": self.model.hparams.learning_rate,
                "dropout": self.model.hparams.dropout,
                "hidden_size": self.model.hparams.hidden_size,
                "attention_head_size": self.model.hparams.attention_head_size,
                "loss_function": str(type(self.model.loss).__name__),
            }
            
            # Check checkpoint name for best epoch
            ckpt_name = CHECKPOINT_PATH.name
            if "epoch=" in ckpt_name:
                epoch_str = ckpt_name.split("epoch=")[1].split("-")[0]
                metrics["best_epoch"] = int(epoch_str)
            
            if "val_loss=" in ckpt_name:
                val_loss_str = ckpt_name.split("val_loss=")[1].replace(".ckpt", "")
                metrics["best_val_loss"] = float(val_loss_str)
            
            notes = []
            status = "PASS"
            
            if metrics.get("best_epoch", 99) == 0:
                notes.append("CRITICAL: Best epoch = 0 indicates model may not have learned!")
                status = "FAIL"
            
            if metrics["loss_function"] != "QuantileLoss":
                notes.append("WARNING: Loss function is not QuantileLoss")
                status = "WARN" if status == "PASS" else status
            
            result.complete(status, metrics, notes)
            
        except Exception as e:
            result.complete("FAIL", notes=[f"Error: {str(e)}"])
        
        self.results["section_1"] = result
        return result

    # ========================================================================
    # SECTION 2: DATA SHAPE & AVAILABILITY
    # ========================================================================
    def audit_section_2_data(self):
        """Audit data shapes and availability"""
        result = AuditResult("Section 2: Data Shape & Availability")
        
        try:
            metrics = {
                "actual_shape": str(self.actual_test.shape),
                "predicted_shape": str(self.predicted_test.shape),
                "actual_nan_count": int(np.isnan(self.actual_test).sum()),
                "predicted_nan_count": int(np.isnan(self.predicted_test).sum()),
                "actual_mean": float(np.nanmean(self.actual_test)),
                "actual_std": float(np.nanstd(self.actual_test)),
                "predicted_mean": float(np.nanmean(self.predicted_test)),
                "predicted_std": float(np.nanstd(self.predicted_test)),
                "actual_min": float(np.nanmin(self.actual_test)),
                "actual_max": float(np.nanmax(self.actual_test)),
                "predicted_min": float(np.nanmin(self.predicted_test)),
                "predicted_max": float(np.nanmax(self.predicted_test)),
            }
            
            notes = []
            status = "PASS"
            
            if metrics["actual_nan_count"] > 0 or metrics["predicted_nan_count"] > 0:
                notes.append(f"WARNING: NaN values detected!")
                status = "WARN"
            
            result.complete(status, metrics, notes)
            
        except Exception as e:
            result.complete("FAIL", notes=[f"Error: {str(e)}"])
        
        self.results["section_2"] = result
        return result

    # ========================================================================
    # SECTION 3: BIAS & RMSE DEEP DIVE
    # ========================================================================
    def audit_section_3_bias_rmse(self):
        """Audit bias and RMSE metrics"""
        result = AuditResult("Section 3: Bias & RMSE Deep Dive")
        
        try:
            errors = self.predicted_test - self.actual_test
            
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(self.actual_test, self.predicted_test))),
                "mae": float(mean_absolute_error(self.actual_test, self.predicted_test)),
                "r2": float(r2_score(self.actual_test, self.predicted_test)),
                "pearson_r": float(np.corrcoef(self.actual_test, self.predicted_test)[0, 1]),
                "bias_mean_error": float(np.mean(errors)),
                "error_std": float(np.std(errors)),
                "error_min": float(np.min(errors)),
                "error_max": float(np.max(errors)),
                "error_p25": float(np.percentile(np.abs(errors), 25)),
                "error_p50": float(np.percentile(np.abs(errors), 50)),
                "error_p75": float(np.percentile(np.abs(errors), 75)),
                "error_p95": float(np.percentile(np.abs(errors), 95)),
            }
            
            notes = []
            status = "PASS"
            
            if abs(metrics["bias_mean_error"]) > 0.1:
                notes.append(f"WARNING: High bias detected ({metrics['bias_mean_error']:.4f})")
                status = "WARN"
            
            if metrics["rmse"] > 0.3:
                notes.append(f"WARNING: RMSE above threshold ({metrics['rmse']:.4f})")
                status = "WARN"
            
            if metrics["pearson_r"] < 0.85:
                notes.append(f"WARNING: Low correlation ({metrics['pearson_r']:.4f})")
                status = "WARN" if status == "PASS" else status
            
            result.complete(status, metrics, notes)
            
        except Exception as e:
            result.complete("FAIL", notes=[f"Error: {str(e)}"])
        
        self.results["section_3"] = result
        return result

    # ========================================================================
    # SECTION 4: LAG DETECTION
    # ========================================================================
    def audit_section_4_lag(self):
        """Audit prediction lag"""
        result = AuditResult("Section 4: Lag Detection")
        
        try:
            lags = range(-10, 11)
            correlations = []
            
            for lag in lags:
                if lag >= 0:
                    if lag == 0:
                        corr = np.corrcoef(self.actual_test, self.predicted_test)[0, 1]
                    else:
                        corr = np.corrcoef(
                            self.actual_test[lag:], 
                            self.predicted_test[:-lag]
                        )[0, 1]
                else:
                    corr = np.corrcoef(
                        self.actual_test[:lag], 
                        self.predicted_test[-lag:]
                    )[0, 1]
                correlations.append(corr)
            
            optimal_lag_idx = np.argmax(correlations)
            optimal_lag = list(lags)[optimal_lag_idx]
            
            metrics = {
                "optimal_lag_days": int(optimal_lag),
                "correlation_at_lag_0": float(correlations[10]),  # lag=0 is at index 10
                "correlation_at_optimal_lag": float(max(correlations)),
                "lag_improvement": float(max(correlations) - correlations[10]),
            }
            
            notes = []
            status = "PASS"
            
            if abs(optimal_lag) > 2:
                notes.append(f"WARNING: Significant lag detected ({optimal_lag} days)")
                status = "WARN"
            
            if optimal_lag < 0:
                notes.append("CRITICAL: Negative lag detected - possible future leakage!")
                status = "FAIL"
            
            result.complete(status, metrics, notes)
            
        except Exception as e:
            result.complete("FAIL", notes=[f"Error: {str(e)}"])
        
        self.results["section_4"] = result
        return result

    # ========================================================================
    # SECTION 5: VARIANCE RATIO
    # ========================================================================
    def audit_section_5_variance(self):
        """Audit variance ratio"""
        result = AuditResult("Section 5: Variance Ratio")
        
        try:
            std_actual = np.std(self.actual_test)
            std_predicted = np.std(self.predicted_test)
            variance_ratio = std_predicted / std_actual
            
            # Percentile comparison
            percentiles = [1, 5, 25, 50, 75, 95, 99]
            percentile_diffs = {}
            for p in percentiles:
                actual_p = np.percentile(self.actual_test, p)
                pred_p = np.percentile(self.predicted_test, p)
                percentile_diffs[f"p{p}_diff"] = float(pred_p - actual_p)
            
            metrics = {
                "actual_std": float(std_actual),
                "predicted_std": float(std_predicted),
                "variance_ratio": float(variance_ratio),
                **percentile_diffs
            }
            
            notes = []
            if variance_ratio < 0.7:
                status = "FAIL"
                notes.append("UNDER-RESPONSIVE (Too smooth)")
            elif 0.7 <= variance_ratio < 0.85:
                status = "WARN"
                notes.append("SLIGHTLY DAMPENED (Acceptable)")
            elif 0.85 <= variance_ratio <= 1.15:
                status = "PASS"
                notes.append("OPTIMAL (Balanced)")
            elif 1.15 < variance_ratio <= 1.30:
                status = "WARN"
                notes.append("OVER-SENSITIVE (Monitor false alarms)")
            else:
                status = "FAIL"
                notes.append("HYPER-SENSITIVE (High false alarm risk)")
            
            result.complete(status, metrics, notes)
            
        except Exception as e:
            result.complete("FAIL", notes=[f"Error: {str(e)}"])
        
        self.results["section_5"] = result
        return result

    # ========================================================================
    # SECTION 6: SHOCK DETECTION & FALSE ALARMS
    # ========================================================================
    def audit_section_6_shocks(self):
        """Audit shock detection and false alarms"""
        result = AuditResult("Section 6: Shock Detection & False Alarms")
        
        try:
            # Shock detection (day-to-day change > 0.5)
            actual_changes = np.abs(np.diff(self.actual_test))
            pred_changes = np.abs(np.diff(self.predicted_test))
            
            actual_shocks = actual_changes > 0.5
            pred_shocks = pred_changes > 0.5
            
            n_actual_shocks = np.sum(actual_shocks)
            n_pred_shocks = np.sum(pred_shocks)
            
            # Match shocks within ±2 days
            matched_shocks = 0
            shock_indices = np.where(actual_shocks)[0]
            for i in shock_indices:
                start = max(0, i - 2)
                end = min(len(pred_shocks), i + 3)
                if np.any(pred_shocks[start:end]):
                    matched_shocks += 1
            
            detection_rate = matched_shocks / n_actual_shocks if n_actual_shocks > 0 else 0
            false_alarm_rate = (n_pred_shocks - matched_shocks) / n_pred_shocks if n_pred_shocks > 0 else 0
            
            # Drought detection (SPEI < -1.5)
            actual_droughts = self.actual_test < -1.5
            pred_droughts = self.predicted_test < -1.5
            
            n_actual_droughts = np.sum(actual_droughts)
            n_pred_droughts = np.sum(pred_droughts)
            
            true_positives = np.sum(actual_droughts & pred_droughts)
            false_positives = np.sum(pred_droughts & ~actual_droughts)
            false_negatives = np.sum(actual_droughts & ~pred_droughts)
            
            sensitivity = true_positives / n_actual_droughts if n_actual_droughts > 0 else 0
            fa_rate_drought = false_positives / n_pred_droughts if n_pred_droughts > 0 else 0
            
            metrics = {
                "total_actual_shocks": int(n_actual_shocks),
                "total_predicted_shocks": int(n_pred_shocks),
                "matched_shocks": int(matched_shocks),
                "shock_detection_rate": float(detection_rate),
                "shock_false_alarm_rate": float(false_alarm_rate),
                "actual_drought_days": int(n_actual_droughts),
                "predicted_drought_days": int(n_pred_droughts),
                "drought_true_positives": int(true_positives),
                "drought_false_positives": int(false_positives),
                "drought_false_negatives": int(false_negatives),
                "drought_sensitivity": float(sensitivity),
                "drought_false_alarm_rate": float(fa_rate_drought),
            }
            
            notes = []
            status = "PASS"
            
            if detection_rate < 0.8:
                notes.append(f"WARNING: Low shock detection rate ({detection_rate:.1%})")
                status = "WARN"
            
            if false_alarm_rate > 0.25:
                notes.append(f"WARNING: High false alarm rate ({false_alarm_rate:.1%})")
                status = "WARN" if status == "PASS" else status
            
            result.complete(status, metrics, notes)
            
        except Exception as e:
            result.complete("FAIL", notes=[f"Error: {str(e)}"])
        
        self.results["section_6"] = result
        return result

    # ========================================================================
    # SECTION 7: SMOOTHNESS VS REACTIVITY
    # ========================================================================
    def audit_section_7_smoothness(self):
        """Audit smoothness vs reactivity trade-off"""
        result = AuditResult("Section 7: Smoothness vs Reactivity")
        
        try:
            window = 7
            actual_volatility = pd.Series(self.actual_test).rolling(window).std()
            pred_volatility = pd.Series(self.predicted_test).rolling(window).std()
            
            avg_actual_vol = actual_volatility.mean()
            avg_pred_vol = pred_volatility.mean()
            volatility_ratio = avg_pred_vol / avg_actual_vol
            
            # Error autocorrelation
            errors = self.predicted_test - self.actual_test
            acf_values = []
            for lag in range(1, 11):
                acf_val = np.corrcoef(errors[:-lag], errors[lag:])[0, 1]
                acf_values.append(acf_val)
            
            metrics = {
                "avg_actual_volatility_7d": float(avg_actual_vol),
                "avg_predicted_volatility_7d": float(avg_pred_vol),
                "volatility_ratio": float(volatility_ratio),
                "error_acf_lag1": float(acf_values[0]),
                "error_acf_lag2": float(acf_values[1]),
                "error_acf_lag5": float(acf_values[4]),
            }
            
            notes = []
            if volatility_ratio < 0.6:
                status = "FAIL"
                notes.append("TOO SMOOTH (Under-reactive)")
            elif 0.6 <= volatility_ratio < 0.8:
                status = "WARN"
                notes.append("SLIGHTLY SMOOTH (Acceptable for stability)")
            elif 0.8 <= volatility_ratio <= 1.2:
                status = "PASS"
                notes.append("OPTIMAL (Reactive but stable)")
            else:
                status = "WARN"
                notes.append("TOO REACTIVE (Noisy predictions)")
            
            if abs(acf_values[0]) > 0.5:
                notes.append("High error autocorrelation - Model misses temporal patterns")
                status = "WARN" if status == "PASS" else status
            
            result.complete(status, metrics, notes)
            
        except Exception as e:
            result.complete("FAIL", notes=[f"Error: {str(e)}"])
        
        self.results["section_7"] = result
        return result

    # ========================================================================
    # SECTION 8: VISUAL DIAGNOSTICS
    # ========================================================================
    def audit_section_8_visuals(self):
        """Generate visual diagnostic plots"""
        result = AuditResult("Section 8: Visual Diagnostics")
        
        try:
            # Plot 1: Time Series (First 150 steps)
            fig, ax = plt.subplots(figsize=(14, 6))
            n_steps = min(150, len(self.actual_test))
            ax.plot(self.actual_test[:n_steps], label='Actual', linewidth=2)
            ax.plot(self.predicted_test[:n_steps], label='Predicted', linewidth=2, alpha=0.8)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(-1.5, color='red', linestyle='--', alpha=0.3, label='Drought Threshold')
            ax.legend()
            ax.set_title('Time Series: First 150 Steps')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('SPEI-3')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'audit_timeseries_150.png', dpi=150)
            plt.close()
            
            # Plot 2: Residual Analysis (4 subplots)
            errors = self.predicted_test - self.actual_test
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].hist(errors, bins=50, edgecolor='black')
            axes[0, 0].axvline(0, color='red', linestyle='--')
            axes[0, 0].set_title(f'Error Distribution (Mean={errors.mean():.3f})')
            axes[0, 0].set_xlabel('Error')
            
            axes[0, 1].scatter(self.predicted_test, errors, alpha=0.3, s=5)
            axes[0, 1].axhline(0, color='red', linestyle='--')
            axes[0, 1].set_title('Residuals vs Predicted')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Error')
            
            axes[1, 0].plot(errors)
            axes[1, 0].axhline(0, color='red', linestyle='--')
            axes[1, 0].set_title('Residuals Over Time')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Error')
            
            stats.probplot(errors, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot (Normality Check)')
            
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'audit_residuals.png', dpi=150)
            plt.close()
            
            metrics = {
                "plots_generated": 2,
                "timeseries_plot": str(PLOTS_DIR / 'audit_timeseries_150.png'),
                "residuals_plot": str(PLOTS_DIR / 'audit_residuals.png'),
            }
            
            result.complete("PASS", metrics, ["Plots generated successfully"])
            
        except Exception as e:
            result.complete("FAIL", notes=[f"Error: {str(e)}"])
        
        self.results["section_8"] = result
        return result

    # ========================================================================
    # SECTION 9: PRODUCTION READINESS SCORECARD
    # ========================================================================
    def audit_section_9_scorecard(self):
        """Calculate production readiness scorecard"""
        result = AuditResult("Section 9: Production Readiness Scorecard")
        
        try:
            scores = {}
            
            # Training Convergence (10 pts)
            s1 = self.results.get("section_1")
            if s1 and s1.metrics.get("best_epoch", 0) > 5:
                scores["training_convergence"] = 10
            elif s1 and s1.metrics.get("best_epoch", 0) > 0:
                scores["training_convergence"] = 5
            else:
                scores["training_convergence"] = 0
            
            # Bias Control (10 pts)
            s3 = self.results.get("section_3")
            if s3:
                bias = abs(s3.metrics.get("bias_mean_error", 1))
                if bias < 0.05:
                    scores["bias_control"] = 10
                elif bias < 0.1:
                    scores["bias_control"] = 8
                elif bias < 0.2:
                    scores["bias_control"] = 5
                else:
                    scores["bias_control"] = 2
            
            # RMSE (10 pts)
            if s3:
                rmse = s3.metrics.get("rmse", 1)
                if rmse < 0.2:
                    scores["rmse"] = 10
                elif rmse < 0.25:
                    scores["rmse"] = 8
                elif rmse < 0.3:
                    scores["rmse"] = 6
                else:
                    scores["rmse"] = 3
            
            # Correlation (10 pts)
            if s3:
                r = s3.metrics.get("pearson_r", 0)
                if r > 0.95:
                    scores["correlation"] = 10
                elif r > 0.90:
                    scores["correlation"] = 8
                elif r > 0.85:
                    scores["correlation"] = 6
                else:
                    scores["correlation"] = 3
            
            # Lag (10 pts)
            s4 = self.results.get("section_4")
            if s4:
                lag = abs(s4.metrics.get("optimal_lag_days", 99))
                if lag <= 1:
                    scores["lag"] = 10
                elif lag <= 2:
                    scores["lag"] = 8
                elif lag <= 4:
                    scores["lag"] = 5
                else:
                    scores["lag"] = 2
            
            # Variance Ratio (10 pts)
            s5 = self.results.get("section_5")
            if s5:
                vr = s5.metrics.get("variance_ratio", 0)
                if 0.85 <= vr <= 1.15:
                    scores["variance_ratio"] = 10
                elif 0.7 <= vr <= 1.3:
                    scores["variance_ratio"] = 6
                else:
                    scores["variance_ratio"] = 2
            
            # Shock Detection (10 pts)
            s6 = self.results.get("section_6")
            if s6:
                det_rate = s6.metrics.get("shock_detection_rate", 0)
                if det_rate > 0.9:
                    scores["shock_detection"] = 10
                elif det_rate > 0.8:
                    scores["shock_detection"] = 8
                elif det_rate > 0.6:
                    scores["shock_detection"] = 5
                else:
                    scores["shock_detection"] = 2
            
            # False Alarm Rate (10 pts)
            if s6:
                fa_rate = s6.metrics.get("shock_false_alarm_rate", 1)
                if fa_rate < 0.15:
                    scores["false_alarm"] = 10
                elif fa_rate < 0.25:
                    scores["false_alarm"] = 7
                elif fa_rate < 0.4:
                    scores["false_alarm"] = 4
                else:
                    scores["false_alarm"] = 1
            
            # Residual Quality (10 pts)
            s7 = self.results.get("section_7")
            if s7:
                acf1 = abs(s7.metrics.get("error_acf_lag1", 1))
                if acf1 < 0.2:
                    scores["residual_quality"] = 10
                elif acf1 < 0.3:
                    scores["residual_quality"] = 7
                elif acf1 < 0.5:
                    scores["residual_quality"] = 4
                else:
                    scores["residual_quality"] = 1
            
            # Stability (10 pts)
            s2 = self.results.get("section_2")
            if s2 and s2.metrics.get("actual_nan_count", 1) == 0:
                scores["stability"] = 10
            else:
                scores["stability"] = 5
            
            total_score = sum(scores.values())
            
            if total_score >= 90:
                classification = "PRODUCTION READY (Deploy immediately)"
            elif total_score >= 80:
                classification = "GOOD (Minor tuning recommended)"
            elif total_score >= 70:
                classification = "MARGINAL (Needs improvement)"
            elif total_score >= 60:
                classification = "CONCERNS (Significant issues)"
            else:
                classification = "NOT READY (Major redesign needed)"
            
            metrics = {
                **scores,
                "total_score": total_score,
                "classification": classification,
            }
            
            status = "PASS" if total_score >= 70 else "WARN" if total_score >= 60 else "FAIL"
            result.complete(status, metrics, [classification])
            
        except Exception as e:
            result.complete("FAIL", notes=[f"Error: {str(e)}"])
        
        self.results["section_9"] = result
        return result

    # ========================================================================
    # SECTION 10: REGRESSION CHECK
    # ========================================================================
    def audit_section_10_regression(self):
        """Check for regressions vs initial model"""
        result = AuditResult("Section 10: Regression Check")
        
        try:
            # Baseline values from v1 model (from evaluation_summary)
            v1_baseline = {
                "best_epoch": 0,
                "rmse": 0.29,
                "bias": -0.176,
                "optimal_lag": 4,
                "variance_ratio": 0.545,
                "correlation": 0.935,
            }
            
            # Current values
            s1 = self.results.get("section_1", AuditResult(""))
            s3 = self.results.get("section_3", AuditResult(""))
            s4 = self.results.get("section_4", AuditResult(""))
            s5 = self.results.get("section_5", AuditResult(""))
            
            current = {
                "best_epoch": s1.metrics.get("best_epoch", 0),
                "rmse": s3.metrics.get("rmse", 0),
                "bias": s3.metrics.get("bias_mean_error", 0),
                "optimal_lag": s4.metrics.get("optimal_lag_days", 0),
                "variance_ratio": s5.metrics.get("variance_ratio", 0),
                "correlation": s3.metrics.get("pearson_r", 0),
            }
            
            # Compare
            comparisons = {}
            regressions = []
            improvements = []
            
            for key in v1_baseline:
                v1_val = v1_baseline[key]
                curr_val = current[key]
                
                if key in ["rmse", "optimal_lag"]:  # Lower is better
                    if curr_val < v1_val:
                        improvements.append(key)
                        comparisons[f"{key}_status"] = "IMPROVED"
                    elif curr_val > v1_val * 1.1:  # 10% tolerance
                        regressions.append(key)
                        comparisons[f"{key}_status"] = "REGRESSED"
                    else:
                        comparisons[f"{key}_status"] = "STABLE"
                elif key == "bias":  # Closer to 0 is better
                    if abs(curr_val) < abs(v1_val):
                        improvements.append(key)
                        comparisons[f"{key}_status"] = "IMPROVED"
                    elif abs(curr_val) > abs(v1_val) * 1.1:
                        regressions.append(key)
                        comparisons[f"{key}_status"] = "REGRESSED"
                    else:
                        comparisons[f"{key}_status"] = "STABLE"
                else:  # Higher is better
                    if curr_val > v1_val:
                        improvements.append(key)
                        comparisons[f"{key}_status"] = "IMPROVED"
                    elif curr_val < v1_val * 0.9:
                        regressions.append(key)
                        comparisons[f"{key}_status"] = "REGRESSED"
                    else:
                        comparisons[f"{key}_status"] = "STABLE"
                
                comparisons[f"{key}_v1"] = v1_val
                comparisons[f"{key}_current"] = curr_val
            
            metrics = {
                **comparisons,
                "regressions_detected": len(regressions) > 0,
                "regressed_metrics": regressions,
                "improved_metrics": improvements,
            }
            
            status = "FAIL" if len(regressions) > 2 else "WARN" if regressions else "PASS"
            notes = []
            if regressions:
                notes.append(f"Regressions detected in: {', '.join(regressions)}")
            if improvements:
                notes.append(f"Improvements in: {', '.join(improvements)}")
            
            result.complete(status, metrics, notes)
            
        except Exception as e:
            result.complete("FAIL", notes=[f"Error: {str(e)}"])
        
        self.results["section_10"] = result
        return result

    # ========================================================================
    # SECTION 11: RECOMMENDATIONS
    # ========================================================================
    def audit_section_11_recommendations(self):
        """Generate final recommendations"""
        result = AuditResult("Section 11: Final Recommendations")
        
        try:
            priority_1 = []  # Critical
            priority_2 = []  # High
            priority_3 = []  # Nice-to-have
            
            # Check each section for issues
            s1 = self.results.get("section_1", AuditResult(""))
            if s1.metrics.get("best_epoch", 0) == 0:
                priority_1.append({
                    "issue": "Model stopped at epoch 0",
                    "root_cause": "Early stopping triggered too early or learning rate too high",
                    "fix": "Increase patience, reduce learning rate to 0.0001",
                    "effort": "2-4 hours",
                })
            
            s3 = self.results.get("section_3", AuditResult(""))
            if abs(s3.metrics.get("bias_mean_error", 0)) > 0.1:
                priority_2.append({
                    "issue": f"Bias detected: {s3.metrics.get('bias_mean_error', 0):.4f}",
                    "root_cause": "Systematic prediction offset",
                    "fix": "Apply bias correction in post-processing",
                    "effort": "30 minutes",
                })
            
            s5 = self.results.get("section_5", AuditResult(""))
            vr = s5.metrics.get("variance_ratio", 1)
            if vr < 0.7:
                priority_1.append({
                    "issue": f"Variance ratio too low: {vr:.3f}",
                    "root_cause": "Model is over-smoothing predictions",
                    "fix": "Reduce dropout, increase hidden_size, or apply variance scaling",
                    "effort": "2-4 hours",
                })
            
            s4 = self.results.get("section_4", AuditResult(""))
            lag = s4.metrics.get("optimal_lag_days", 0)
            if abs(lag) > 2:
                priority_2.append({
                    "issue": f"Prediction lag: {lag} days",
                    "root_cause": "Model relies too heavily on recent history",
                    "fix": "Reduce encoder length or add attention bias",
                    "effort": "1-2 hours",
                })
            
            s7 = self.results.get("section_7", AuditResult(""))
            acf1 = s7.metrics.get("error_acf_lag1", 0)
            if abs(acf1) > 0.3:
                priority_3.append({
                    "issue": f"High error autocorrelation: {acf1:.3f}",
                    "root_cause": "Model missing temporal patterns",
                    "fix": "Consider residual connection or longer encoder",
                    "effort": "2-3 hours",
                })
            
            # Determine go/no-go
            s9 = self.results.get("section_9", AuditResult(""))
            total_score = s9.metrics.get("total_score", 0)
            
            if total_score >= 80 and not priority_1:
                go_decision = "DEPLOY"
                go_reasoning = "Model meets production criteria with minor issues"
            elif total_score >= 70 and len(priority_1) <= 1:
                go_decision = "HOLD"
                go_reasoning = "Address priority 1 issues before deployment"
            else:
                go_decision = "RE-TRAIN"
                go_reasoning = "Significant issues require model retraining"
            
            metrics = {
                "priority_1_count": len(priority_1),
                "priority_2_count": len(priority_2),
                "priority_3_count": len(priority_3),
                "go_decision": go_decision,
                "go_reasoning": go_reasoning,
            }
            
            result.priority_1 = priority_1
            result.priority_2 = priority_2
            result.priority_3 = priority_3
            
            status = "PASS" if go_decision == "DEPLOY" else "WARN" if go_decision == "HOLD" else "FAIL"
            result.complete(status, metrics, [go_reasoning])
            
        except Exception as e:
            result.complete("FAIL", notes=[f"Error: {str(e)}"])
        
        self.results["section_11"] = result
        return result

    # ========================================================================
    # RUN ALL AUDITS
    # ========================================================================
    def run_full_audit(self):
        """Execute all audit sections"""
        print("\n" + "="*60)
        print("SONAR AUDIT - TFT SPEI MODEL v3")
        print("="*60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        self.load_model_and_data()
        
        print("\n" + "-"*60)
        print("Running Section 1: Model Configuration...")
        self.audit_section_1_config()
        
        print("Running Section 2: Data Shape & Availability...")
        self.audit_section_2_data()
        
        print("Running Section 3: Bias & RMSE Deep Dive...")
        self.audit_section_3_bias_rmse()
        
        print("Running Section 4: Lag Detection...")
        self.audit_section_4_lag()
        
        print("Running Section 5: Variance Ratio...")
        self.audit_section_5_variance()
        
        print("Running Section 6: Shock Detection & False Alarms...")
        self.audit_section_6_shocks()
        
        print("Running Section 7: Smoothness vs Reactivity...")
        self.audit_section_7_smoothness()
        
        print("Running Section 8: Visual Diagnostics...")
        self.audit_section_8_visuals()
        
        print("Running Section 9: Production Readiness Scorecard...")
        self.audit_section_9_scorecard()
        
        print("Running Section 10: Regression Check...")
        self.audit_section_10_regression()
        
        print("Running Section 11: Final Recommendations...")
        self.audit_section_11_recommendations()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*60)
        print("AUDIT COMPLETE")
        print("="*60)
        
        return self.results

    def generate_report(self):
        """Generate comprehensive audit report"""
        report_path = OUTPUT_DIR / "audit_report.txt"
        
        with open(report_path, "w") as f:
            f.write("="*70 + "\n")
            f.write("TFT SPEI MODEL - COMPREHENSIVE AUDIT REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Model: {CHECKPOINT_PATH.name}\n\n")
            
            for section_key, result in self.results.items():
                f.write("-"*70 + "\n")
                f.write(f"{result.section_name}\n")
                f.write(f"Status: {result.status}\n")
                f.write("-"*70 + "\n")
                
                if result.metrics:
                    f.write("Metrics:\n")
                    for k, v in result.metrics.items():
                        if isinstance(v, float):
                            f.write(f"  {k}: {v:.4f}\n")
                        else:
                            f.write(f"  {k}: {v}\n")
                
                if result.notes:
                    f.write("Notes:\n")
                    for note in result.notes:
                        f.write(f"  - {note}\n")
                
                f.write("\n")
            
            # Summary
            s9 = self.results.get("section_9", AuditResult(""))
            s11 = self.results.get("section_11", AuditResult(""))
            
            f.write("="*70 + "\n")
            f.write("EXECUTIVE SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Production Readiness Score: {s9.metrics.get('total_score', 'N/A')} / 100\n")
            f.write(f"Classification: {s9.metrics.get('classification', 'N/A')}\n")
            f.write(f"Go/No-Go Decision: {s11.metrics.get('go_decision', 'N/A')}\n")
            f.write(f"Reasoning: {s11.metrics.get('go_reasoning', 'N/A')}\n")
        
        print(f"\nReport saved to: {report_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    audit = SonarAudit()
    results = audit.run_full_audit()
