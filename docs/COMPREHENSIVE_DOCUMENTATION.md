# 📚 DOKUMENTASI LENGKAP PROJECT
## Peramalan Multi‑Horizon Indeks Kekeringan Lahan Pertanian (SPEI) Menggunakan Temporal Fusion Transformer (TFT)

**Versi:** 1.0  
**Tanggal:** 2026-01-19  
**Status:** ✅ Complete

---

# BAGIAN 1: KONTEKS PROJECT

## 1.1 Judul Lengkap Skripsi

> **"Peramalan Multi‑Horizon Indeks Kekeringan Lahan Pertanian (SPEI) di Sentra Padi Jawa Timur Menggunakan Temporal Fusion Transformer (TFT)"**

## 1.2 Breakdown Judul

| Komponen | Penjelasan |
|----------|------------|
| **Peramalan Multi‑Horizon** | Prediksi 30 hari ke depan secara berurutan |
| **Indeks Kekeringan (SPEI‑3)** | Indeks kekeringan berbasis iklim (3‑month scale) |
| **Sentra Padi Jatim** | Lokasi studi: Bojonegoro, Lamongan, Nganjuk, Ngawi, Tuban |
| **Temporal Fusion Transformer** | Model deep learning untuk time series multivariat dan interpretable |

## 1.3 Tujuan Project

### Tujuan Utama
Membangun sistem peramalan SPEI‑3 multi‑horizon untuk mendukung monitoring kekeringan lahan pertanian di sentra padi Jawa Timur.

### Tujuan Teknis
1. Prediksi kuantil probabilistik (P10, P50, P90) untuk ketidakpastian.
2. Evaluasi kuantitatif (RMSE, MAE, R², Pearson r, quantile loss, bias).
3. Interpretabilitas (variable importance dan attention weights).
4. Deteksi kekeringan (SPEI < -1.5) dan metrik klasifikasi.

---

# BAGIAN 2: DATA

## 2.1 Sumber Data

| Aspek | Detail |
|-------|--------|
| **Format** | Parquet |
| **Lokasi** | data/processed/spei_dataset.parquet |
| **Rentang Waktu** | 2005‑06‑29 s.d. 2025‑01‑01 |
| **Lokasi** | Bojonegoro, Lamongan, Nganjuk, Ngawi, Tuban |

## 2.2 Variabel Utama (dari Data)

| Variabel | Deskripsi |
|----------|-----------|
| SPEI_3 | Target utama (SPEI‑3) |
| precipitation_log | Curah hujan (transformasi log) |
| temperature_2m_max | Suhu maksimum |
| temperature_2m_min | Suhu minimum |
| soil_moisture | Kelembapan tanah |
| et0_fao_evapotranspiration | Evapotranspirasi potensial |
| month_sin, month_cos | Fitur musiman (cyclical encoding) |
| time_idx | Indeks waktu kontinyu |
| elevation | Elevasi lokasi |
| location_id | ID lokasi (categorical) |

**Variabel Tambahan (auto-generated oleh TFT):**
- `relative_time_idx`: Indeks waktu relatif (generated via `add_relative_time_idx=True`)
- `target_scales`, `encoder_length`: Features tambahan untuk TFT

---

# BAGIAN 3: METODE

## 3.1 Arsitektur Model (TFT)

```
Input multivariat → Variable Selection Network (VSN) → LSTM Encoder/Decoder
→ Multi‑head attention → Gated residual network → Quantile outputs (P10/P50/P90)
```

## 3.2 Konfigurasi Penting

| Parameter | Nilai |
|-----------|-------|
| Encoder length | 30 hari |
| Prediction length | 30 hari |
| Quantiles | P10, P50, P90 |
| Checkpoint | logs/checkpoints/epoch=8-val_loss=0.37.ckpt |

## 3.3 Skema Split

| Set | Periode | Tujuan |
|-----|---------|--------|
| **Train** | < 2023 | Pembelajaran model |
| **Test** | 2023–2025 | Evaluasi akhir (default) |

**Catatan:** Test period dapat dikonfigurasi via parameter `test_year_start` di `evaluate_model()`. Default: 2023.

---

# BAGIAN 4: PIPELINE LENGKAP

## 4.1 Step‑by‑Step Pipeline

### STEP 1: Load Data
- File: data/processed/spei_dataset.parquet
- Fungsi: `pd.read_parquet()`

### STEP 2: Build Dataset
- File: src/models/dataset.py
- Fungsi: `create_dataset()`
- Output: TimeSeriesDataSet object

### STEP 3: Load Model
- File: logs/checkpoints/epoch=8-val_loss=0.37.ckpt
- Method: `TemporalFusionTransformer.load_from_checkpoint()`

### STEP 4: Generate Predictions
- Quantile outputs (P10, P50, P90)
- Ensemble rolling window predictions
- Fungsi: `generate_predictions()` di evaluate.py

### STEP 5: Calibration
- **Raw TFT:** Prediksi langsung dari model
- **Global Calibration:** Bias + variance scaling untuk semua lokasi
- **Per‑location Calibration:** Kalibrasi terpisah per lokasi (paling akurat)

### STEP 6: Evaluation
- Metrik keseluruhan (overall)
- Metrik per‑location (5 lokasi)
- Metrik per‑horizon (1-30 hari)
- Drought detection (SPEI < -1.5)
- Fungsi: `calculate_metrics()` di evaluate.py

### STEP 7: Export
- results/predictions_eval.csv (prediksi utama)
- results/evaluation_metrics_detailed.json (metrik lengkap)
- results/evaluation_output_YYYYMMDD_HHMMSS.txt (log evaluasi)
- Plot visualisasi (6 plots utama)

---

# BAGIAN 5: OUTPUT UTAMA (WAJIB)

## 5.1 Predictions Output
- **File Utama:** results/predictions_eval.csv
- **File Alternatif:** results/predictions_2023_2025.csv, results/predictions_2024_2025.csv (tergantung test period)
- **Kolom:**
  - date: Tanggal prediksi
  - location: Nama lokasi
  - actual: SPEI-3 aktual
  - predicted_p10: Prediksi kuantil 10% (batas bawah)
  - predicted_p50: Prediksi median
  - predicted_p90: Prediksi kuantil 90% (batas atas)
  - lower_bound: Batas bawah interval prediksi
  - upper_bound: Batas atas interval prediksi
  - prediction_interval_width: Lebar interval prediksi
  - Alert: Status kekeringan (Drought/Normal)

## 5.2 Quantitative Metrics
- **File:** results/evaluation_metrics_detailed.json
- **Struktur:**
  - `overall_raw`: Metrics dari raw TFT predictions
  - `overall_global_calib`: Metrics setelah global calibration
  - `overall_loc_calib`: Metrics setelah per-location calibration (TERBAIK)
  - `per_location`: Breakdown metrics per 5 lokasi
  - `per_horizon`: Metrics degradasi 1-30 hari ke depan
  - `drought_detection`: Confusion matrix, F1, sensitivity, specificity

## 5.3 Evaluation Log
- **File:** results/evaluation_output_YYYYMMDD_HHMMSS.txt
- **Isi:** Log lengkap hasil evaluasi dengan timestamp
- **Generated by:** run_evaluation.py

## 5.4 Visualisasi
- **Utama:**
  - results/actual_vs_predicted_multisite.png (time series per lokasi)
  - results/prediction_scatter.png (scatter actual vs predicted)
  - results/variable_importance.png (feature importance dari VSN)
  - results/attention_weights.png (temporal attention weights)
  - results/error_distribution.png (distribusi error)
  - results/timeseries_sample.png (sample prediksi detail)
- **Tambahan:**
  - results/residual_analysis.png (analisis residual)
  - results/confusion_matrix.png (deteksi kekeringan)
  - results/interpretation_encoder_variables.png
  - results/interpretation_decoder_variables.png
  - results/interpretation_static_variables.png
  - results/interpretation_attention.png

---

# BAGIAN 6: HASIL EVALUASI TERBARU

Berdasarkan evaluasi terbaru (2023‑2025) dengan **per‑location calibration**:

| Metrik | Nilai |
|--------|-------|
| RMSE | 0.363 |
| MAE | 0.289 |
| R² | 0.828 |
| Pearson r | 0.914 |
| Bias | ~0.00 |
| Total Samples | 3585 |

**Catatan:** Nilai persis tersedia di `results/evaluation_metrics_detailed.json`.

## 6.1 Perbandingan Metode Kalibrasi

| Metode | RMSE | MAE | R² | Pearson r |
|--------|------|-----|-----|-----------|
| Raw TFT | 0.515 | 0.427 | 0.655 | 0.910 |
| Global Calibration | 0.372 | 0.294 | 0.820 | 0.910 |
| **Per-Location Calibration** | **0.363** | **0.289** | **0.828** | **0.914** |

**Insight:** Per-location calibration memberikan peningkatan signifikan dalam RMSE (-29.5%), MAE (-32.3%), dan R² (+26.4%) dibanding raw TFT.

---

# BAGIAN 7: INTERPRETABILITAS

## 7.1 Variable Importance (VSN)
- Menunjukkan variabel paling berkontribusi terhadap prediksi SPEI-3
- Dihasilkan dari Variable Selection Network (VSN) di TFT
- Output: results/variable_importance.png
- Top 3 variables (umumnya):
  1. SPEI_3 historis
  2. precipitation_log
  3. soil_moisture

## 7.2 Attention Weights
- Menunjukkan timesteps historis yang paling berpengaruh
- Visualisasi multi-head attention mechanism
- Output: results/attention_weights.png atau results/interpretation_attention.png
- Membantu memahami "lookback window" optimal

---

# BAGIAN 8: EVALUASI KEKERINGAN

## 8.1 Kriteria Deteksi
- **Threshold:** SPEI < -1.5 (kekeringan parah)
- **Klasifikasi:** Binary (Drought vs Normal)

## 8.2 Metrics
- Confusion matrix: results/confusion_matrix.png
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- Precision
- F1-score
- False Alarm Rate

## 8.3 Output
Tersimpan di `drought_detection` section dalam evaluation_metrics_detailed.json

---

# BAGIAN 9: STRUKTUR PROJECT

```
Skripsi_Nopal/
├── data/
│   ├── raw/
│   │   └── weather_history_east_java.parquet
│   └── processed/
│       └── spei_dataset.parquet
├── logs/
│   ├── checkpoints/
│   │   └── epoch=8-val_loss=0.37.ckpt (MODEL TERBAIK)
│   └── lightning_logs/
│       └── version_*/
├── notebooks/
│   ├── TFT_SPEI_Evaluation.ipynb (notebook evaluasi)
│   ├── inspect_ckpt.py
│   ├── verify_fix.py
│   └── visualize_fix.py
├── results/
│   ├── predictions_eval.csv (OUTPUT UTAMA)
│   ├── evaluation_metrics_detailed.json (OUTPUT UTAMA)
│   ├── evaluation_output_*.txt (LOG)
│   ├── predictions_2023_2025.csv
│   ├── predictions_2024_2025.csv
│   ├── actual_vs_predicted_multisite.png
│   ├── prediction_scatter.png
│   ├── variable_importance.png
│   ├── attention_weights.png
│   ├── error_distribution.png
│   ├── timeseries_sample.png
│   ├── residual_analysis.png
│   ├── confusion_matrix.png
│   └── interpretation_*.png
├── src/
│   ├── data/
│   │   ├── ingest.py (download data dari API)
│   │   ├── preprocess.py (preprocessing + SPEI calculation)
│   │   └── spei.py (fungsi kalkulasi SPEI)
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/
│   │   ├── dataset.py (TimeSeriesDataSet configuration)
│   │   └── tft.py (TFT model architecture)
│   ├── training/
│   │   └── train.py (training script)
│   └── visualization/
│       └── generate_visualizations.py
├── scripts/
│   └── detailed_actual_vs_predict.py
├── sonar_audit/
│   └── run_audit.py
├── evaluate.py (SCRIPT EVALUASI UTAMA)
├── run_evaluation.py (WRAPPER + LOG)
├── main.py
└── requirements.txt
```

---

# BAGIAN 10: CAPTION SIAP PAKAI

**Gambar 1 - Time Series Multi-Site:**
Perbandingan SPEI‑3 Aktual dan Prediksi TFT untuk Lima Sentra Padi Jawa Timur (Periode Uji: 2023‑2025). Garis hitam menunjukkan nilai aktual, garis biru putus‑putus menunjukkan prediksi median (P50), dan area berwarna menunjukkan interval prediksi 80% (P10‑P90). Garis merah putus‑putus menandai ambang kekeringan parah (SPEI < -1.5). Model menggunakan per‑location calibration untuk meningkatkan akurasi prediksi.

**Gambar 2 - Scatter Plot:**
Scatter plot perbandingan nilai SPEI-3 aktual vs prediksi dengan per-location calibration. Garis diagonal menunjukkan prediksi sempurna. R² = 0.828 menunjukkan tingkat kecocokan yang sangat baik antara prediksi dan nilai aktual.

**Gambar 3 - Variable Importance:**
Tingkat kepentingan variabel input terhadap prediksi SPEI-3 yang dihasilkan dari Variable Selection Network (VSN) dalam arsitektur TFT. Variabel SPEI-3 historis, precipitation_log, dan soil_moisture menunjukkan kontribusi tertinggi.

---

# BAGIAN 11: VARIABEL LENGKAP YANG DIGUNAKAN

## 11.1 Model Configuration (dari src/models/dataset.py)

**Target:**
- SPEI_3 (Standardized Precipitation Evapotranspiration Index - 3 month scale)

**Static Categoricals:**
- location_id (Bojonegoro, Lamongan, Nganjuk, Ngawi, Tuban)

**Time-Varying Known Reals (diketahui di masa depan):**
- time_idx (indeks waktu continuous)
- month_sin (encoding cyclical bulan - sinus)
- month_cos (encoding cyclical bulan - cosinus)
- elevation (elevasi lokasi - static tapi masuk known reals)

**Time-Varying Unknown Reals (hanya diketahui di masa lalu):**
- SPEI_3 (nilai historis sebagai feature)
- precipitation_log (log transform curah hujan)
- et0_fao_evapotranspiration (evapotranspirasi potensial)
- soil_moisture (kelembapan tanah)
- temperature_2m_max (suhu maksimum)
- temperature_2m_min (suhu minimum)

**Normalizer:**
- GroupNormalizer per location_id, transformation=None (SPEI sudah normalized)

## 11.2 Encoder-Decoder Configuration
- MAX_ENCODER_LENGTH: 30 hari (history window)
- MAX_PREDICTION_LENGTH: 30 hari (forecast horizon)
- min_encoder_length: 15 hari
- min_prediction_length: 1 hari
- Quantile outputs: P10, P50, P90 (indices [1, 3, 5])
- Additional features: add_relative_time_idx, add_target_scales, add_encoder_length
- allow_missing_timesteps: True

---

# BAGIAN 12: CARA MENJALANKAN EVALUASI

## 12.1 Script Utama

```bash
# Evaluasi default (test period 2023-2025)
python evaluate.py

# Atau via wrapper dengan logging
python run_evaluation.py
```

## 12.2 Custom Test Period

**Opsi 1:** Edit `evaluate.py` line 14:
```python
def evaluate_model(checkpoint_path="...", test_year_start=2024):  # ubah 2023 ke 2024
```

**Opsi 2:** Panggil langsung di Python:
```python
from evaluate import evaluate_model
evaluate_model(test_year_start=2024)
```

**Opsi 3:** Custom checkpoint:
```python
evaluate_model(
    checkpoint_path="logs/checkpoints/epoch=8-val_loss=0.37.ckpt",
    test_year_start=2024
)
```

---

# BAGIAN 13: CATATAN PENTING

## 13.1 Kalibrasi
Prediksi yang "menempel" didorong oleh **per‑location calibration**:
- **Raw TFT:** Cenderung under-predict (bias +0.364)
- **Global Calibration:** Bias correction untuk semua lokasi (bias ~0, RMSE 0.372)
- **Per-Location Calibration:** Bias + variance scaling per lokasi (RMSE 0.363, R² 0.828) ✅

## 13.2 Output Files
- File utama: `predictions_eval.csv` dan `evaluation_metrics_detailed.json`
- File log: `evaluation_output_YYYYMMDD_HHMMSS.txt`
- File alternatif sesuai test period yang dipilih

## 13.3 Validasi Ulang
Jalankan `run_evaluation.py` untuk regenerate semua output dan log dengan timestamp baru.

## 13.4 Variabel Consistency
Semua variabel yang didokumentasikan di Bagian 11.1 **sudah terverifikasi sama** dengan kode di `src/models/dataset.py`.

## 13.5 Test Period
- **Default:** 2023-2025 (3 tahun, 3585 samples)
- **Configurable:** Via parameter `test_year_start` di `evaluate_model()`
- **Training:** Semua data sebelum test_year_start

## 13.6 Checkpoint Model
- **Best model:** logs/checkpoints/epoch=8-val_loss=0.37.ckpt
- **Validation loss:** 0.37
- **Training:** PyTorch Lightning dengan early stopping

---

**END OF DOCUMENTATION**  
**Last Updated:** 2026-01-19  
**Verified:** All variables, metrics, and file paths cross-checked with workspace ✅
