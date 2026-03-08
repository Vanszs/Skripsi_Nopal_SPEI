# Plan Fix & Audit Lanjutan — TFT SPEI Forecasting

> Dibuat: 2026-03-08  
> Status: **AKTIF**  
> Checkpoint saat ini: `enc30-epoch=1-val_loss=0.2892.ckpt`  
> Laporan detail: [`THESIS_READINESS_REPORT.md`](THESIS_READINESS_REPORT.md)

---

## Ringkasan Masalah Utama

Model TFT **kalah dari naive persistence** (SPEI(t) ≈ SPEI(t-1)):

| Metrik | TFT Model | Naive | Gap |
|--------|-----------|-------|-----|
| RMSE | 0.3915 | 0.2370 | +65% lebih buruk |
| MAE | 0.2755 | 0.0896 | +207% lebih buruk |
| Pearson r | 0.9051 | 0.9659 | |
| Skill Score | -65.2% | 0% | Negatif = kalah |

**Akar masalah**: SPEI-3 memiliki autocorrelation lag-1 = 0.97 (sangat smooth karena rolling 90 hari). Baseline naive yang hanya mengikuti observasi terakhir sudah sangat kuat. Model TFT under-dispersed (variance ratio 0.38–0.97) dan over-regularized.

---

## FASE 1: Bug Fix Pipeline (SELESAI ✅)

### 1.1 Double Denormalization di `full_evaluation.py` ✅
- **Masalah**: `model.predict(mode="raw")` sudah mengembalikan output yang di-denormalize oleh `transform_output()`. Manual `pv * scale + center` = double-denorm.
- **Fix**: Hapus manual denorm, pakai output langsung.
- **Dampak**: Minimal (±0.006 RMSE) karena GroupNormalizer near-identity pada SPEI.

### 1.2 Default Checkpoint Salah di `evaluate.py` ✅
- **Masalah**: Default = `epoch=0-val_loss=0.35.ckpt` (tidak ada).
- **Fix**: Diganti ke `enc30-epoch=1-val_loss=0.2892.ckpt`.

### 1.3 Checkpoint Selection di `run_evaluation.py` ✅
- **Masalah**: Pilih alphabetical terakhir, bukan val_loss terendah.
- **Fix**: `min(checkpoints, key=parse_val_loss)`.

### 1.4 Encoder Mismatch di `detailed_actual_vs_predict.py` ✅
- **Masalah**: Hardcoded `MAX_ENCODER_LENGTH=90` padahal checkpoint enc=30. Juga pakai `predict=True` + full data.
- **Fix**: Baca dari checkpoint hparams; pakai `predict=False` + test data only.

---

## FASE 2: Retraining — Encoder=90 (BELUM)

### Tujuan
Memberikan model konteks penuh rolling window SPEI-3 (90 hari) agar model bisa melihat seluruh informasi yang membentuk target.

### 2.1 Modifikasi Training Config
- **File**: `src/training/train.py`
- **Perubahan**:
  - `max_encoder_length=90` (sudah default di `dataset.py`, tapi train.py call harus eksplisit)
  - Kurangi regularisasi:
    - `dropout`: 0.35 → 0.20
    - `hidden_size`: 48 → 64
    - `attention_head_size`: 1 → 2
  - `max_epochs`: 60 → 80 (beri ruang lebih untuk konvergen)
  - `patience`: 25 → 30
- **Risiko**: GPU memory lebih besar (encoder 3x lebih panjang). RTX 3050 = 4GB VRAM.
  - Mitigasi: `batch_size` 32 → 16 jika OOM.

### 2.2 Hyperparameter Ablation Plan

| Experiment | Encoder | Hidden | Dropout | Head | Catatan |
|------------|---------|--------|---------|------|---------|
| A (baseline saat ini) | 30 | 48 | 0.35 | 1 | Sudah ada |
| B | 90 | 48 | 0.35 | 1 | Encoder saja |
| C | 90 | 64 | 0.20 | 2 | Kandidat utama |
| D | 90 | 128 | 0.15 | 4 | Aggressive (mungkin overfit) |

### 2.3 Perintah Eksekusi
```bash
# Experiment B
python -c "
from src.training.train import train_pipeline
train_pipeline(max_encoder_length=90, max_epochs=80, batch_size=32)
"

# Experiment C (modifikasi build_tft_model diperlukan)
# → Lihat Fase 2.4
```

### 2.4 Parameterize `build_tft_model`
- **File**: `src/models/tft.py`
- **Perubahan**: Tambah parameter `hidden_size`, `dropout`, `attention_head_size` ke `build_tft_model()` dan `train_pipeline()` agar bisa di-override tanpa edit file.

---

## FASE 3: Feature Engineering (BELUM)

### 3.1 Tambah SPEI Diff Feature
SPEI-3 sangat autokorelasi → model perlu belajar **ARAH perubahan**, bukan hanya level.

- **File**: `src/data/preprocess.py`
- **Feature baru**:
  ```python
  group["SPEI_3_diff"] = group["SPEI_3"].diff().fillna(0)
  ```
- **Registrasi**: Tambahkan `SPEI_3_diff` ke `time_varying_unknown_reals` di `dataset.py`

### 3.2 Tambah Lagged SPEI Features (Opsional)
- `SPEI_3_lag7`, `SPEI_3_lag30` — eksplisit lag agar model tidak perlu "belajar" shift sendiri.
- Hati-hati: jangan sampai leakage (lag harus dari waktu sebelumnya saja).

### 3.3 Seasonal Decomposition (Opsional)
- Dekomposisi SPEI-3 menjadi trend + seasonal + residual (STL)
- Prediksi residual saja → tambahkan trend+seasonal kembali saat evaluasi
- **Risiko**: Complexity bertambah, sulit dijelaskan di skripsi

---

## FASE 4: Loss Function & Training Strategy (BELUM)

### 4.1 Alternative Loss Functions
QuantileLoss cenderung over-smooth pada target yang highly autocorrelated.

| Opsi | Pro | Kontra |
|------|-----|--------|
| QuantileLoss (saat ini) | Standard, probabilistic | Compress variance |
| MAE (L1Loss) | Sharp predictions | Tidak ada uncertainty |
| RMSE (MSELoss) | Penalize outlier errors | Smooth, no quantiles |
| QuantileLoss + variance penalty | Best of both worlds | Custom implementation |

**Rekomendasi**: Tetap QuantileLoss (standar untuk TFT, dan skripsi perlu PICP), tapi perbaiki model capacity agar variance tidak compressed.

### 4.2 Learning Rate Schedule
- Saat ini: `reduce_on_plateau_patience=3` (sangat agresif)
- **Ubah** ke: `patience=5` agar LR tidak turun terlalu cepat

### 4.3 Gradient Clipping
- Saat ini: `gradient_clip_val=0.1` — cukup ketat
- Bisa coba `0.5` jika training lambat konvergen

---

## FASE 5: Audit Lanjutan — Validasi Setelah Fix (BELUM)

### 5.1 Checklist Evaluasi Post-Retraining

- [ ] Model RMSE < Naive RMSE (Skill Score > 0%)
- [ ] Variance ratio per lokasi mendekati 1.0 (0.7–1.3 acceptable range)
- [ ] PICP P10-P90 dalam range 0.75–0.85 (tidak over/under-coverage)
- [ ] Horizon degradation: RMSE Day-1 < Day-30 (expected, tapi gap reasonable)
- [ ] Bias overall < 0.05 (mendekati unbiased)
- [ ] Per-location: tidak ada lokasi dengan R² < 0.5

### 5.2 Cek Konsistensi Pipeline End-to-End

- [ ] `test_pipeline.py` — 6 tests PASS
- [ ] `evaluate.py` — checkpoint default benar, berjalan tanpa error
- [ ] `full_evaluation.py` — no double-denorm, metrics konsisten dengan `evaluate.py`
- [ ] `scripts/detailed_actual_vs_predict.py` — encoder length dari checkpoint
- [ ] `scripts/visualize_predictions.py` — CSV path mengarah ke run terbaru

### 5.3 Validasi SPEI Computation

- [ ] SPEI-3 mean ≈ 0 ± 0.1 per lokasi pada train set (sudah ✅: mean ≈ 0)
- [ ] SPEI-3 std ≈ 1.0 ± 0.1 pada train set (sudah ✅: std ≈ 1.02)
- [ ] Distribusi log-logistic (fisk) per bulan konvergen (tidak ada fallback NaN)
- [ ] Tidak ada data leakage: scaler fit hanya pada train (year < 2023)

### 5.4 Validasi Data Split Integrity

- [ ] Train tidak mengandung data 2023+
- [ ] Val hanya 2023 (1,825 rows = 365 × 5 lokasi)
- [ ] Test hanya 2024+ (1,835 rows ≈ 367 × 5 lokasi)
- [ ] `time_idx` kontinu dan tidak ada gap antar split

### 5.5 Audit GroupNormalizer Behavior

| Lokasi | Center | Scale | Catatan |
|--------|--------|-------|---------|
| Bojonegoro | -0.003 | 1.021 | Near-identity ✅ |
| Lamongan | -0.084 | 1.029 | Slight offset ⚠️ |
| Nganjuk | +0.002 | 0.984 | Near-identity ✅ |
| Ngawi | -0.003 | 1.003 | Near-identity ✅ |
| Tuban | -0.074 | 1.035 | Slight offset ⚠️ |

Center/scale near-identity mengonfirmasi SPEI sudah Z-score → GroupNormalizer hanya melakukan normalisasi ringan. Double-denorm dampaknya minimal tapi SALAH secara prinsip.

---

## FASE 6: Dokumentasi Skripsi (BELUM)

### 6.1 Update yang Diperlukan Setelah Fix
- Tabel metrik harus pakai hasil run terbaru (tanpa double-denorm)
- Jika retrain encoder=90, tambah tabel perbandingan ablasi
- Jelaskan mengapa naive baseline sangat kuat (autocorrelation SPEI-3)
- Tambah penjelasan skill score dan interpretasinya

### 6.2 Visualisasi yang Perlu Di-regenerate
- `A1_timeseries_per_lokasi.png` — dari run terbaru
- Scatter plot actual vs predicted
- Horizon degradation chart

---

## Prioritas Eksekusi

```
FASE 1 ████████████████████ SELESAI
FASE 2 ░░░░░░░░░░░░░░░░░░░ PRIORITAS 1 — Retrain enc=90
FASE 3 ░░░░░░░░░░░░░░░░░░░ PRIORITAS 2 — SPEI_3_diff
FASE 4 ░░░░░░░░░░░░░░░░░░░ PRIORITAS 3 — HP tuning
FASE 5 ░░░░░░░░░░░░░░░░░░░ SETELAH retrain selesai
FASE 6 ░░░░░░░░░░░░░░░░░░░ TERAKHIR — setelah metrik final
```

### Skenario Keputusan

```
Retrain enc=90 selesai
  │
  ├─ Skill Score > 0% → BERHASIL → lanjut Fase 5 + 6
  │
  └─ Skill Score masih negatif
       │
       ├─ Tambah SPEI_3_diff (Fase 3) → retrain lagi
       │
       └─ Masih gagal → tulis di skripsi bahwa SPEI-3 terlalu
          autokorelasi untuk daily 1-step TFT; fokuskan pada
          multi-step horizon (Day 7, 14, 30) di mana naive
          makin lemah dan TFT punya keunggulan
```

> **Catatan penting**: Pada horizon Day-30, naive RMSE ≈ 0.63 (jauh lebih buruk dari Day-1). Jika TFT bisa mempertahankan RMSE < 0.63 pada Day-30, itu sudah menunjukkan skill di long-range. Argumen skripsi bisa digeser ke "TFT unggul di multi-day horizon".
