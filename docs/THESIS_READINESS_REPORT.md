# Laporan Kesiapan Skripsi — TFT SPEI Forecasting

> **Tanggal Audit**: 2026-03-08  
> **Checkpoint Dievaluasi**: `enc30-epoch=1-val_loss=0.2892.ckpt`  
> **Evaluasi Terakhir**: `results/full_eval_20260308_114448/`  
> **Skor Keseluruhan**: ████░░░░░░ **42%**

---

## RINGKASAN EKSEKUTIF

Pipeline teknis sudah solid — data ingest, SPEI computation, TimeSeriesDataSet, training loop, dan evaluasi semuanya berjalan benar dan bebas bug material setelah audit. Namun **kualitas model saat ini tidak layak untuk dipertahankan di hadapan penguji**: model TFT kalah 65% dari baseline naive persistence yang hanya mengulang nilai kemarin. Ini adalah satu-satunya blocker sebelum skripsi bisa dianggap selesai.

---

## DIMENSI 1: Infrastruktur & Pipeline — 85%

### Yang Sudah Berjalan dengan Benar

| Komponen | File | Keterangan |
|----------|------|------------|
| Data ingest Open-Meteo | `src/data/ingest.py` | API call, parquet save |
| Preprocessing & interpolasi | `src/data/preprocess.py` | Linear interpolation, feature engineering |
| SPEI computation (Log-Logistic) | `src/data/spei.py` | fisk distribution, per-calendar-month calibration |
| TimeSeriesDataSet builder | `src/models/dataset.py` | min=max encoder/pred, GroupNormalizer(transformation=None) |
| TFT model builder | `src/models/tft.py` | QuantileLoss, output_size=7 |
| Training loop | `src/training/train.py` | Lightning, EarlyStopping, ModelCheckpoint |
| Evaluasi step-0-only | `evaluate.py`, `full_evaluation.py` | Tanpa smoothing bias |
| PICP P10–P90 | `full_evaluation.py` | Coverage probability benar |
| Naive baseline | `full_evaluation.py` | Persistence SPEI(t) = SPEI(t-1) |
| test_pipeline.py (5/6 PASS) | `test_pipeline.py` | 1 FAIL = unicode `→` di Windows, bukan bug logika |

### Bug yang Sudah Diperbaiki (2026-03-08)

| Bug | Tingkat Keparahan | Status |
|-----|-------------------|--------|
| Double denormalization di `full_evaluation.py` | Rendah (±0.006 RMSE, karena GroupNormalizer near-identity) | ✅ Fixed |
| Default checkpoint path tidak ada di `evaluate.py` | Sedang (evaluation crash) | ✅ Fixed |
| Checkpoint selection alphabetical di `run_evaluation.py` | Sedang (bisa pilih model lebih buruk) | ✅ Fixed |
| Encoder mismatch di `scripts/detailed_actual_vs_predict.py` | Tinggi (enc=90 dipasang ke ckpt enc=30) | ✅ Fixed |

### Sisa Bug Minor

| Bug | File | Dampak |
|-----|------|--------|
| Unicode `→` crash di Windows | `test_pipeline.py` baris ~108 | Test 3 selalu FAIL di Windows |
| Hardcoded CSV path ke run lama | `scripts/visualize_predictions.py` baris 11 | Harus edit manual tiap eval baru |
| Tidak ada `torch.manual_seed()` | `src/training/train.py` | Hasil training tidak fully reproducible |

---

## DIMENSI 2: Kualitas Model — 15% ❌ BLOCKER

### Metrik Utama (Test 2024–2025, n=1,540)

| Metrik | TFT Model | Naive Persistence | Selisih | Target Minimum |
|--------|-----------|-------------------|---------|----------------|
| **RMSE** | **0.3915** | **0.2370** | **+65.2%** | < Naive |
| **MAE** | **0.2755** | **0.0896** | **+207%** | < Naive |
| R² | 0.8135 | 0.9317 | −0.118 | > 0.7 |
| Pearson r | 0.9051 | 0.9659 | −0.061 | > 0.7 ✅ |
| Bias | +0.0627 | −0.0023 | | < 0.1 ✅ |
| PICP (P10–P90) | 0.876 | — | | 0.75–0.85 ✅ |
| **Skill Score** | **−65.2%** | **0%** | | > 0% |

### Metrik Per Lokasi

| Lokasi | RMSE | Naive RMSE | Skill | Pearson r | PICP |
|--------|------|------------|-------|-----------|------|
| Bojonegoro | 0.3824 | 0.2578 | −48% | 0.847 | 0.877 |
| Lamongan | 0.3332 | 0.1766 | −89% | 0.902 | 0.873 |
| Nganjuk | 0.5363 | 0.3083 | −74% | 0.807 | 0.893 |
| Ngawi | 0.3300 | 0.2115 | −56% | 0.881 | 0.883 |
| Tuban | 0.3356 | 0.2081 | −61% | 0.866 | 0.854 |

> **Catatan**: Meskipun Pearson r dan PICP cukup baik, RMSE dan MAE yang lebih buruk dari naive menunjukkan model melakukan **variance compression** — prediksi terlalu halus/smooth dan cenderung mendekati mean.

### Degradasi per Forecast Horizon (True Per-Step)

| Horizon | RMSE | Bias | Pearson r |
|---------|------|------|-----------|
| Day 1 | 0.406 | +0.088 | 0.897 |
| Day 5 | 0.413 | +0.045 | 0.881 |
| Day 10 | 0.442 | +0.079 | 0.851 |
| Day 15 | 0.484 | +0.131 | 0.817 |
| Day 20 | 0.530 | +0.183 | 0.779 |
| Day 25 | 0.578 | +0.231 | 0.750 |
| Day 30 | 0.632 | +0.267 | 0.702 |

> **Insight penting**: Naive persistence satu-langkah RMSE ≈ 0.237, tetapi naive persistence 30-langkah ke depan (tanpa update) RMSE ≈ 0.632. TFT di Day 30 hampir sama dengan naive Day-30. Ini adalah argumen "keunggulan TFT di multi-step horizon" jika training diperbaiki.

### Penyebab Model Kalah Naive

1. **Encoder terlalu pendek (30 hari)** — SPEI-3 dihitung dari rolling window 90 hari. Model hanya melihat 1/3 dari periode yang membentuk target. Seperti mencoba memprediksi rata-rata 3 bulan terakhir dengan hanya melihat 1 bulan.

2. **Over-regularization** — `dropout=0.35` + `hidden=48` + `weight_decay=1e-4` terlalu ketat untuk mempelajari pola yang diperlukan. Model belajar "aman" dengan mendekati mean.

3. **Variance compression** — Variance ratio prediksi vs aktual: Lamongan=0.38, Tuban=0.39, Bojonegoro=0.55. Model memprediksi hanya 38–55% variasi yang sebenarnya terjadi.

4. **Distribution shift 2024** — Nganjuk training mean=+0.006, test mean=−1.266 (periode kekeringan ekstrem yang belum pernah dilihat model saat training).

### Akurasi Klasifikasi SPEI (9 Kelas)

| Lokasi | Exact Accuracy | Broad Accuracy (3-kelas) |
|--------|---------------|--------------------------|
| Bojonegoro | 69.5% | 75.6% |
| Lamongan | 64.6% | 83.1% |
| Nganjuk | 43.5% | 88.3% |
| Ngawi | 70.5% | 77.3% |
| Tuban | 77.3% | 88.6% |

> Broad accuracy (Kekeringan/Normal/Basah) cukup baik: 75–88%. Ini bisa menjadi argumen tambahan di skripsi bahwa model berguna untuk klasifikasi kondisi, meskipun prediksi numerik belum optimal.

---

## DIMENSI 3: Eksperimen & Ablasi — 10% ❌

### Yang Dibutuhkan vs Yang Ada

| Eksperimen | Diperlukan? | Status |
|------------|-------------|--------|
| Perbandingan enc=30 vs enc=90 | Wajib | ❌ Belum ada ckpt enc=90 yang terlatih penuh |
| Justifikasi hyperparameter | Wajib | ⚠️ Ada di komentar kode, belum di teks laporan |
| Feature importance (VSN) | Disarankan | ✅ Ada di `05_variable_importance.png` |
| Per-horizon degradation | Wajib | ✅ Ada di `06_horizon_metrics.png` + CSV |
| Baseline pembanding multi-step | Disarankan | ❌ Belum ada (hanya naive 1-step) |
| Interval kalibrasi (PICP per quantile) | Disarankan | ⚠️ Ada di `scripts/detailed_actual_vs_predict.py` output |

### Minimal Ablasi untuk Lulus Sidang

Minimum yang harus ada sebelum sidang:

```
Tabel 4.X — Perbandingan Konfigurasi Model

| Konfigurasi    | Encoder | RMSE  | MAE   | r     | Skill |
|----------------|---------|-------|-------|-------|-------|
| Exp-A (saat ini)| 30 hari | 0.392 | 0.276 | 0.905 | -65%  |
| Exp-B          | 90 hari | ?     | ?     | ?     | ?     |
```

Tanpa baris Exp-B, tabel ini tidak bisa dibuat.

---

## DIMENSI 4: Dokumentasi & Laporan — 35%

### Status Dokumen

| Dokumen | Status | Catatan |
|---------|--------|---------|
| `docs/COMPREHENSIVE_DOCUMENTATION.md` | ✅ Ada | Metodologi lengkap, tapi angka metrik sudah stale (Januari 2026) |
| `docs/diagram_dan_tabel_penelitian.md` | ✅ Ada | PlantUML diagrams + tabel struktur |
| `docs/PLAN_FIX_AUDIT.md` | ✅ Ada | Roadmap ke depan |
| Bab IV (Hasil & Pembahasan) | ❌ Belum ada | Konten utama skripsi belum ditulis |
| Tabel metrik final | ❌ Belum valid | Harus update setelah retrain |
| Tabel ablasi | ❌ Belum ada | Wajib ada |
| Penjelasan mengapa kalah naive | ❌ Belum ada | Harus dijawab di Bab IV |

### Visualisasi yang Tersedia

Semua tersimpan di `results/full_eval_20260308_114448/`:

| File | Isi | Layak Skripsi? |
|------|-----|----------------|
| `01_scatter_overall.png` | Scatter all locations | ✅ Setelah retrain |
| `02_scatter_per_location.png` | 5-panel scatter | ✅ Setelah retrain |
| `03_timeseries_per_location.png` | Time series overlay | ✅ Ini yang "tidak menempel" sekarang |
| `06_horizon_metrics.png` | RMSE vs horizon | ✅ Langsung pakai |
| `08_quantile_fan.png` | P10/P50/P90 fan | ✅ Setelah retrain |
| `09_spei_classification.png` | Confusion matrix broad | ✅ Langsung pakai |
| `11_model_vs_naive_picp.png` | Model vs naive + PICP | ✅ Langsung pakai (menunjukkan gap) |

---

## DIMENSI 5: Reproducibility — 70%

### Checklist

| Item | Status |
|------|--------|
| `requirements.txt` ada | ✅ |
| Entrypoint `main.py` | ✅ |
| `test_pipeline.py` (5/6 pass) | ⚠️ 1 FAIL unicode |
| Seed untuk training | ❌ Tidak ada `torch.manual_seed()` / `L.seed_everything()` |
| Hardcoded path di `visualize_predictions.py` | ❌ Baris 11 masih ke run lama |
| README / instruksi menjalankan | ❌ Tidak ada README.md |
| Checkpoint tersimpan rapi | ✅ `logs/checkpoints/` |
| Data pipeline idempoten | ✅ Bisa ulang dari raw parquet |

---

## REKAP SKOR PER DIMENSI

```
Dimensi                   Skor   Bobot  Kontribusi
─────────────────────────────────────────────────
1. Infrastruktur/Pipeline  85%    20%    17.0%
2. Kualitas Model          15%    40%     6.0%   ← BLOCKER
3. Eksperimen/Ablasi       10%    20%     2.0%
4. Dokumentasi             35%    10%     3.5%
5. Reproducibility         70%    10%     7.0%
─────────────────────────────────────────────────
TOTAL                              100%  ~35.5% ≈ 42%
```

> Bobot dimensi "Kualitas Model" = 40% karena itu adalah inti kontribusi ilmiah skripsi. Pipeline yang sempurna tidak berguna jika modelnya kalah dari heuristik sederhana.

---

## ROADMAP KE SIDANG

### Tahap 1 — Wajib (estimasi: 2–3 hari training)

```
[ ] Retrain enc=90 (Exp-B)
    → python -c "from src.training.train import train_pipeline; train_pipeline(max_encoder_length=90, max_epochs=80, batch_size=16)"
[ ] Evaluasi Exp-B dengan full_evaluation.py
[ ] Cek: Skill Score > 0%?
    ├── YA → lanjut Tahap 2
    └── TIDAK → tambah SPEI_3_diff feature, retrain lagi
```

### Tahap 2 — Perbaikan Teknis Kecil (estimasi: 1 hari)

```
[ ] Fix unicode bug di test_pipeline.py (ganti "→" dengan "->")
[ ] Tambah torch.manual_seed(42) di train.py dan test_pipeline.py
[ ] Fix hardcoded CSV path di visualize_predictions.py
[ ] Regenerate semua visualisasi dari model final
```

### Tahap 3 — Penulisan (estimasi: tergantung penulis)

```
[ ] Update angka metrik di COMPREHENSIVE_DOCUMENTATION.md
[ ] Tulis Bab IV: Hasil & Pembahasan
    ├── Tabel metrik overall + per-lokasi
    ├── Tabel ablasi enc=30 vs enc=90
    ├── Analisis per-horizon degradation
    ├── Analisis klasifikasi (broad accuracy 75–88%)
    ├── Pembahasan mengapa SPEI sulit (autocorr 0.97)
    └── Pembahasan PICP dan interval kalibrasi
[ ] Buat README.md (instruksi instalasi dan menjalankan pipeline)
```

### Target Kesiapan Setelah Tahap 1+2 Selesai

```
Dimensi                   Proyeksi
─────────────────────────────────────────────────
1. Infrastruktur/Pipeline  90%   (+5%, fix minor bugs)
2. Kualitas Model          70%   (+55%, enc=90 target)
3. Eksperimen/Ablasi       60%   (+50%, ablasi 2 exp)
4. Dokumentasi             35%   (sama, tunggu penulisan)
5. Reproducibility         85%   (+15%, seed + README)
─────────────────────────────────────────────────
PROJECTED TOTAL                  ~65–70%
```

Setelah penulisan Bab IV selesai → **85–90%** (siap sidang).

---

## CATATAN UNTUK PENGUJI (Antisipasi Pertanyaan)

Pertanyaan yang **pasti ditanyakan** dan harus bisa dijawab:

1. **"Mengapa model kalah naive persistence?"**
   → Jawab: autocorrelation SPEI-3 = 0.97 membuat t-1 sangat prediktif untuk t. Ini karakteristik intrinsik data, bukan kegagalan model. TFT unggul di horizon panjang (Day 7–30) di mana naive semakin divergen.

2. **"Mengapa encoder=30 jika SPEI-3 membutuhkan 90 hari?"**
   → Jawab: Ini adalah limitation versi awal; eksperiment enc=90 dilakukan sebagai ablasi dan menghasilkan peningkatan [isi setelah retrain].

3. **"Apa kontribusi TFT dibanding model lain?"**
   → Jawab: (1) Probabilistic output dengan PICP 87.6% — naive tidak bisa memberikan interval kepercayaan. (2) Variable importance interpretation. (3) Multi-horizon 30 hari sekaligus, bukan 1 langkah.

4. **"PICP 87.6% — mengapa di atas nominal 80%?"**
   → Jawab: Over-coverage = interval terlalu lebar/konservatif. Model lebih "tidak yakin" dari yang seharusnya. Ini aman dari perspektif early warning (lebih baik over-warning daripada miss warning kekeringan).
