# PLAN FIX ROBUST — TFT-SPEI Pipeline (Post-Audit)

Tanggal: 2026-05-31
Basis: hasil audit swarm (4 sub-agent) + verifikasi manual.
Tujuan global: SPEI drought forecasting Jawa Timur dengan Temporal Fusion Transformer,
schema v2, super-node per kota, **leakage-safe**, **reproducible**, dan **konsisten kode↔dokumen↔judul**.

Aturan eksekusi: edit semua sekaligus, lalu **audit ulang persis prosedur awal** (swarm 4 track + verifikasi manual). Lapor hanya jika robust.

---

## 0. KEPUTUSAN TERKONFIRMASI (dari klarifikasi judul asli)

Judul asli: **"Peramalan Multi-Horizon Indeks Kekeringan Lahan Pertanian (SPEI) di Sentra Padi Jawa Timur Menggunakan Temporal Fusion Transformer (TFT)"**.

- **Q1 — Multi-horizon (CONFIRMED):** Model TFT sudah benar memprediksi 30 hari sekaligus (direct multi-horizon, `max_prediction_length=30`). **Tidak perlu ubah model / tidak perlu retrain untuk ini.** Yang diubah hanya **pelaporan metrik angka**:
  - Metrik headline (`metrics_summary.json`, `metrics_report.txt`, MD report) = **rata-rata seluruh 30 horizon**, bukan t+1 saja.
  - Tetap tampilkan **degradasi per-horizon h=1..30** (sudah ada: `horizon_metrics.csv`, `06_horizon_metrics.png`). t+1 tetap ada sebagai titik h=1.
  - Gambar timeseries actual-vs-predict & plot lain **tidak berubah**.
  - → Menggantikan/ memperjelas temuan M1.
- **Q2 — Kalibrasi per lokasi (CONFIRMED, Opsi C hybrid):** Proposal (rumusan masalah #2, tujuan khusus #2) menjanjikan "kalibrasi per lokasi" sebagai metode, tapi kode hanya **mengukur** (PICP per kota), belum **melakukan** kalibrasi. Catatan penting: super-node = agregasi DATA INPUT (sudah benar) ≠ kalibrasi = koreksi OUTPUT interval P10–P90. Keputusan:
  - **Ukur** (sudah ada): PICP + reliability per kota.
  - **Koreksi ringan per kota (BARU):** faktor pelebaran/penyempitan interval P10–P90 di-fit dari **validation per kota**, diterapkan ke **test** (leakage-safe). Metode: per-city conformal / quantile offset. Modul kecil `src/evaluation/calibration.py`.
  - **Bandingkan sebelum vs sesudah:** PICP per kota → mendekati nominal 80% = bukti "efektivitas kalibrasi per lokasi" (tujuan khusus #2).
  - **Tanpa retrain** (murni pasca-proses output). **LARANGAN:** jangan fit kalibrasi dari statistik test/ground-truth (notebook lama `verify_fix.py` curang — jangan diangkat ke pipeline).
  - → Gap judul-vs-kode baru: **GAP-A**.
- **Q3 — Deteksi kekeringan parah (CONFIRMED, saran diterima):** Tujuan khusus #4 janji "mendeteksi kejadian kekeringan parah". Kode kini hanya `classify_spei` (9 kelas deskriptif). Keputusan: **tambah metrik deteksi event** sebagai event biner "kering" (ambang **SPEI ≤ −1.5** = parah+ekstrem; sekunder ambang **SPEI ≤ −1.0** untuk sedang+). Metrik: **POD/Recall, FAR, F1, CSI** + confusion, dihitung dari P50 vs aktual, **per kota & overall, per beberapa horizon**. Reuse `classify_spei` (ambang konsisten). **Tanpa retrain** (pasca-proses). Output ke `metrics_summary.json` + 1 plot ringkas.
  - → Gap judul-vs-kode baru: **GAP-B**.
- **Q4 — Jumlah variabel (CONFIRMED: 8 / Opsi A):** Pakai **8 variabel**. Tambah 3: `relative_humidity_2m_mean`, `shortwave_radiation_sum`, `wind_speed_10m_mean` sebagai **input model TFT** (bukan untuk hitung SPEI; ET0 tetap dipakai untuk water_deficit). Konsekuensi: **WAJIB re-ingest 21thn** dari Open-Meteo (3 var baru) → regenerate preprocess → retrain. Edit: `ingest.py` (`REQUIRED_VARIABLES`+mapping), `preprocess.py` (`WEATHER_COLS` + `_validate_raw_schema`), `dataset.py` (`time_varying_unknown_reals`+`real_scalers`+`_validate_schema`). → menggantikan H1; default plan jadi Opsi A.
- **Q5 — Daftar kabupaten (CONFIRMED, FINAL):** 5 kabupaten sentra padi sudah final: **Lamongan, Ngawi, Bojonegoro, Tuban, Nganjuk** (`data/config/city_centers.json`). Tidak diubah. Pipeline tetap config-driven (jangan hardcode jumlah).
- **Q6 — Encoder/Horizon (CONFIRMED, FINAL):** `max_encoder_length=90` (3 bulan, selaras SPEI-3) & `max_prediction_length=30` (jangka pendek-menengah). **Dikunci** — tidak diubah lagi agar tidak ada retrain berulang.
- **GPU — Training wajib maksimal pakai GPU (CONFIRMED):** Hardware terdeteksi = **NVIDIA RTX 3050 Laptop, VRAM hanya 4 GB**, `torch.cuda.is_available()=True`. Requirement:
  - `accelerator="gpu"`, `devices=1` (sudah ada). **Gagalkan/peringatkan keras bila CUDA tidak tersedia** saat training produksi (jangan diam-diam fallback CPU).
  - **Mixed precision `precision="16-mixed"`** (saat ini `32`) → hemat VRAM + lebih cepat di tensor core; krusial untuk 4 GB.
  - **Batas VRAM 4 GB**: `batch_size` harus dipilih agar muat (mulai 32; turunkan ke 16/8 bila OOM). Jika perlu batch efektif besar, pakai `accumulate_grad_batches`. Catat batch final di `run_config.json`.
  - `pin_memory=True` (sudah). `num_workers`: boleh dinaikkan (>0) bila stabil di Linux untuk throughput; default 0 aman.
  - `torch.set_float32_matmul_precision("medium")` (sudah) tetap.
  - Verifikasi: log device aktual + `nvidia-smi` utilisasi saat training; pastikan benar-benar di GPU, bukan CPU.

**STATUS: SEMUA Q1–Q6 + GPU TERKONFIRMASI. Plan siap dieksekusi.**

---

## 1. RINGKASAN TEMUAN → AKSI

| ID | Severity | Temuan | Aksi inti |
|----|----------|--------|-----------|
| C1 | CRITICAL | Fit distribusi SPEI (fisk) memakai SELURUH seri (2005–2026) → target val/test bocor | Fit train-only (≤ cutoff), transform seluruh seri |
| GAP-A | HIGH | Judul/proposal janjikan "kalibrasi per lokasi" (tujuan khusus #2); kode hanya UKUR PICP, belum LAKUKAN kalibrasi | Modul `src/evaluation/calibration.py` per-city fit-on-val (Opsi C); PICP before/after |
| GAP-B | HIGH | Tujuan khusus #4 janji "deteksi kekeringan parah"; kode hanya klasifikasi deskriptif, belum ada metrik deteksi event | Metrik event biner (SPEI≤−1.5): POD/Recall, FAR, F1, CSI per kota+horizon |
| C2 | CRITICAL | Migrasi hanya di kode; artefak `spei_dataset.parquet` masih ada `SPEI_6`, `run_config.json` masih enc=30, belum ada checkpoint enc=90 produksi | Regenerate artefak → retrain enc=90 → regenerate `results/` |
| H1 | HIGH | Judul/metodologi rekomendasi 8 variabel, kode pakai 5 | Tambah 3 variabel end-to-end **atau** dokumentasikan 5-core (lihat §4) |
| H2 | HIGH | Dokumen masih enc=30 / output_size=7 / dropout=0.35 / hidden=48 / SPEI_6 | Sinkronkan dokumen ke konfigurasi aktual |
| H3 | HIGH | Hardcode `selected_node_count == 5` di train.py | Baca `top_k` dari meta artefak |
| M1 | MEDIUM | Metrik "overall" = step-0 (t+1) saja → ambigu vs klaim "forecasting" | Tambah metrik rata-rata horizon + label eksplisit |
| M2 | MEDIUM | `except Exception: → NaN` di SPEI (asal-fix, menelan bug) | Persempit exception + warning |
| M3 | MEDIUM | `SPEI_3_diff.fillna(0.0)` memalsukan baris pertama | Pakai NaN → dibuang dropna |
| M4 | MEDIUM | `ffill()` tanpa limit (rawan jalarkan nilai basi) | `limit=7` + warning |
| M5 | MEDIUM | `detailed_actual_vs_predict.py` pakai `return_y=True` (potensi mismatch skala) | Pakai `preds.x["decoder_target"]` |
| M6 | MEDIUM | `min_len` truncation menyembunyikan bug shape | Assert panjang sama |
| L1 | LOW | Default checkpoint hardcoded di `evaluate.py` | Default `None` + resolusi dinamis |
| L2 | LOW | File mati / clutter (`_diag2.py`, `sonar_audit/`, 400+ `lightning_logs/`) | `.gitignore` + arsip |
| S1 | HIGH | `super_node_id` & `city_id` keduanya static_categoricals padahal 1-to-1 (collinear, embedding redundan, rawan hafal identitas) | Pertahankan 1 saja sbg categorical (lihat §8) |
| S2 | HIGH | `location_id == super_node_id` 100% identik (kolom alias redundan, sumber ambiguitas fallback eval) | Jadikan `super_node_id` satu-satunya entity key; deprecate `location_id` |
| S3 | MEDIUM | Agregasi mean membuang variabilitas intra-kota; klaim doc "tanpa kehilangan konteks lokal" overstate | Tambah fitur dispersi (std) ATAU dokumentasikan sebagai limitation |
| S4 | MEDIUM | Hardcode `== 5` super-node juga di `smoke_e2e_v2.py:102` & `main.py:141` (selain train.py H3) | Samakan ke `top_k` dari meta (config-driven) |
| S5 | LOW | `entity_count_gt_5` di smoke test asumsikan >5 kota, produksi tepat 5 (kontras asumsi) | Selaraskan assertion ke kardinalitas aktual / beri komentar |

---

## 2. FILE YANG DIEDIT (langsung diubah logic-nya)

### 2.1 `src/data/spei.py`  — C1, M2
- **`calculate_spei(series, scale=3, fit_mask=None)`**: tambah parameter `fit_mask` (boolean Series index-aligned, True = periode train).
  - Fit `fisk` per bulan **hanya** pada `d_accumulated[fit_mask & month_mask]`.
  - Transform (CDF→z) tetap diterapkan ke **seluruh** baris bulan tsb (train+val+test) memakai parameter train.
  - `shift` (offset domain positif) dihitung dari **data train** lalu dipakai konsisten untuk fit & transform.
  - **Edge-case**: nilai val/test bisa < min train → `shifted_month` jatuh ke domain ≤0 (fisk tak terdefinisi). Tangani: clip `shifted_month` ke `>= eps` sebelum `fisk.cdf`, dan dokumentasikan bahwa nilai di luar dukungan train di-clamp (jangan NaN diam-diam).
- Persempit `except Exception` → `except (RuntimeError, ValueError, FloatingPointError)` + `warnings.warn(...)` (M2).
- Klarifikasi docstring: ET0 dipakai sebagai proxy PET (LOW #9).

### 2.2 `src/data/preprocess.py` — C1, M3, M4
- Saat memanggil `calculate_spei`, bentuk `fit_mask = indexed.index <= pd.Timestamp(selection_end_date)` dan teruskan (C1).
- `SPEI_3_diff`: ganti `.fillna(0.0)` → biarkan NaN; baris NaN dibuang oleh `dropna(subset=["SPEI_3","water_deficit"])` (tambahkan `SPEI_3_diff` opsional ke subset atau biarkan dataset handle) (M3).
- `_interpolate_per_node`: `ffill(limit=7)` + warning bila ada gap > limit tersisa (M4).
- Pastikan tidak ada lagi penulisan `SPEI_6` (sudah bersih, verifikasi).

### 2.3 `src/training/train.py` — H3, S4, GPU
- `_validate_training_schema`: hapus hardcode `== 5`.
  - Baca `top_k` dari `data/processed/node_selection_v2.meta.json` (fallback: gunakan modus `selected_node_count` dari data, dengan syarat seragam per kota).
  - Validasi `selected_node_count` seragam == `top_k` untuk semua kota.
- **GPU maksimal (VRAM 4 GB)**: `precision="16-mixed"` (dari 32); guard error/warning keras bila CUDA tak tersedia di run produksi; pilih `batch_size` muat 4 GB (32→turun bila OOM) + opsi `accumulate_grad_batches`; log device aktual. `run_experiment.py` teruskan flag precision/batch.

### 2.3b `src/models/dataset.py` — S1, S2
- `static_categoricals`: hindari collinear `super_node_id`+`city_id`. Pilih sesuai §8 (default: pertahankan `super_node_id` sebagai group_id + satu static categorical; drop `city_id` dari static_categoricals ATAU sebaliknya — satu saja).
- Kurangi ketergantungan `location_id` (alias redundan): tetap divalidasi ada untuk kompatibilitas, tapi entity key tunggal = `super_node_id`.

### 2.3c `main.py` & `scripts/smoke_e2e_v2.py` — S4, S5
- `main.py:141`: validasi `selected_node_count` terhadap `top_k` (bukan implisit 5).
- `smoke_e2e_v2.py:102-103`: ganti `selected_node_count == 5` → `== top_k`; sesuaikan/komentari `entity_count_gt_5` agar tidak kontradiktif dengan produksi (5 entitas).

### 2.4 `full_evaluation.py` & `run_experiment.py` — M1
- `full_evaluation.py`: tambah `overall_all_horizons` (rata-rata metrik lintas 30 horizon) ke `metrics_summary.json`; label `overall` = "t+1 (step-0)" di JSON.
- Tambah field `skill_score` (1 - RMSE_model/RMSE_naive) agar konsisten dgn `run_experiment.py`.
- `run_experiment.py._write_md_report`: beri label eksplisit metrik headline = "t+1 (step-0)" + tampilkan `overall_all_horizons`.

### 2.5 `evaluate.py` — L1
- Default `checkpoint_path=None`; resolusi via `_checkpoint_from_run_config()` lalu `_best_checkpoint()`; baris 438 fallback ikut dinamis.

### 2.6 `scripts/detailed_actual_vs_predict.py` — M5, M6, S2
- Ganti `return_y=True` → ambil aktual dari `preds.x["decoder_target"]` (skala konsisten dgn `mode="raw"`).
- Gunakan `MODEL_GROUP_COL` untuk grouping per-entity (baris 210 ganti `location_id` → `MODEL_GROUP_COL`).
- Ganti `min_len` truncation → `assert len(a)==len(b)`.

### 2.7 `test_pipeline.py` — M6
- Ganti truncation `min_len` → assert panjang sama; rename `actuals_raw`→`actuals_normalized` (kosmetik kejelasan).

### 2.10 Konsumen hilir `location_id` (S2) & path hardcoded
- File yang pakai fallback `location_id` sebagai entity: `evaluate.py:49-50`, `full_evaluation.py:79-80`, `scripts/visualize_predictions.py:39`, `notebooks/verify_fix.py:26`, `notebooks/visualize_fix.py:31`, `_diag2.py:18`.
- Aksi: pastikan semua resolusi entity konsisten pakai `MODEL_GROUP_COL` lebih dulu (fallback `location_id` boleh tetap untuk kompatibilitas, **tapi tidak boleh** jadi grouping utama). Verifikasi tidak ada skrip yang mengandalkan `location_id` ≠ `super_node_id`.
- `scripts/visualize_predictions.py`: hardcoded CSV path ke run lama (temuan audit awal) → jadikan argumen/auto-resolve ke `results/` terbaru.

### 2.8 Dokumen — H2, S3
- `docs/COMPREHENSIVE_DOCUMENTATION.md`: enc 30→90, prediction length, fitur, quantile.
- `docs/THESIS_READINESS_REPORT.md`: output_size 7→3, quantiles [0.1,0.5,0.9], hidden 64, dropout 0.20, heads 2, hcs 10; update checkpoint terbaru.
- `docs/PLAN_FIX_AUDIT.md`, `now_percentage.md`: arsipkan/tandai usang (atau hapus).
- (S3) `DOKUMENTASI_METODE...SUPERNODE.md`: koreksi klaim "tanpa kehilangan konteks lokal" → nyatakan agregasi mean membuang variabilitas intra-kota (limitation), atau dokumentasikan fitur dispersi bila ditambahkan.

### 2.9 `.gitignore` — L2
- Tambah `lightning_logs/` (root & notebooks), `notebooks/lightning_logs/`. Pindahkan/arsip `_diag2.py`, `sonar_audit/`, `sonar_*.txt` (opsional, non-blocking).

---

## 3. FILE TERDAMPAK (tidak diedit, tapi WAJIB di-regenerate / re-run) — C2

Urutan eksekusi pipeline (end-to-end):
1. `data/processed/spei_dataset.parquet` → **regenerate** via `preprocess_pipeline()` (hapus SPEI_6, terapkan fix C1).
2. `data/processed/node_selection_v2.parquet` + `.meta.json` → ikut ter-regenerate.
3. Training enc=90 → checkpoint baru `logs/checkpoints/enc90-run...ckpt` + `logs/run_config.json` baru.
4. `results/full_eval_*` → **regenerate** via `full_evaluation.py` dari checkpoint enc=90.
5. Artefak lama (checkpoint enc30, results lama) → arsip, jangan dipakai untuk laporan.

> Catatan biaya: langkah 3 (training) butuh GPU + waktu. Jika ingest ulang (H1 opsi A) dijalankan, langkah 0 (ingest) butuh jaringan + rate-limit Open-Meteo.

---

## 4. KEPUTUSAN H1 (5 vs 8 variabel)

Dua jalur, pilih satu sebelum eksekusi:

- **Opsi A — Selaraskan kode ke judul (8 variabel).** Tambah 3 var: `relative_humidity_2m_mean`, `shortwave_radiation_sum`, `wind_speed_10m_mean`. Edit `src/data/ingest.py` (`REQUIRED_VARIABLES` + mapping `df_data`), `WEATHER_COLS` di `preprocess.py`, dan `time_varying_unknown_reals` + `real_scalers` + `_validate_schema` di `src/models/dataset.py`, serta `_validate_raw_schema`/`WEATHER_COLS` guard di preprocess. Konsekuensi: **re-ingest 21thn × ~45 node** (jaringan, lambat, bisa rate-limit) → regenerate seluruh pipeline. Catatan: nama kolom Open-Meteo `soil_moisture_0_to_7cm_mean` disimpan sebagai `soil_moisture` (pola alias sudah ada di ingest).
- **Opsi B — Selaraskan judul ke kode (5 core).** Tidak ubah kode data; tulis justifikasi eksplisit di metodologi bahwa 5 variabel core SPEI dipakai (precipitation, ET0, soil_moisture, temp_max, temp_min), 3 sisanya = future work. Lebih cepat, tanpa risiko jaringan.

**KEPUTUSAN FINAL (Q4): Opsi A — 8 variabel.** Re-ingest wajib. Fallback Opsi B hanya bila ingest gagal total setelah retry (resume mode ada di ingest.py).

---

## 4b. KEPUTUSAN S1/S2 (struktur entitas super-node)

Terverifikasi: `super_node_id` ⟷ `city_id` adalah **1-to-1**, dan `location_id == super_node_id` 100%. Jadi ada 3 kolom yang secara efektif merepresentasikan entitas yang sama. Pilih satu skema sebelum retrain:

- **Opsi 1 (default) — `super_node_id` sebagai satu-satunya entity key.**
  - `group_ids=[super_node_id]` (tetap).
  - `static_categoricals=[super_node_id]` saja (drop `city_id` dari static_categoricals karena redundan).
  - `location_id` dipertahankan hanya untuk kompatibilitas validasi, tidak dipakai sebagai grouping.
  - Konsekuensi: hilangkan embedding ganda; lebih bersih; perlu retrain.
- **Opsi 2 — `city_id` sebagai static categorical, `super_node_id` hanya group_id.**
  - Berguna jika nanti >1 super-node per kota; untuk sekarang (1-to-1) efeknya setara Opsi 1.

> Default plan: **Opsi 1**. Karena hanya 5 entitas, mempertahankan dua categorical collinear berisiko model menghafal identitas kota. Keputusan final dikonfirmasi sebelum retrain (memengaruhi arsitektur embedding).

---

## 5. URUTAN EKSEKUSI

1. Backup/arsip dulu (§9): rename `results/` lama, salin `logs/run_config.json` & checkpoint enc30 ke folder arsip.
2. Edit kode logic fixes: §2.1, §2.2, §2.3, §2.3b, §2.3c, §2.4, §2.5, §2.6, §2.7, §2.10 (semua, termasuk super-node S1/S2/S4).
3. (H1 §4) & (S1/S2 §4b): putuskan & terapkan opsi.
4. Regenerate data: `preprocess_pipeline()` (+ ingest bila Opsi A).
5. Verifikasi parquet: tidak ada `SPEI_6`; ada `SPEI_3`,`SPEI_3_diff`; tidak ada NaN target; `(super_node_id,time)` unik; **+ verifikasi anti-leakage C1 (§10)**.
6. Cek reproducibility (§10): rerun preprocess seed sama → node selection identik (fingerprint meta cocok).
7. Retrain enc=90 (`run_experiment.py --encoder 90`) → checkpoint + run_config baru.
8. Regenerate evaluasi (`full_evaluation.py`) → `results/` baru + skill score.
9. Sinkronkan dokumen §2.8.
10. Jalankan `test_pipeline.py` + `scripts/smoke_e2e_v2.py` → semua PASS.
11. Commit + push per checkpoint (§9).
12. **AUDIT ULANG** (swarm 4 track + verifikasi manual) persis prosedur awal.
13. Lapor bila robust.

---

## 6. KRITERIA "ROBUST" (definition of done)

- [ ] SPEI fit train-only terbukti (parameter tidak berubah saat val/test ditambah).
- [ ] `spei_dataset.parquet` tanpa SPEI_6; target tanpa NaN; unik per (super_node_id,time).
- [ ] `run_config.json` `max_encoder_length=90` + checkpoint enc=90 nyata.
- [ ] `metrics_summary.json` punya overall(step-0, berlabel) + overall_all_horizons + skill_score; PICP benar.
- [ ] Tidak ada hardcode jumlah node; `top_k` dari meta.
- [ ] Entity super-node: tidak ada static_categorical collinear (super_node_id vs city_id); `location_id` tidak dipakai sbg grouping; tidak ada hardcode `==5` super-node di main.py/smoke.
- [ ] Dokumen konsisten dengan kode (enc=90/3-quantile/no SPEI_6/fitur final) + limitation agregasi mean dinyatakan.
- [ ] `test_pipeline.py` PASS penuh.
- [ ] Audit ulang: tidak ada CRITICAL/HIGH tersisa, tidak ada regresi "worse-than-before".

---

## 7. RISIKO & MITIGASI

| Risiko | Mitigasi |
|--------|----------|
| Training enc=90 lama / tanpa GPU | Jalankan `--epochs` wajar; checkpoint top_k=1; bisa dijalankan async oleh user |
| Re-ingest (Opsi A) rate-limit/gagal | Resume mode sudah ada di ingest.py; fallback Opsi B |
| Fix C1 mengubah skala SPEI → metrik berubah | Wajar & benar; bandingkan distribusi train vs full pasca-fix |
| Regenerate menimpa artefak lama | Arsipkan dulu (rename folder results lama) sebelum overwrite |

---

## 8. STRATEGI BACKUP, COMMIT & PUSH (§9 di urutan eksekusi)

- **Arsip sebelum overwrite**: `results/` → `results_archive_enc30_<tanggal>/`; salin `logs/run_config.json` + checkpoint enc30 terbaik ke `logs/archive_enc30/`.
- **Commit bertahap** (agar mudah revert): (a) commit logic fixes (kode) sebelum retrain; (b) commit artefak/doc setelah retrain & evaluasi.
- **Push**: ke `origin/main` hanya setelah `test_pipeline.py` PASS pada tiap tahap. Jangan push artefak besar (cek `.gitignore`: parquet/ckpt/png sudah di-ignore).

---

## 9. VERIFIKASI ANTI-LEAKAGE (C1) & REPRODUCIBILITY (§5/§6 di urutan eksekusi)

- **Bukti C1 konkret**: hitung parameter fisk per-bulan saat fit pakai (i) hanya train vs (ii) train+val+test. Parameter **harus identik** pada skema train-only. Simpan ringkasan ke log/console sebagai bukti.
- **Cross-check distribusi**: SPEI train-only vs SPEI lama (full-fit) — laporkan pergeseran mean/std (wajar berubah, dokumentasikan).
- **Reproducibility**: rerun `preprocess_pipeline()` dengan seed sama → `node_selection_v2.meta.json` fingerprint (`raw_fingerprint_sha256`, `selected_nodes_fingerprint_sha256`) **identik**. Bila beda → ada non-determinisme, harus dibereskan (aturan repo).

---

## 10. SCOPE-GUARD (yang TIDAK boleh disentuh)

- Jangan ubah: algoritma inti TFT (pytorch-forecasting), formula water_deficit (P−PET), metode node-selection hybrid (0.7 behavior + 0.3 distance), klasifikasi SPEI (`classify_spei`), schema v2 guard.
- Jangan tambah fitur/abstraksi di luar temuan audit. Hanya minimal change sesuai ID temuan.
- Jangan hapus file mentah/data; hanya arsip.

---

## 11. TRACEABILITY (Plan ID ↔ Todo Task)

| Plan ID | Todo Task |
|---------|-----------|
| C1 | #1 |
| M2 | #2 |
| M3, M4 | #3 |
| H3 | #4 |
| H1 | #5 |
| M1 | #6 |
| M5, M6 | #7 |
| L1 | #8 |
| H2, S3 | #9 |
| L2 | #10 |
| C2 (regenerate data) | #11 |
| C2 (retrain enc90) | #12 |
| C2 (regenerate results) | #13 |
| test/build | #14 |
| Audit ulang | #15 |
| Lapor | #16 |
| S1, S2 | #17 |
| S4, S5 | #18 |
| **S2 hilir + visualize path (§2.10)** | **belum ada → tambahkan task** |
| **Backup/commit (§8)** | **implisit di #11–#13, jadikan eksplisit** |
