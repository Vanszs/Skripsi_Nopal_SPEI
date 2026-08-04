# ALL-IN-ONE AUDIT REPORT
## Draft Nopal1 (PDF) vs Implementation Code

**Date:** 2026-07-11  
**Scope:** BAB 1–3 dari `Draft Nopal1.pdf` dibandingkan dengan kode di `src/`, skrip pelatihan, dan evaluasi  
**Sources:**
- `Draft Nopal1.pdf` (Extracted pages 1–68)
- Repository code: `src/`, `evaluate.py`, `full_evaluation.py`, `run_evaluation.py`, `run_experiment.py`
- Evaluation artifacts: `docs/THESIS_READINESS_REPORT.md`

---

## EXECUTIVE SUMMARY

**Verdict: NEEDS_MAJOR_REVISIONS**

Secara keseluruhan, repositori kode telah memiliki pipeline yang sangat lengkap (data ingestion, preprocessing, komputasi SPEI, dataset generator, model building, training, evaluasi, hingga kalibrasi). Namun, draf tulisan skripsi (terutama BAB I, II, dan III) masih memiliki ketidakselarasan dengan implementasi kode.

Isu terbesar adalah **ketiadaan penjelasan matematis dan metodologis untuk Conformal-style Calibration** pada Bab III, padahal metode ini diajukan sebagai salah satu tujuan khusus penelitian dan diimplementasikan secara aktif di kode. Selain itu, terdapat ketidaksesuaian terminologi klasifikasi tingkat kekeringan SPEI dan nama variabel di dataset.

### Issue Count

| Level | Count |
|---|---|
| FATAL | 0 |
| MAJOR | 4 |
| MINOR | 4 |

---

## SECTION 1: ISSUES IN PDF VS CODE

### 1.1 FATAL
*Tidak ditemukan isu dengan kategori FATAL yang menghentikan jalannya program atau kontradiksi arsitektur tingkat tinggi.*

---

### 1.2 MAJOR

#### 1.2.1 Conformal-style Calibration: Missing from Bab III (Methodology)

**Lokasi PDF:** BAB 1 (Rumusan Masalah 2, Tujuan Khusus 2), BAB 2 (2.2.5.3 Pentingnya Kalibrasi), BAB 3 (hanya disebut sekali secara umum di awal bab).

**Klaim PDF:** Menjanjikan penerapan "kalibrasi per lokasi" untuk meningkatkan keandalan prediksi probabilistik dan menangani heterogenitas spasial.

**Bukti Kode:**
Implementasi sesungguhnya menggunakan *conformal-style per-city multiplicative interval calibration* yang memetakan bias deviasi pada dataset validasi (tahun 2023) untuk meluaskan/menyempitkan rentang prediksi P10-P90 secara dinamis:
```python
# src/evaluation/calibration.py
def fit_per_city_interval_calibration(df_val: pd.DataFrame, city_col: str = "city_id", nominal: float = 0.80) -> dict:
    ...
    abs_norm_resid = np.abs(grp["actual"].values[mask] - center.values[mask]) / half_width.values[mask]
    factor = float(np.quantile(abs_norm_resid, nominal))
    factors[str(city)] = max(factor, 0.5)  # floor 0.5: avoid extreme interval shrinkage (C3)
    return factors
```

**Analisis:** Conformal-style calibration ini merupakan kontribusi metodologis penting untuk meluruskan coverage probabilistik (target PICP 80%). Namun, langkah-langkah, rumus matematika, dan justifikasi penggunaan Conformal interval scaling ini sama sekali tidak dijelaskan pada Bab III (Metode Penelitian).

**Perbaikan:** Tambahkan subbab khusus di Bab III mengenai metode dan formula matematika Conformal Interval Calibration per lokasi (menjelaskan scaling factor, penggunaan data validasi 2023, serta aplikasinya pada data uji).

---

#### 1.2.2 SPEI Classification: Mismatch in Categories and Thresholds

**Lokasi PDF:** BAB III, Tabel 3.3 (Kategori Indeks SPEI)

**Klaim PDF:**
- `< -2.0`: Sangat Kering
- `-1.5 hingga -1.99`: Kering Berat
- `-1.0 hingga -1.49`: Kering Sedang
- `-0.99 hingga 0.99`: Normal (Tidak ada kategori Kering/Basah Ringan)

**Bukti Kode:**
Di dalam `src/data/spei.py` fungsi `classify_spei` mengimplementasikan kategori standar McKee et al. (1993) / WMO yang berbeda secara terminologi dan pembagian kelas:
```python
# src/data/spei.py
def classify_spei(value):
    if value <= -2.0: return "Kekeringan Ekstrem"
    elif value <= -1.5: return "Kekeringan Parah"
    elif value <= -1.0: return "Kekeringan Sedang"
    elif value < -0.5: return "Kekeringan Ringan"
    elif value <= 0.5: return "Normal"
    ...
```

**Analisis:** Ketidaksesuaian kategori ini membingungkan pembaca. Sebagai contoh, nilai `-0.7` akan diklasifikasikan sebagai "Kekeringan Ringan" oleh kode, namun masuk sebagai "Normal" di dokumen skripsi. Selain itu, istilah "Sangat Kering" di dokumen bertentangan dengan istilah "Kekeringan Ekstrem" di kode.

**Perbaikan:** Selaraskan Tabel 3.3 pada Bab III agar menggunakan terminologi dan rentang ambang batas yang sama persis dengan fungsi `classify_spei` di kode (menyertakan Kekeringan Ringan dan menggunakan istilah Ekstrem/Parah/Sedang/Ringan).

---

#### 1.2.3 Web Dashboard Deliverable: Missing Frontend / FastAPI Implementation

**Lokasi PDF:** BAB I (Rumusan Masalah 5, Tujuan Khusus 5, Manfaat 5), BAB III (Tahap Evaluasi & Desain)

**Klaim PDF:** Perancangan dan implementasi website dashboard untuk menyajikan hasil peramalan SPEI-3 secara visual dan interaktif.

**Bukti Kode:**
- Tidak terdapat folder frontend/dashboard (seperti Vue.js/React) maupun backend API (FastAPI/Flask) di dalam repositori.
- Pipeline eksperimen berakhir pada pembuatan berkas visualisasi statis `.png` (seperti `results/variable_importance.png`, `results/attention_weights.png`).

**Analisis:** Klaim pembuatan website dashboard di Bab I dan III bersifat spekulatif karena implementasinya belum ada di dalam codebase saat ini. Dosen penguji dapat mempertanyakan keberadaan dashboard ini saat sidang/evaluasi.

**Perbaikan:**
- Jika dashboard akan dibuat, segera tambahkan folder implementasi dasar API FastAPI dan Web UI Vue.js.
- Jika tidak diimplementasikan, hapus Rumusan Masalah 5 dan Tujuan 5 dari Bab I, dan nyatakan visualisasi interaktif sebagai **pekerjaan masa depan (future work)** di bab kesimpulan.

---

### 1.3 MINOR

#### 1.3.1 Soil Moisture Variable Naming

**Lokasi PDF:** BAB III, Tabel 3.2 (Parameter Penelitian)

**Klaim PDF:** Nama parameter pada dataset untuk Kelembaban Tanah Lapisan Atas adalah `soil_moisture_0_to_7cm_mean`.

**Bukti Kode:**
Di dalam `src/data/ingest.py` data diambil menggunakan parameter tersebut dari Open-Meteo, namun langsung disimpan/diubah namanya menjadi `soil_moisture`:
```python
# src/data/preprocess.py
WEATHER_COLS = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "soil_moisture",  # Bukan soil_moisture_0_to_7cm_mean
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "shortwave_radiation_sum",
    "wind_speed_10m_mean",
]
```

**Analisis:** Perbedaan nama variabel ini minor, namun menyebabkan kebingungan bagi pembaca yang mencoba mereproduksi dataset menggunakan kode pra-pemrosesan.

**Perbaikan:** Ubah nama kolom target di Tabel 3.2 kolom "Nama Parameter pada Dataset" dari `soil_moisture_0_to_7cm_mean` menjadi `soil_moisture` (dengan catatan bahwa data asli di-ingest dari Open-Meteo API menggunakan nama `soil_moisture_0_to_7cm_mean`).

---

#### 1.3.2 Omission of Key TFT Hyperparameters in Table 3.4

**Lokasi PDF:** BAB III, Tabel 3.4 (Konfigurasi Model Temporal Fusion Transformer)

**Klaim PDF:** Tabel hanya menampilkan parameter umum (Encoder Length = 90 hari, Prediction Length = 30 hari, Quantiles = P10, P50, P90).

**Bukti Kode:**
Model dilatih menggunakan hyperparameter spesifik yang penting untuk performa model TFT:
```python
# src/training/train.py
    hidden_size=48,
    dropout=0.40,
    attention_head_size=1,
    hidden_continuous_size=8,
    learning_rate=3e-4,
    weight_decay=1e-4,
    gradient_clip_val=0.5,
```

**Analisis:** Omitnya nilai hyperparameter seperti `hidden_size`, `dropout`, dan `learning_rate` di Bab III membuat rancangan model kurang transparan untuk direproduksi.

**Perbaikan:** Tambahkan baris di Tabel 3.4 untuk hyperparameter model TFT: Hidden Size (48), Dropout (0.40), Attention Heads (1), Learning Rate (0.0003), Weight Decay (1e-4), dan Gradient Clip (0.5).

---

#### 1.3.3 FAO Penman-Monteith Formulation Details Missing

**Lokasi PDF:** BAB III (3.3.3 Komputasi Indeks SPEI)

**Klaim PDF:** Menyatakan evapotranspirasi potensial (PET) harian menggunakan metode FAO Penman-Monteith.

**Bukti Kode:**
Kode langsung mengambil kolom `et0_fao_evapotranspiration` dari Open-Meteo API (yang sudah dihitung berdasarkan standar FAO Penman-Monteith oleh Open-Meteo menggunakan data radiasi, suhu, kelembaban, dan kecepatan angin):
```python
# src/data/spei.py
def calculate_water_deficit(df):
    """
    D = P - PET (Water Deficit)
    ET0 (FAO Penman-Monteith) is used as PET proxy.
    """
    return df["precipitation_sum"] - df["et0_fao_evapotranspiration"]
```

**Analisis:** Dokumen menulis seolah-olah proses perhitungan FAO Penman-Monteith dilakukan secara manual di dalam kode/pipeline, padahal nilai tersebut merupakan data *ready-to-use* dari Open-Meteo API.

**Perbaikan:** Berikan klarifikasi pada Bab III bahwa perhitungan ET0 (FAO Penman-Monteith) disediakan langsung oleh Open-Meteo Archive API, bukan dihitung secara manual baris demi baris di kode skripsi.

---

## SECTION 2: VERIFICATION STATUS

| Aspek | Status di Kode | Keterangan |
|---|---|---|
| Spasial Super-node | ✅ VALID | Agregasi data cuaca dari 5 node spasial terdekat per kota berjalan benar di `preprocess.py`. |
| Chronological Split | ✅ VALID | Pemisahan tahun <2023 (Train), 2023 (Val), >=2024 (Test) terimplementasi dengan benar. |
| SPEI-3 Computation | ✅ VALID | Menggunakan distribusi Fisk (Log-logistic) per bulan kalender. Extrapolasi tail-end diimplementasikan untuk stabilitas z-score. |
| TFT Quantile Output | ✅ VALID | Model menghasilkan output kuantil P10, P50, dan P90 dengan loss function Quantile Loss. |
| Performance Metrics | ✅ VALID | Penghitungan RMSE, MAE, R², Pearson r, dan PICP diimplementasikan dengan benar pada dataset uji di `full_evaluation.py`. |
