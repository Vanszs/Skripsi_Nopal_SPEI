# Dokumentasi Metode Penelitian Proyek Peramalan SPEI Berbasis TFT

## 1. Objek dan Lingkup Proyek

Objek penelitian pada proyek ini adalah sistem peramalan kekeringan berbasis indeks **SPEI-3** (Standardized Precipitation Evapotranspiration Index, skala 3 bulan) dengan encoder window **90 hari** (3 bulan, sesuai skala SPEI-3) dan horizon prediksi **30 hari**. Fokus utama sistem bukan sekadar memprediksi curah hujan, tetapi memodelkan kondisi keseimbangan air (surplus-defisit) yang lebih relevan untuk pemantauan kekeringan.

Secara spasial, studi dibatasi pada lima wilayah di Jawa Timur, yaitu **Bojonegoro** (-7.155, 111.88), **Lamongan** (-7.128, 112.316), **Nganjuk** (-7.604, 111.905), **Ngawi** (-7.403, 111.445), dan **Tuban** (-6.895, 112.045). Konfigurasi koordinat pusat kota disimpan dalam `data/config/city_centers.json`. Secara temporal, data yang diproses mencakup periode **2005-01-01 hingga 2026-01-01** (data harian). Batasan ini sengaja dipilih agar model dibangun pada data historis panjang, namun tetap dievaluasi pada periode modern (2024 ke atas) yang merepresentasikan penggunaan operasional.

Arsitektur spasial yang digunakan adalah **super-node per kota**. Setiap kota tidak lagi direpresentasikan oleh satu titik tunggal, tetapi melalui proses dua tahap: (1) membentuk 9 kandidat node grid di sekitar pusat kota menggunakan offset tetap (`DEFAULT_NODE_OFFSETS`), lalu (2) memilih lima node paling representatif dan mengagregasikannya menjadi satu entitas model (super-node). Pendekatan ini dipilih karena data iklim pada satu titik dapat bersifat noisy; dengan agregasi node yang secara perilaku mirip, sistem menjadi lebih robust tanpa kehilangan konteks lokal kota.

Konteks penggunaan sistem adalah **dukungan analisis risiko kekeringan** untuk pemantauan tren kelembapan-kekeringan jangka pendek (30 hari), bukan untuk pengambilan keputusan hidrologi tingkat bendungan secara langsung. Dengan demikian, output utama sistem adalah seri prediksi SPEI dan probabilitas rentang ketidakpastian (quantile P10/P50/P90), yang kemudian dapat diinterpretasikan ke kelas kondisi kering-normal-basah.

## 2. Jenis dan Sumber Data

Data yang digunakan merupakan **data sekunder time series multivariat** dengan resolusi harian. Seluruh data diperoleh dari sumber eksternal yang konsisten, yaitu API arsip meteorologi:

- Sumber: **Open-Meteo Archive API** (`https://archive-api.open-meteo.com/v1/archive`)
- Zona waktu pengambilan: `Asia/Jakarta`
- Variabel harian utama:
  - `precipitation_sum`
  - `et0_fao_evapotranspiration`
  - `soil_moisture_0_to_7cm_mean` (disimpan sebagai `soil_moisture`)
  - `temperature_2m_max`
  - `temperature_2m_min`

Konfigurasi kota diambil dari berkas lokal `data/config/city_centers.json`, kemudian sistem membangun kandidat grid node menggunakan offset spasial tetap di sekitar pusat kota. Offset yang digunakan (`DEFAULT_NODE_OFFSETS` di `src/data/ingest.py` baris 16–26) adalah:

| Indeks | Lat Offset | Lon Offset | Keterangan |
|--------|-----------|-----------|------------|
| n00 | 0.00 | 0.00 | Pusat kota |
| n01 | +0.12 | 0.00 | Utara (~13 km) |
| n02 | -0.12 | 0.00 | Selatan (~13 km) |
| n03 | 0.00 | +0.12 | Timur (~13 km) |
| n04 | 0.00 | -0.12 | Barat (~13 km) |
| n05 | +0.08 | +0.08 | Timur Laut |
| n06 | +0.08 | -0.08 | Barat Laut |
| n07 | -0.08 | +0.08 | Tenggara |
| n08 | -0.08 | -0.08 | Barat Daya |

Hasil struktur data saat ini adalah sebagai berikut:

- **Raw dataset**: `data/raw/weather_history_east_java.parquet`
  - Jumlah baris: **345.195**
  - Jumlah kota: **5**
  - Jumlah raw node: **45** (9 node per kota)
  - Periode: **2005-01-01 s.d. 2026-01-01**
  - Skema: **schema_version = 2**
- **Node selection artifact**: `data/processed/node_selection_v2.parquet`
  - Jumlah baris: **25** (5 node terpilih × 5 kota)
  - `top_k = 5`, `selection_end_date = 2022-12-31`
- **Processed dataset**: `data/processed/spei_dataset.parquet`
  - Jumlah baris: **37.460**
  - Jumlah super-node: **5** (1 per kota)
  - Periode efektif (setelah rolling SPEI): **2005-06-29 s.d. 2026-01-01**
  - Skema: **schema_version = 2**

Jenis data ini sesuai untuk peramalan deret waktu karena memiliki urutan temporal jelas, dependensi musiman, dan interaksi multivariat antar-variabel cuaca.

## 3. Tahapan Proyek (End-to-End Pipeline)

Pipeline dirancang agar **reproducible**, **leakage-safe**, dan dapat dievaluasi ulang secara konsisten. Urutan tahapan yang diterapkan adalah sebagai berikut.

### 3.1 Pengumpulan Data

Tahap ingest dimulai dari daftar pusat kota. Untuk setiap kota, sistem membentuk 9 kandidat node menggunakan pola offset tetap (`DEFAULT_NODE_OFFSETS`). Setiap node diberi identitas deterministik dan collision-safe:

1. `node_local_id` dibentuk dari indeks offset (`n00`, `n01`, dst.).
2. `node_id` dibangun dari format `"{city_id}__{node_local_id}__{coord_token}"` di mana `coord_token` adalah 8 karakter pertama SHA-1 hash dari `"{lat:.6f},{lon:.6f}"` (baris 75–78 `src/data/ingest.py`). Skema ini menjamin identitas deterministik dan collision-safe.
3. Data harian tiap node diambil dari Open-Meteo API dengan mekanisme retry: maksimum 6 percobaan (baris 202), sleep `max(1, int(Retry-After))` untuk HTTP 429, sleep `3*(attempt+1)` untuk HTTP 5xx, dan sleep `2*(attempt+1)` untuk exception umum.

Alasan desain ini adalah menjaga dua hal sekaligus: cakupan spasial memadai (lebih dari satu titik per kota) dan konsistensi identitas node agar tidak terjadi pencampuran sequence antar-node.

### 3.2 Preprocessing Data

Preprocessing dilakukan di level node terlebih dahulu, baru kemudian agregasi super-node. Urutan ini kritikal untuk mencegah leakage.

Langkah operasional:

1. **Validasi skema raw**  
   Sistem memastikan kolom wajib tersedia (`city_id`, `node_id`, `raw_node_id`, `time`, `lat`, `lon`, variabel cuaca, dsb.) dan `schema_version = 2`.

2. **Interpolasi/penanganan missing**  
   Nilai variabel cuaca diisi dengan **forward fill** per `raw_node_id`. Forward-only dipilih agar nilai masa depan tidak bocor ke masa lalu.

3. **Seleksi node berbasis kemiripan (train-only)**  
    Seleksi node dilakukan hanya pada data hingga `2022-12-31` (`SELECTION_END_DATE`, baris 12 `src/data/preprocess.py`).  
    Untuk setiap node di suatu kota:
    - Dibangun profil node dari seri 5 variabel cuaca (`WEATHER_COLS`).
    - Dibangun profil kota pembanding dengan skema **leave-one-node-out** (node yang dinilai tidak ikut rata-rata pembanding).
    - Dihitung `behavior_score` dari rerata Pearson correlation antar-variabel cuaca (`np.nanmean(corr_scores)`); jika kosong, diisi `-1.0` (baris 89–99 `src/data/preprocess.py`).
    - Dihitung `distance_score = 1/(1+jarak_km)` terhadap pusat kota menggunakan jarak haversine (baris 102–103).
    - Dibentuk `hybrid_score = 0.7*behavior_score + 0.3*distance_score` (baris 104).

4. **Pemilihan top-5 deterministik**  
    Node dipilih berdasarkan urutan stabil: `hybrid_score desc`, `behavior_score desc`, `distance_score desc`, `raw_node_id asc`. `DEFAULT_TOP_K = 5` (baris 13 `src/data/preprocess.py`). Seed dan fingerprint metadata (SHA-256) disimpan agar hasil seleksi dapat direplikasi.

5. **Agregasi menjadi super-node**  
    Lima node terpilih per kota digabung per `(city_id, time)` dengan agregasi mean untuk variabel cuaca dan atribut spasial (`lat`, `lon`, `elevation`).  
    Hasilnya diberi `super_node_id` dengan format `f"SN_{city_id}"` (baris 259 `src/data/preprocess.py`) dan `selected_node_count = 5`.

6. **Feature engineering akhir**  
     Setelah super-node terbentuk, sistem menghitung (baris 267–279 `src/data/preprocess.py`):
     - `water_deficit = precipitation_sum - et0_fao_evapotranspiration` (baris 267)
     - `SPEI_3` melalui fitting distribusi Fisk (log-logistic) per bulan kalender (baris 269)
     - `SPEI_3_diff = SPEI_3.diff().fillna(0.0)` (baris 270)
     - `time_idx = (time - time.min()).dt.days` (baris 275)
     - `month_sin = sin(2*pi*month/12)`, `month_cos = cos(2*pi*month/12)` (baris 277–278)
     - `precipitation_log = log1p(precipitation_sum)` (baris 279)

7. **Quality gate akhir**  
   Sistem memastikan tidak ada duplikasi `(super_node_id, time)`, tidak ada pelanggaran `selected_node_count`, dan membersihkan NaN/inf.

### 3.3 Pemilihan Metode/Algoritma dan Alasan

Model utama yang dipakai adalah **Temporal Fusion Transformer (TFT)** dengan loss quantile. Pemilihan ini didasarkan pada kebutuhan proyek:

- Data bersifat multivariat time series dengan dependensi jangka pendek-menengah.
- Diperlukan kemampuan menangani kombinasi fitur statis (lokasi) dan dinamis (cuaca harian).
- Diperlukan output probabilistik (quantile) untuk ketidakpastian, bukan hanya satu titik prediksi.
- TFT menyediakan interpretasi kontribusi variabel (variable importance), penting untuk konteks penelitian.

Sebagai baseline, digunakan **naive persistence**, sehingga hasil model dapat diuji apakah benar memberi nilai tambah dibanding pendekatan sangat sederhana.

### 3.4 Implementasi Sistem/Program

Implementasi dibagi modular:

- `src/data/ingest.py`: akuisisi data node-level.
- `src/data/preprocess.py`: seleksi node train-only + agregasi super-node + fitur SPEI.
- `src/models/dataset.py`: pembentukan `TimeSeriesDataSet` dengan `group_ids = super_node_id`, `MAX_ENCODER_LENGTH = 90`.
- `src/models/tft.py`: konstruksi arsitektur TFT (`output_size=3` quantile, `QuantileLoss(quantiles=[0.1, 0.5, 0.9])`).
- `src/training/train.py`: training, checkpointing, dan logging run metadata (default `max_encoder_length=90`, `hidden_size=64`, `dropout=0.20`, `attention_head_size=2`).
- `evaluate.py` / `full_evaluation.py`: evaluasi numerik, baseline comparison, visualisasi.
- `main.py`: orkestrasi ingest → preprocess → train dengan guard skema versi.

### 3.5 Proses Prediksi

Model memproduksi prediksi quantile untuk horizon 30 hari dengan 3 quantile: **P10** (quantile 0.1), **P50** (quantile 0.5, median), dan **P90** (quantile 0.9).  
Untuk evaluasi point forecast, digunakan median (`P50`, index 1 pada output tensor).  
Untuk evaluasi ketidakpastian, digunakan interval `P10-P90` (nominal 80% coverage).

### 3.6 Evaluasi dan Validasi Hasil

Evaluasi dilakukan pada periode `year >= 2024`, dengan pembagian data:

- Train: `year < 2023` (31.975 baris)
- Validasi: `year == 2023` (1.975 baris)
- Test: `year >= 2024` (3.365 baris)

Selain metrik agregat, evaluasi juga dibuat per super-node dan per kota agar perbandingan dengan baseline historis tetap konsisten.

## 4. Metode Analisis Data

### 4.1 Transformasi Hidroklimat: Water Deficit dan SPEI

Tahap analisis inti dimulai dari defisit air harian:

\[
D_t = P_t - PET_t
\]

dengan:
- \(P_t\): `precipitation_sum` pada hari ke-\(t\)
- \(PET_t\): `et0_fao_evapotranspiration` pada hari ke-\(t\)

Untuk SPEI skala \(k\) bulan (di implementasi harian, \(k \times 30\) hari), digunakan akumulasi rolling:

\[
X_t^{(k)} = \sum_{i=0}^{k \times 30 - 1} D_{t-i}
\]

Setiap bulan kalender difit ke distribusi log-logistic (Fisk), lalu diubah ke skor normal baku:

\[
SPEI_t = \Phi^{-1}(F_{\text{fisk}}(X_t^{(k)}))
\]

\(\Phi^{-1}\) adalah inverse CDF normal.  
Interpretasi: nilai negatif menunjukkan kondisi kering, positif menunjukkan kondisi basah.

### 4.2 Pemodelan Deret Waktu dengan TFT

Set data dibentuk dengan:
- `group_ids = super_node_id` (menjaga batas sequence antarkota),
- `max_encoder_length = 90` (3 bulan, sesuai skala SPEI-3),
- `max_prediction_length = 30`,
- static categorical: `["super_node_id", "city_id"]`,
- static real: `["elevation", "lat", "lon"]`,
- time-varying known real: `["time_idx", "month_sin", "month_cos"]`,
- time-varying unknown real: `["SPEI_3", "SPEI_3_diff", "water_deficit", "precipitation_log", "et0_fao_evapotranspiration", "soil_moisture", "temperature_2m_max", "temperature_2m_min"]`,
- `target_normalizer = EncoderNormalizer(transformation=None)`.

Hyperparameter default model TFT (berdasarkan `src/models/tft.py` dan `src/training/train.py`):
- `hidden_size = 64`
- `dropout = 0.20`
- `attention_head_size = 2`
- `hidden_continuous_size = 10`
- `learning_rate = 3e-4`
- `weight_decay = 1e-4`
- `gradient_clip_val = 0.5`
- `batch_size = 32`
- `EarlyStopping patience = 30`

Loss utama adalah **Quantile Loss (Pinball Loss)**:

\[
L_q(y, \hat{y}_q) = \max \left(q(y-\hat{y}_q), (q-1)(y-\hat{y}_q)\right)
\]

Loss total adalah agregasi untuk seluruh quantile yang dipakai (`[0.1, 0.5, 0.9]`).

Konsekuensi metodologisnya:
- model tidak hanya mempelajari nilai tengah, tetapi distribusi prediksi pada tiga titik kuantil (P10, P50, P90),
- interval P10-P90 memberikan estimasi ketidakpastian dengan nominal 80% coverage,
- hasil dapat digunakan untuk risk-aware decision (misalnya interval ketidakpastian).

### 4.3 Cara Output Dihasilkan dan Digunakan

Output model digunakan dalam dua jalur:

1. **Point forecast**: ambil `P50` untuk RMSE/MAE/R² dan analisis error.
2. **Uncertainty forecast**: gunakan `P10-P90` untuk coverage (PICP).

Secara aplikasi, output ini dapat dipakai untuk:
- menilai tren kekeringan per kota 30 hari ke depan,
- membandingkan performa antar-wilayah,
- mendukung interpretasi variabel dominan melalui mekanisme interpretasi TFT.

## 5. Validasi dan Evaluasi

### 5.1 Metode Evaluasi

Metrik utama yang digunakan:

1. **RMSE**
\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]
Menekankan penalti error besar.

2. **MAE**
\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]
Lebih robust terhadap outlier.

3. **R²**, **Bias**, dan **Pearson r**  
Digunakan untuk menilai kekuatan hubungan, arah deviasi sistematis, dan kualitas penjelasan variansi.

4. **PICP (Prediction Interval Coverage Probability)**  
Persentase observasi aktual yang jatuh di antara `P10` dan `P90` (nominal target 80%).

5. **Naive persistence baseline**  
Digunakan untuk menguji apakah model benar-benar lebih baik dari pendekatan sederhana.

### 5.2 Strategi Validasi

Validasi dilakukan dengan prinsip berikut:

- **Chronological split**: tidak ada random split untuk mencegah kebocoran waktu.
- **Train-only selection**: pemilihan node super-node hanya memakai data hingga cutoff training.
- **Entity-safe grouping**: semua sequence dibatasi oleh `super_node_id`.
- **Schema guard**: pipeline menolak data stale/skema lama.

Dengan desain ini, hasil evaluasi lebih representatif terhadap kondisi prediksi nyata (future unseen period).

### 5.3 Kriteria Keberhasilan

Model dinyatakan berhasil secara operasional jika memenuhi kondisi berikut:

1. Metrik error utama tidak lebih buruk dari baseline naive pada mayoritas horizon.
2. PICP berada dekat nominal 80% (tidak under-coverage ekstrem).
3. Tidak ada pelanggaran integritas data (duplikasi key, mismatch skema, pelanggaran top-k node).
4. Hasil konsisten antar-run dengan konfigurasi yang sama (reproducibility via seed + fingerprint artifact).

### 5.4 Ringkasan Hasil Evaluasi Terkini (Snapshot Proyek)

> **Catatan**: Hasil evaluasi berikut berasal dari checkpoint konfigurasi sebelumnya (`enc30`, 7 quantile). Setelah perubahan konfigurasi ke `encoder=90`, 3 quantile (`[0.1, 0.5, 0.9]`), diperlukan retraining untuk mendapatkan hasil evaluasi baru.

Checkpoint: `enc30-run20260422_034701-epoch=5-val_loss=0.1845.ckpt`

- RMSE: **0,1649**
- MAE: **0,1052**
- R²: **0,9581**
- Pearson r: **0,9802**
- Bias: **0,0375**
- PICP (P10–P90): **0,8808**
- Naive RMSE: **0,1675**
- Horizon win terhadap naive: **30/30** (berdasarkan `full_evaluation` terbaru)

Per-kota (model raw):

| Kota | RMSE | MAE | R² | Bias | Pearson r | PICP |
|------|------|-----|----|------|-----------|------|
| Bojonegoro | 0,1751 | 0,1145 | 0,9559 | 0,0423 | 0,9798 | 0,8648 |
| Lamongan | 0,1282 | 0,0863 | 0,9582 | 0,0390 | 0,9809 | 0,8945 |
| Nganjuk | 0,2067 | 0,1271 | 0,9462 | 0,0373 | 0,9739 | 0,8886 |
| Ngawi | 0,1711 | 0,1135 | 0,9560 | 0,0296 | 0,9793 | 0,8960 |
| Tuban | 0,1296 | 0,0845 | 0,9296 | 0,0393 | 0,9675 | 0,8603 |

Hasil ini menunjukkan model sudah memberikan peningkatan dibanding baseline persistence pada konfigurasi sebelumnya, dengan kalibrasi interval yang masih dapat diterima untuk penggunaan analitik.

---

## Catatan Asumsi Operasional

Dokumentasi ini menggunakan asumsi yang konsisten dengan implementasi saat ini:

1. Resolusi SPEI dihitung dari data harian dengan pendekatan skala bulan \(\approx 30\) hari.
2. Fokus prediksi adalah SPEI-3 (satu-satunya target dan indeks yang digunakan).
3. Sistem ditujukan untuk analisis risiko kekeringan jangka pendek 30 hari, bukan simulasi hidrologi proses-fisik penuh.
