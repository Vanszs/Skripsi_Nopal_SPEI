# Dokumentasi Metode Penelitian Proyek Peramalan SPEI Berbasis TFT

## 1. Objek dan Lingkup Proyek

Objek penelitian pada proyek ini adalah sistem peramalan kekeringan berbasis indeks **SPEI-3** (Standardized Precipitation Evapotranspiration Index, skala 3 bulan) dengan horizon prediksi **30 hari**. Fokus utama sistem bukan sekadar memprediksi curah hujan, tetapi memodelkan kondisi keseimbangan air (surplus-defisit) yang lebih relevan untuk pemantauan kekeringan.

Secara spasial, studi dibatasi pada lima wilayah di Jawa Timur, yaitu **Bojonegoro, Lamongan, Nganjuk, Ngawi, dan Tuban**. Secara temporal, data yang diproses saat ini mencakup periode **2005-01-01 hingga 2026-01-01** (data harian). Batasan ini sengaja dipilih agar model dibangun pada data historis panjang, namun tetap dievaluasi pada periode modern (2024 ke atas) yang merepresentasikan penggunaan operasional.

Arsitektur spasial yang digunakan adalah **super-node per kota**. Setiap kota tidak lagi direpresentasikan oleh satu titik tunggal, tetapi melalui proses dua tahap: (1) membentuk kandidat node grid di sekitar pusat kota, lalu (2) memilih lima node paling representatif dan mengagregasikannya menjadi satu entitas model (super-node). Pendekatan ini dipilih karena data iklim pada satu titik dapat bersifat noisy; dengan agregasi node yang secara perilaku mirip, sistem menjadi lebih robust tanpa kehilangan konteks lokal kota.

Konteks penggunaan sistem adalah **dukungan analisis risiko kekeringan** untuk pemantauan tren kelembapan-kekeringan jangka pendek (30 hari), bukan untuk pengambilan keputusan hidrologi tingkat bendungan secara langsung. Dengan demikian, output utama sistem adalah seri prediksi SPEI dan probabilitas rentang ketidakpastian (quantile), yang kemudian dapat diinterpretasikan ke kelas kondisi kering-normal-basah.

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

Konfigurasi kota diambil dari berkas lokal `data/config/city_centers.json`, kemudian sistem membangun kandidat grid node menggunakan offset spasial tetap di sekitar pusat kota. Hasil struktur data saat ini adalah sebagai berikut:

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
2. `node_id` dibangun dari kombinasi `city_id + node_local_id + hash(lat,lon)`.
3. Data harian tiap node diambil dari Open-Meteo API dengan mekanisme retry terhadap error jaringan, rate limit (HTTP 429), dan server error (5xx).

Alasan desain ini adalah menjaga dua hal sekaligus: cakupan spasial memadai (lebih dari satu titik per kota) dan konsistensi identitas node agar tidak terjadi pencampuran sequence antar-node.

### 3.2 Preprocessing Data

Preprocessing dilakukan di level node terlebih dahulu, baru kemudian agregasi super-node. Urutan ini kritikal untuk mencegah leakage.

Langkah operasional:

1. **Validasi skema raw**  
   Sistem memastikan kolom wajib tersedia (`city_id`, `node_id`, `raw_node_id`, `time`, `lat`, `lon`, variabel cuaca, dsb.) dan `schema_version = 2`.

2. **Interpolasi/penanganan missing**  
   Nilai variabel cuaca diisi dengan **forward fill** per `raw_node_id`. Forward-only dipilih agar nilai masa depan tidak bocor ke masa lalu.

3. **Seleksi node berbasis kemiripan (train-only)**  
   Seleksi node dilakukan hanya pada data hingga `2022-12-31` (`selection_end_date`).  
   Untuk setiap node di suatu kota:
   - Dibangun profil node dari seri variabel cuaca.
   - Dibangun profil kota pembanding dengan skema **leave-one-node-out** (node yang dinilai tidak ikut rata-rata pembanding).
   - Dihitung `behavior_score` dari rerata korelasi antar-variabel cuaca.
   - Dihitung `distance_score = 1/(1+jarak_km)` terhadap pusat kota.
   - Dibentuk `hybrid_score = 0.7*behavior_score + 0.3*distance_score`.

4. **Pemilihan top-5 deterministik**  
   Node dipilih berdasarkan urutan stabil: `hybrid_score desc`, `behavior_score desc`, `distance_score desc`, `raw_node_id asc`.  
   Seed dan fingerprint metadata disimpan agar hasil seleksi dapat direplikasi.

5. **Agregasi menjadi super-node**  
   Lima node terpilih per kota digabung per `(city_id, time)` dengan agregasi mean untuk variabel cuaca dan atribut spasial (`lat`, `lon`, `elevation`).  
   Hasilnya diberi `super_node_id = SN_<city_id>` dan `selected_node_count = 5`.

6. **Feature engineering akhir**  
   Setelah super-node terbentuk, sistem menghitung:
   - `water_deficit`
   - `SPEI_3`, `SPEI_6`
   - `SPEI_3_diff`
   - `time_idx`, `month_sin`, `month_cos`
   - `precipitation_log`

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
- `src/models/dataset.py`: pembentukan `TimeSeriesDataSet` dengan `group_ids = super_node_id`.
- `src/models/tft.py`: konstruksi arsitektur TFT (`output_size=7` quantile).
- `src/training/train.py`: training, checkpointing, dan logging run metadata.
- `evaluate.py` / `full_evaluation.py`: evaluasi numerik, baseline comparison, visualisasi.
- `main.py`: orkestrasi ingest → preprocess → train dengan guard skema versi.

### 3.5 Proses Prediksi

Model memproduksi prediksi quantile untuk horizon 30 hari (`P02, P10, P25, P50, P75, P90, P98`).  
Untuk evaluasi point forecast, digunakan median (`P50`).  
Untuk evaluasi ketidakpastian, digunakan interval `P10-P90` (nominal 80% coverage).

### 3.6 Evaluasi dan Validasi Hasil

Evaluasi dilakukan pada periode `year >= 2024`, dengan pembagian data:

- Train: `year < 2023` (31.975 baris)
- Validasi: `year == 2023` (1.825 baris)
- Test: `year >= 2024` (3.660 baris)

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
- `max_encoder_length = 30`,
- `max_prediction_length = 30`,
- fitur statis: `city_id`, `super_node_id`, `elevation`, `lat`, `lon`,
- fitur dinamis known/unknown sesuai skema dataset.

Loss utama adalah **Quantile Loss (Pinball Loss)**:

\[
L_q(y, \hat{y}_q) = \max \left(q(y-\hat{y}_q), (q-1)(y-\hat{y}_q)\right)
\]

Loss total adalah agregasi untuk seluruh quantile yang dipakai (`[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]`).

Konsekuensi metodologisnya:
- model tidak hanya mempelajari nilai tengah, tetapi distribusi prediksi,
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

Berdasarkan artefak evaluasi terbaru (`results/evaluation_metrics_detailed.json`) pada checkpoint:
`enc30-run20260422_034701-epoch=5-val_loss=0.1845.ckpt`, diperoleh:

- RMSE: **0,1649**
- MAE: **0,1052**
- R²: **0,9581**
- Pearson r: **0,9802**
- Bias: **0,0375**
- PICP (P10–P90): **0,8808**
- Naive RMSE: **0,1675**
- Horizon win terhadap naive: **30/30** (berdasarkan `full_evaluation` terbaru)

Hasil ini menunjukkan model sudah memberikan peningkatan dibanding baseline persistence, dengan kalibrasi interval yang masih dapat diterima untuk penggunaan analitik.

---

## Catatan Asumsi Operasional

Dokumentasi ini menggunakan asumsi yang konsisten dengan implementasi saat ini:

1. Resolusi SPEI dihitung dari data harian dengan pendekatan skala bulan \(\approx 30\) hari.
2. Fokus prediksi adalah SPEI-3 (target utama), sementara SPEI-6 dipakai sebagai fitur pendukung.
3. Sistem ditujukan untuk analisis risiko kekeringan jangka pendek 30 hari, bukan simulasi hidrologi proses-fisik penuh.
