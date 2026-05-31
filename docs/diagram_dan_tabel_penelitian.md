# Diagram dan Tabel Penelitian
## Sistem Peramalan Kekeringan Berbasis SPEI Menggunakan Temporal Fusion Transformer

---

# Bagian 1: Diagram (PlantUML)

---

## 1. Diagram Alur Penelitian

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam activityBackgroundColor #E8F4FD
skinparam activityBorderColor #2C3E50
skinparam arrowColor #2C3E50
skinparam noteBorderColor #7F8C8D
skinparam defaultFontName Arial
skinparam defaultFontSize 12

title **Diagram Alur Penelitian Peramalan Kekeringan SPEI\nMenggunakan Temporal Fusion Transformer**

start

:== **Tahap 1: Pengumpulan Data**
Akuisisi data iklim harian
dari Open-Meteo Archive API
(2005-01-01 s.d. 2025-12-31)
untuk N kabupaten (config-driven) di Jawa Timur;
note right
  Variabel: presipitasi, ETâ‚€,
  kelembaban tanah, suhu maks/min,
  elevasi
end note

:== **Tahap 2: Preprocessing Data**
Interpolasi linier per lokasi,
transformasi logaritmik presipitasi,
rekayasa fitur temporal
(month_sin, month_cos);

:== **Tahap 3: Komputasi Indeks SPEI**
Perhitungan defisit air (P âˆ’ ETâ‚€),
akumulasi rolling window,
fitting distribusi Log-Logistic (fisk),
standardisasi Z-score
â†’ SPEI-3;
note right
  Distribusi: Log-Logistic
  Kalibrasi: per bulan kalender
  Window: 90 hari (SPEI-3)
end note

:== **Tahap 4: Pembentukan Dataset**
Konstruksi TimeSeriesDataSet
dengan encoder length = 90 hari,
prediction horizon = 30 hari,
tanpa partial window;
note right
  min_encoder = max_encoder
  min_prediction = max_prediction
  GroupNormalizer (tanpa transformasi)
end note

:== **Tahap 5: Pembagian Dataset**
Training (< 2023),
Validation (2023),
Testing (â‰¥ 2024);
note right
  Scaler hanya di-fit
  pada data training
  â†’ mencegah kebocoran data
end note

:== **Tahap 6: Perancangan Model TFT**
Konfigurasi arsitektur:
hidden_size=48, attention_head=1,
dropout=0.35, 7 output kuantil;

:== **Tahap 7: Proses Training**
Pelatihan model dengan
QuantileLoss, early stopping
(patience=25), gradient clipping=0.1,
learning rate=3Ã—10â»â´;
note right
  Optimizer: Adam
  Batch size: 32
  Max epochs: 60
  GPU: RTX 3050
end note

:== **Tahap 8: Peramalan Multi-Horizon**
Prediksi SPEI-3 untuk 30 hari
ke depan secara simultan
dengan dekoding step-0-only;
note right
  Setiap time_idx hanya
  menggunakan prediksi
  pada step=0 (konteks
  encoder terbaru)
end note

:== **Tahap 9: Evaluasi Model**
Perhitungan metrik regresi
(RMSE, MAE, RÂ², Pearson r, Bias),
metrik probabilistik (PICP),
perbandingan baseline naif;
note right
  Naive persistence:
  SPEI(t) â‰ˆ SPEI(tâˆ’1)
  PICP nominal: 80%
  (interval P10â€“P90)
end note

:== **Tahap 10: Analisis Hasil**
Analisis per lokasi,
variable importance (VSN),
attention weight temporal,
deteksi kejadian kekeringan;

stop

@enduml
```

---

## 2. Diagram Desain Sistem Peramalan SPEI Berbasis TFT

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam componentBackgroundColor #E8F4FD
skinparam componentBorderColor #2C3E50
skinparam databaseBackgroundColor #FFF3CD
skinparam databaseBorderColor #856404
skinparam packageBackgroundColor #F8F9FA
skinparam packageBorderColor #6C757D
skinparam arrowColor #2C3E50
skinparam defaultFontName Arial
skinparam defaultFontSize 11

title **Desain Sistem Peramalan SPEI\nBerbasis Temporal Fusion Transformer**

' === Sumber Data ===
package "Sumber Data Eksternal" as src {
  component [Open-Meteo\nArchive API] as api
  note bottom of api
    Data iklim harian
    2005â€“2025
    N kabupaten (config-driven) Jawa Timur
  end note
}

' === Modul Akuisisi & Preprocessing ===
package "Modul Akuisisi & Preprocessing" as preproc {
  component [Ingestion\n(ingest.py)] as ingest
  database "Raw Data\n(.parquet)" as raw_db
  component [Preprocessing\n(preprocess.py)] as preprocess
  component [Komputasi SPEI\n(spei.py)] as spei
  database "Processed Data\n(.parquet)" as proc_db
}

' === Input Data Multivariat ===
package "Input Data Multivariat" as inputs {
  component [Variabel Iklim\nprecipitation_log, ETâ‚€,\nsoil_moisture, temp_max,\ntemp_min, water_deficit] as climate_var
  component [Variabel Temporal\ntime_idx, month_sin,\nmonth_cos] as temporal_var
  component [Variabel Spasial\nlocation_id (kategorikal),\nelevation (kontinu)] as spatial_var
  component [Variabel Target\nSPEI-3] as target_var
}

' === Modul Pembentukan Dataset ===
package "Modul Pembentukan Dataset" as ds_module {
  component [TimeSeriesDataSet\nBuilder (dataset.py)] as ts_builder
  note bottom of ts_builder
    encoder = 90 hari
    prediction = 30 hari
    tanpa partial window
    GroupNormalizer
  end note
  component [Pembagian Data\nTrain/Val/Test] as data_split
}

' === Model TFT ===
package "Model Temporal Fusion Transformer" as tft_pkg {
  component [Variable Selection\nNetwork (VSN)] as vsn
  component [Gated Residual\nNetwork (GRN)] as grn
  component [Multi-Head\nAttention] as attn
  component [Quantile\nRegression Output] as quantile
  note bottom of quantile
    7 kuantil:
    [0.02, 0.1, 0.25,
     0.5, 0.75, 0.9, 0.98]
  end note
}

' === Modul Evaluasi ===
package "Modul Evaluasi Kinerja" as eval_pkg {
  component [Dekoding Step-0-Only\n(evaluate.py)] as step0
  component [Metrik Regresi\nRMSE, MAE, RÂ²,\nPearson r, Bias] as reg_metrics
  component [Metrik Probabilistik\nPICP (P10â€“P90)] as prob_metrics
  component [Baseline Naif\nSPEI(t) â‰ˆ SPEI(tâˆ’1)] as naive
  component [Analisis Interpretasi\nVariable Importance,\nAttention Weights] as interpret
}

' === Output ===
package "Output Sistem" as output_pkg {
  database "Hasil Prediksi\n(.csv)" as pred_out
  component [Visualisasi &\nLaporan Evaluasi] as viz
}

' === Koneksi ===
api --> ingest
ingest --> raw_db
raw_db --> preprocess
preprocess --> spei
spei --> proc_db

proc_db --> climate_var
proc_db --> temporal_var
proc_db --> spatial_var
proc_db --> target_var

climate_var --> ts_builder
temporal_var --> ts_builder
spatial_var --> ts_builder
target_var --> ts_builder

ts_builder --> data_split

data_split --> vsn : Training &\nValidation Set
vsn --> grn
grn --> attn
attn --> quantile

quantile --> step0 : Raw Predictions
data_split --> step0 : Test Set

step0 --> reg_metrics
step0 --> prob_metrics
step0 --> naive
step0 --> interpret

reg_metrics --> pred_out
prob_metrics --> pred_out
naive --> pred_out
interpret --> pred_out

pred_out --> viz

@enduml
```

---

## 3. Diagram Mekanisme Peramalan Multi-Horizon pada Model TFT

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam defaultFontName Arial
skinparam defaultFontSize 11
skinparam rectangleBackgroundColor #E8F4FD
skinparam rectangleBorderColor #2C3E50
skinparam arrowColor #2C3E50

title **Mekanisme Peramalan Multi-Horizon\npada Model Temporal Fusion Transformer**

' === Timeline ===
rectangle "**Timeline Input-Output**" as timeline {
  rectangle "                        " as spacer #white;line:white

  ' Encoder window
  rectangle "**Historical Input Window (Encoder)**\ntâˆ’89  tâˆ’88  tâˆ’87  ...  tâˆ’2   tâˆ’1   tâ‚€\nâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ 90 hari â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€" as encoder #D5E8D4 {
  }

  note bottom of encoder
    **Input Encoder (90 timestep):**
    â”€â”€ Time-Varying Unknown â”€â”€
    â€¢ SPEI-3
    â€¢ water_deficit
    â€¢ precipitation_log
    â€¢ et0_fao_evapotranspiration
    â€¢ soil_moisture
    â€¢ temperature_2m_max, temperature_2m_min
    â”€â”€ Time-Varying Known â”€â”€
    â€¢ time_idx, month_sin, month_cos
    â”€â”€ Static â”€â”€
    â€¢ location_id, elevation
  end note

  ' Decoder window
  rectangle "**Prediction Window (Decoder)**\nt+1   t+2   t+3  ...  t+29  t+30\nâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ 30 hari â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€" as decoder #FFF2CC {
  }

  note bottom of decoder
    **Input Decoder (30 timestep):**
    â”€â”€ Time-Varying Known saja â”€â”€
    â€¢ time_idx (diketahui di masa depan)
    â€¢ month_sin (diturunkan dari kalender)
    â€¢ month_cos (diturunkan dari kalender)
    Variabel unknown TIDAK tersedia
    di window decoder
  end note
}

encoder -right-> decoder : Transisi\nEncoderâ†’Decoder

' === Arsitektur Internal ===
rectangle "**Arsitektur Internal TFT**" as arch {

  rectangle "Variable Selection Network\n(memilih fitur relevan\nper timestep)" as vsn #E8DAEF

  rectangle "LSTM Encoder\n(merangkum pola historis\nmenjadi hidden state)" as lstm_enc #D5E8D4

  rectangle "LSTM Decoder\n(memproses input\ntime-varying known\ndi masa depan)" as lstm_dec #FFF2CC

  rectangle "Interpretable Multi-Head\nAttention (1 head)\n(membobot kontribusi\nsetiap timestep historis)" as mha #FADBD8

  rectangle "Gated Residual Network\n(transformasi non-linear\ndengan skip connection)" as grn #D6EAF8

  rectangle "Quantile Output Layer\n(menghasilkan 7 kuantil\nper horizon)" as qout #F9E79F
}

vsn --> lstm_enc
vsn --> lstm_dec
lstm_enc --> mha : Hidden\nStates
lstm_dec --> mha : Query
mha --> grn
grn --> qout

' === Output Multi-Horizon ===
rectangle "**Output Prediksi Multi-Horizon**" as output {

  rectangle "t+1: P02, P10, P25, **P50**, P75, P90, P98" as h1 #E8F8F5
  rectangle "t+2: P02, P10, P25, **P50**, P75, P90, P98" as h2 #E8F8F5
  rectangle "t+3: P02, P10, P25, **P50**, P75, P90, P98" as h3 #E8F8F5
  rectangle "  ...  " as hdots #F8F9FA
  rectangle "t+29: P02, P10, P25, **P50**, P75, P90, P98" as h29 #E8F8F5
  rectangle "t+30: P02, P10, P25, **P50**, P75, P90, P98" as h30 #E8F8F5
}

qout --> h1
qout --> h2
qout --> h3
qout --> hdots
qout --> h29
qout --> h30

' === Step-0-Only ===
rectangle "**Dekoding Step-0-Only**\nUntuk setiap time_idx,\nhanya prediksi pada step=0\nyang digunakan (konteks terbaru)" as step0 #FDEDEC

h1 --> step0 : step=0\n**(digunakan)**
h2 -[dashed]-> step0 : step=1\n(diabaikan)
h3 -[dashed]-> step0 : step=2\n(diabaikan)

note bottom of step0
  **Alasan Step-0-Only:**
  Menghindari bias pemulusan temporal
  yang muncul dari rata-rata prediksi
  multi-window dengan konteks encoder
  yang berbeda usia
end note

@enduml
```

---

## 4. Diagram Proses Evaluasi Prediksi Probabilistik (PICP)

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam defaultFontName Arial
skinparam defaultFontSize 11
skinparam activityBackgroundColor #E8F4FD
skinparam activityBorderColor #2C3E50
skinparam arrowColor #2C3E50

title **Proses Evaluasi Prediksi Probabilistik\nPrediction Interval Coverage Probability (PICP)**

start

:== **Input: Raw Prediction dari Model TFT**
Model menghasilkan 7 output kuantil
per timestep per lokasi:
[P02, P10, P25, P50, P75, P90, P98];
note right
  QuantileLoss dengan kuantil:
  [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
  P50 = prediksi median (titik)
end note

:== **Dekoding Step-0-Only**
Untuk setiap time_idx, ambil HANYA
prediksi pada forecast step=0
(konteks encoder paling mutakhir);

:== **Ekstraksi Interval Prediksi**
Batas bawah: P10 (kuantil ke-10)
Batas atas: P90 (kuantil ke-90)
â†’ interval prediksi 80%;

:== **Pencocokan dengan Nilai Aktual**
Untuk setiap sampel i,
periksa apakah:
  P10áµ¢ â‰¤ yáµ¢ â‰¤ P90áµ¢;

if (yáµ¢ berada dalam interval [P10, P90]?) then (ya)
  :Sampel tercakup (covered)
  in_interval = 1;
else (tidak)
  :Sampel tidak tercakup
  in_interval = 0;
endif

:== **Perhitungan PICP**
PICP = Î£(in_interval) / n
dimana n = jumlah total sampel;
note right
  PICP dihitung secara:
  â€¢ Keseluruhan (overall)
  â€¢ Per lokasi (N kabupaten (config-driven))
end note

if (PICP â‰ˆ 0.80?) then (ya, terkalibrasi)
  :Model terkalibrasi dengan baik
  Interval prediksi sesuai
  dengan cakupan nominal;
  #palegreen
elseif (PICP < 0.80?) then (under-coverage)
  :Interval terlalu sempit
  Model terlalu percaya diri
  (overconfident);
  #lightyellow
else (PICP > 0.80, over-coverage)
  :Interval terlalu lebar
  Model terlalu konservatif;
  #lightyellow
endif

:== **Pelaporan**
Tabel PICP overall dan per lokasi
disertakan dalam hasil evaluasi;

stop

@enduml
```

---

## 5. Diagram Alur Evaluasi dan Deteksi Kejadian Kekeringan

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam defaultFontName Arial
skinparam defaultFontSize 11
skinparam activityBackgroundColor #E8F4FD
skinparam activityBorderColor #2C3E50
skinparam arrowColor #2C3E50
skinparam noteBackgroundColor #FFF3CD
skinparam noteBorderColor #856404

title **Alur Evaluasi dan Deteksi Kejadian Kekeringan\nBerbasis Prediksi SPEI-3**

start

:== **Prediksi SPEI-3**
Output model TFT (P50 median)
pada data uji (â‰¥ 2024)
menggunakan dekoding step-0-only;

fork

  ' === Cabang Evaluasi Regresi ===
  :== **Evaluasi Regresi**
  Membandingkan nilai prediksi SPEI-3 (kontinu)
  dengan nilai aktual SPEI-3;

  :Perhitungan Metrik Regresi:
  â€¢ RMSE (root mean squared error)
  â€¢ MAE (mean absolute error)
  â€¢ RÂ² (koefisien determinasi)
  â€¢ Pearson r (korelasi linear)
  â€¢ Bias (kesalahan sistematis);

  :Perbandingan dengan Baseline Naif
  (Naive Persistence: SPEI(t) â‰ˆ SPEI(tâˆ’1));
  note right
    Model harus mengungguli
    baseline naif pada semua
    metrik utama
  end note

  :Evaluasi Metrik Probabilistik
  PICP pada interval P10â€“P90
  (cakupan nominal 80%);

fork again

  ' === Cabang Deteksi Kekeringan ===
  :== **Transformasi ke Deteksi Kekeringan**
  Konversi nilai kontinu SPEI-3
  menjadi kategori kekeringan;

  :Penerapan Threshold SPEI
  (Standar WMO);
  note right
    **Threshold Klasifikasi:**
    â‰¤ âˆ’2.0  : Kekeringan Ekstrem
    âˆ’2.0 ~ âˆ’1.5 : Kekeringan Parah
    âˆ’1.5 ~ âˆ’1.0 : Kekeringan Sedang
    âˆ’1.0 ~ âˆ’0.5 : Kekeringan Ringan
    âˆ’0.5 ~ +0.5 : Normal
    +0.5 ~ +1.0 : Basah Ringan
    +1.0 ~ +1.5 : Basah Sedang
    +1.5 ~ +2.0 : Basah Parah
    > +2.0  : Basah Ekstrem
  end note

  :Klasifikasi Biner Kejadian Kekeringan;
  note right
    **Deteksi Biner:**
    Kekeringan = SPEI < âˆ’0.5
    Tidak Kekeringan = SPEI â‰¥ âˆ’0.5
    (diterapkan pada aktual & prediksi)
  end note

  :== **Evaluasi Klasifikasi**
  Membandingkan label kekeringan
  prediksi vs aktual;

  :Konstruksi Confusion Matrix
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚             â”‚ Pred: Ya â”‚ Pred: Tidakâ”‚
  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
  â”‚ Aktual: Ya  â”‚   TP     â”‚    FN    â”‚
  â”‚ Aktual:Tidakâ”‚   FP     â”‚    TN    â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜;

  :Perhitungan Metrik Klasifikasi:
  â€¢ Precision = TP / (TP + FP)
  â€¢ Recall = TP / (TP + FN)
  â€¢ F1-Score = 2 Ã— (P Ã— R) / (P + R)
  â€¢ Accuracy = (TP + TN) / N;

end fork

:== **Analisis Hasil Komprehensif**
Integrasi metrik regresi, probabilistik,
dan klasifikasi per lokasi dan keseluruhan;

:Laporan evaluasi lengkap
dengan visualisasi dan tabel;

stop

@enduml
```

---

# Bagian 2: Tabel Penelitian

---

## Tabel 1. Variabel Dataset Penelitian

| No | Nama Variabel | Jenis Variabel | Deskripsi | Sumber Data |
|----|---------------|----------------|-----------|-------------|
| 1 | SPEI-3 | Target / Unknown | Standardized Precipitation-Evapotranspiration Index skala 3 bulan (rolling window 90 hari), dihitung menggunakan distribusi Log-Logistic | Dihitung dari data Open-Meteo |
| 2 | `precipitation_log` | Iklim / Unknown | Transformasi logaritmik presipitasi harian: log(1 + precipitation_sum) dalam satuan mm | Open-Meteo Archive API |
| 4 | `et0_fao_evapotranspiration` | Iklim / Unknown | Evapotranspirasi referensi harian berdasarkan persamaan FAO Penman-Monteith (mm) | Open-Meteo Archive API |
| 5 | `water_deficit` | Iklim / Unknown | Defisit air harian: presipitasi dikurangi evapotranspirasi referensi (P âˆ’ ETâ‚€) dalam mm | Dihitung dari data Open-Meteo |
| 6 | `soil_moisture` | Iklim / Unknown | Rata-rata kelembaban tanah harian pada kedalaman 0â€“7 cm (mÂ³/mÂ³) | Open-Meteo Archive API |
| 7 | `temperature_2m_max` | Iklim / Unknown | Suhu udara maksimum harian pada ketinggian 2 meter (Â°C) | Open-Meteo Archive API |
| 8 | `temperature_2m_min` | Iklim / Unknown | Suhu udara minimum harian pada ketinggian 2 meter (Â°C) | Open-Meteo Archive API |
| 9 | `time_idx` | Temporal / Known | Indeks waktu berupa jumlah hari sejak awal dataset, digunakan sebagai referensi sekuensial | Dihitung dari kolom waktu |
| 10 | `month_sin` | Temporal / Known | Komponen sinus dari encoding siklus bulanan: sin(2Ï€ Ã— bulan / 12) | Dihitung dari kolom waktu |
| 11 | `month_cos` | Temporal / Known | Komponen kosinus dari encoding siklus bulanan: cos(2Ï€ Ã— bulan / 12) | Dihitung dari kolom waktu |
| 12 | `location_id` | Spasial / Statis | Identifikator kategorikal kabupaten (City_A, City_B, City_C, City_D, City_E) | Didefinisikan manual |
| 13 | `elevation` | Spasial / Statis | Ketinggian lokasi di atas permukaan laut (m), konstan per lokasi | Open-Meteo Archive API |

---

## Tabel 2. Klasifikasi Variabel pada Model Temporal Fusion Transformer

| No | Nama Variabel | Kategori Variabel | Penjelasan |
|----|---------------|-------------------|------------|
| 1 | `location_id` | Static Categorical | Identifikator lokasi kabupaten yang bersifat tetap sepanjang waktu. Memungkinkan model mempelajari pola spasial spesifik untuk masing-masing dari N kabupaten (config-driven). |
| 2 | `elevation` | Static Real | Ketinggian lokasi (m) yang konstan per kabupaten. Memberikan konteks geografis statis yang memengaruhi pola iklim lokal. |
| 3 | `time_idx` | Time-Varying Known Real | Indeks waktu sekuensial (hari) yang diketahui untuk masa depan. Memungkinkan model memahami posisi temporal absolut. |
| 4 | `month_sin` | Time-Varying Known Real | Komponen sinus dari encoding siklus bulanan. Bersama `month_cos`, menangkap pola musiman secara kontinu tanpa diskontinuitas. Dapat dihitung untuk masa depan. |
| 5 | `month_cos` | Time-Varying Known Real | Komponen kosinus dari encoding siklus bulanan. Melengkapi `month_sin` untuk representasi siklus musiman yang lengkap. |
| 6 | SPEI-3 | Time-Varying Unknown Real | Variabel target utama. Hanya tersedia pada window encoder (historis); model harus memprediksinya untuk window decoder (masa depan). |
| 7 | `water_deficit` | Time-Varying Unknown Real | Defisit air harian (P − ET₀). Merupakan komponen fundamental dalam perhitungan SPEI dan indikator langsung keseimbangan air. |
| 8 | `precipitation_log` | Time-Varying Unknown Real | Presipitasi harian setelah transformasi logaritmik. Mengurangi skewness distribusi presipitasi yang sangat miring ke kanan. |
| 10 | `et0_fao_evapotranspiration` | Time-Varying Unknown Real | Evapotranspirasi referensi FAO. Merepresentasikan permintaan atmosfer terhadap air, dipengaruhi oleh radiasi dan suhu. |
| 11 | `soil_moisture` | Time-Varying Unknown Real | Kelembaban tanah permukaan. Indikator kondisi hidrologi aktual yang merespons presipitasi dan evaporasi. |
| 12 | `temperature_2m_max` | Time-Varying Unknown Real | Suhu udara maksimum harian. Memengaruhi laju evapotranspirasi dan intensitas kekeringan meteorologis. |
| 13 | `temperature_2m_min` | Time-Varying Unknown Real | Suhu udara minimum harian. Memberikan informasi rentang suhu diurnal yang relevan terhadap proses hidrologi. |

---

## Tabel 3. Konfigurasi Parameter Model Temporal Fusion Transformer

| No | Parameter | Nilai | Deskripsi |
|----|-----------|-------|-----------|
| 1 | Encoder Length | 90 hari | Panjang window historis yang digunakan sebagai input. Disesuaikan dengan window akumulasi SPEI-3 (3 Ã— 30 hari). |
| 2 | Prediction Horizon | 30 hari | Jumlah langkah waktu ke depan yang diprediksi secara simultan oleh model. |
| 3 | Hidden Size | 48 unit | Dimensi representasi laten pada setiap lapisan. Dikurangi dari standar 64/128 untuk mencegah overfitting pada dataset N lokasi (config-driven). |
| 4 | Attention Heads | 1 head | Jumlah head pada mekanisme multi-head attention. Satu head memadai untuk skala dataset yang relatif kecil. |
| 5 | Dropout | 0.35 | Probabilitas dropout untuk regularisasi stokastik. Ditingkatkan dari standar 0.1â€“0.3 untuk pengendalian overfitting yang lebih kuat. |
| 6 | Hidden Continuous Size | 8 unit | Dimensi representasi untuk variabel kontinu sebelum diproses oleh GRN. |
| 7 | Output Quantiles | 7 kuantil | Kuantil output: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]. P50 (median) digunakan sebagai prediksi titik utama. |
| 8 | Loss Function | QuantileLoss | Fungsi kerugian regresi kuantil yang mengoptimalkan seluruh 7 kuantil secara bersamaan. |
| 9 | Learning Rate | 3 Ã— 10â»â´ | Laju pembelajaran awal untuk optimizer Adam. |
| 10 | Weight Decay | 1 Ã— 10â»â´ | Regularisasi L2 untuk mencegah bobot model terlalu besar. |
| 11 | Batch Size (Train) | 32 | Ukuran mini-batch selama pelatihan. |
| 12 | Batch Size (Val/Test) | 64 | Ukuran mini-batch selama validasi dan pengujian (lebih besar karena tidak perlu backpropagation). |
| 13 | Max Epochs | 60 | Batas maksimum jumlah epoch pelatihan. |
| 14 | Early Stopping Patience | 25 epoch | Jumlah epoch tanpa perbaikan validation loss sebelum pelatihan dihentikan secara otomatis. |
| 15 | Early Stopping Min Delta | 1 Ã— 10â»â´ | Perubahan minimum pada validation loss yang dianggap sebagai perbaikan signifikan. |
| 16 | Gradient Clipping | 0.1 | Nilai maksimum norma gradien untuk mencegah exploding gradient. |
| 17 | Precision | float32 | Presisi aritmatika selama pelatihan. |
| 18 | Reduce LR Patience | 3 epoch | Jumlah epoch tanpa perbaikan sebelum learning rate diturunkan secara otomatis. |
| 19 | Optimizer | Adam | Algoritma optimisasi adaptif dengan momentum. |
| 20 | Accelerator | GPU (RTX 3050) | Perangkat komputasi yang digunakan untuk pelatihan. |

---

## Tabel 4. Pembagian Dataset Penelitian

| Subset Data | Periode Waktu | Durasi | Jumlah Sampel (Â± per lokasi) | Total Sampel (N lokasi (config-driven)) | Tujuan Penggunaan |
|-------------|---------------|--------|------------------------------|------------------------|-------------------|
| Training | 1 Januari 2005 â€“ 31 Desember 2022 | Â± 18 tahun | Â± 6.380 sekuens | Â± 31.900 sekuens | Pelatihan model dan fitting scaler (GroupNormalizer). Scaler difit secara eksklusif pada subset ini untuk mencegah kebocoran data. |
| Validation | 1 Januari 2023 â€“ 31 Desember 2023 | 1 tahun | Â± 365 sekuens | Â± 1.825 sekuens | Pemantauan overfitting selama pelatihan, penghentian dini (early stopping), dan penyesuaian learning rate. Menggunakan scaler dari data training. |
| Testing | 1 Januari 2024 â€“ 31 Desember 2025 | Â± 2 tahun | Â± 730 sekuens | Â± 3.650 sekuens | Evaluasi akhir model pada data yang tidak pernah dilihat selama pelatihan maupun validasi. Menggunakan scaler dari data training. |

**Catatan:**
- Pembagian dilakukan secara temporal-kronologis (bukan acak) untuk mencerminkan skenario peramalan dunia nyata.
- Tidak terdapat tumpang tindih antar subset: tahun < 2023 (training), tahun = 2023 (validation), tahun â‰¥ 2024 (testing).
- Jumlah sekuens valid memperhitungkan kebutuhan encoder (90 hari) dan prediction horizon (30 hari), sehingga lebih kecil dari jumlah hari mentah.

---

## Tabel 5. Metrik Evaluasi Kinerja Model (Regresi)

| No | Metrik | Rumus | Tujuan Penggunaan |
|----|--------|-------|-------------------|
| 1 | RMSE (Root Mean Squared Error) | $\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Mengukur besaran kesalahan prediksi secara keseluruhan dengan penalti lebih besar terhadap kesalahan besar. Satuan sama dengan variabel target (dimensionless untuk SPEI). |
| 2 | MAE (Mean Absolute Error) | $\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | Mengukur rata-rata kesalahan absolut prediksi. Lebih robust terhadap outlier dibandingkan RMSE. |
| 3 | RÂ² (Koefisien Determinasi) | $R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$ | Mengukur proporsi variansi data aktual yang dapat dijelaskan oleh model. Nilai 1.0 menunjukkan prediksi sempurna; nilai negatif menunjukkan model lebih buruk dari rata-rata. |
| 4 | Pearson r (Korelasi Pearson) | $r = \frac{\sum(y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum(y_i - \bar{y})^2 \cdot \sum(\hat{y}_i - \bar{\hat{y}})^2}}$ | Mengukur kekuatan dan arah hubungan linear antara nilai aktual dan prediksi. Nilai mendekati 1.0 menunjukkan korelasi linear positif sempurna. |
| 5 | Bias | $\text{Bias} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$ | Mengukur kesalahan sistematis model. Nilai positif menunjukkan prediksi cenderung terlalu tinggi (overestimate); nilai negatif menunjukkan underestimate. |
| 6 | PICP (Prediction Interval Coverage Probability) | $\text{PICP} = \frac{\#\{y_i \in [P10_i,\, P90_i]\}}{n}$ | Mengukur proporsi observasi aktual yang berada dalam interval prediksi P10â€“P90 (cakupan nominal 80%). Mengevaluasi kalibrasi ketidakpastian model. |

---

## Tabel 6. Metrik Evaluasi Deteksi Kejadian Kekeringan (Klasifikasi)

| No | Metrik | Deskripsi | Interpretasi |
|----|--------|-----------|--------------|
| 1 | Precision | Proporsi prediksi kekeringan positif yang benar-benar merupakan kejadian kekeringan aktual. Dihitung sebagai TP / (TP + FP). | Nilai tinggi menunjukkan rendahnya alarm palsu (*false alarm*). Precision tinggi penting untuk menghindari pemborosan sumber daya mitigasi akibat peringatan yang tidak tepat. |
| 2 | Recall (Sensitivity) | Proporsi kejadian kekeringan aktual yang berhasil dideteksi oleh model. Dihitung sebagai TP / (TP + FN). | Nilai tinggi menunjukkan model mampu menangkap sebagian besar kejadian kekeringan. Recall tinggi krusial untuk sistem peringatan dini agar tidak melewatkan kejadian kekeringan. |
| 3 | F1-Score | Rata-rata harmonik antara precision dan recall: 2 Ã— (P Ã— R) / (P + R). | Memberikan ukuran keseimbangan antara precision dan recall dalam satu metrik tunggal. Nilai mendekati 1.0 menunjukkan keseimbangan optimal antara deteksi kekeringan dan minimisasi alarm palsu. |
| 4 | Confusion Matrix | Tabel kontingensi 2Ã—2 yang menampilkan distribusi True Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN). | Memberikan gambaran lengkap kinerja klasifikasi biner. TP = kekeringan terdeteksi benar; TN = non-kekeringan terdeteksi benar; FP = alarm palsu; FN = kekeringan tidak terdeteksi. Threshold kekeringan: SPEI < âˆ’0.5. |

**Catatan:**
- Klasifikasi biner menggunakan threshold SPEI = âˆ’0.5 berdasarkan standar WMO: nilai SPEI < âˆ’0.5 dikategorikan sebagai kejadian kekeringan, dan SPEI â‰¥ âˆ’0.5 sebagai kondisi non-kekeringan.
- Metrik dihitung pada data uji (tahun â‰¥ 2024) secara keseluruhan maupun per lokasi.
- Dalam konteks sistem peringatan dini kekeringan, recall umumnya diprioritaskan di atas precision karena kegagalan mendeteksi kekeringan (FN) berpotensi menimbulkan kerugian pertanian yang lebih besar dibandingkan alarm palsu (FP).


## Update Diagram Kontrak Data (Schema v2)
- Tambah layer Raw Node: city_id -> node_local_id/node_id -> time series cuaca harian.
- Tambah tahap Node Selection train-only (top_k=5) sebelum agregasi.
- Tambah layer Super-Node harian: super_node_id per city_id untuk input model TFT.
- Validasi wajib: unik (node_id,time) pada raw dan unik (super_node_id,time) pada processed.


## Update Kontrak Diagram (Schema v2)

Kontrak arsitektur resmi:
- node-level ingestion: banyak node per city.
- agregasi super-node: tepat 1 `super_node_id` per (`city_id`,`time`).
- training group key: `super_node_id`.
- evaluasi: per-entity + per-city.
- semua asumsi daftar kota tetap dianggap tidak valid untuk pipeline produksi.

