# REVISI MATEMATIS DRAF SKRIPSI: RUMUS & VARIABEL PIPELINE SPEI-TFT V2

Dokumen ini berisi kumpulan **5 Rumus Matematika** yang belum tercantum di draf skripsi beserta penjelasan detail **18 Variabel Model TFT** (termasuk target utama). Dokumen ini siap digunakan sebagai bahan revisi Bab III (Metodologi Penelitian).

---

## I. LIMA RUMUS MATEMATIS PIPELINE V2

### 1. Rumus Jarak Spasial (Haversine & Distance Score)
Digunakan pada tahapan seleksi node spasial untuk menghitung jarak geografis antara koordinat kandidat node dengan koordinat pusat wilayah (kota/kabupaten).

*   **Jarak Haversine ($d$):**
    $$d_i = 2R \cdot \arcsin\left(\sqrt{\sin^2\left(\frac{\text{lat}_i - \text{lat}_{\text{center}}}{2}\right) + \cos(\text{lat}_{\text{center}}) \cdot \cos(\text{lat}_i) \cdot \sin^2\left(\frac{\text{lon}_i - \text{lon}_{\text{center}}}{2}\right)}\right)$$
    *Di mana:*
    *   $R = 6371.0 \text{ km}$ adalah rata-rata jari-jari bumi.
    *   $\text{lat}_i, \text{lon}_i$ adalah koordinat kandidat node ke-$i$ (dalam radian).
    *   $\text{lat}_{\text{center}}, \text{lon}_{\text{center}}$ adalah koordinat pusat kota/kabupaten (dalam radian).

*   **Skor Jarak ($\text{distance\_score}$):**
    $$\text{distance\_score}_i = \frac{1}{1 + d_i}$$
    *Fungsi:* Menormalkan nilai jarak menjadi skor berbanding terbalik dalam rentang $(0, 1]$. Node yang tepat berada di pusat wilayah memiliki skor 1.0.

---

### 2. Rumus Seleksi Node Hibrida (Hybrid Score)
Digunakan untuk menentukan 5 node terbaik dari 9 kandidat grid di sekitar pusat kota secara deterministik.

*   **Skor Gabungan Hibrida (Hybrid Score):**
    $$\text{hybrid\_score}_i = 0.7 \cdot \text{behavior\_score}_i + 0.3 \cdot \text{distance\_score}_i$$
    *Catatan:* Bobot kesamaan perilaku iklim dinilai lebih krusial ($70\%$) dibanding kedekatan spasial ($30\%$).

*   **Skor Kesamaan Perilaku (Behavior Score):**
    $$\text{behavior\_score}_i = \frac{1}{M} \sum_{m=1}^{M} \rho(\mathbf{v}_{i, m}, \mathbf{v}_{\text{others}, m})$$
    *Di mana:*
    *   $M = 8$ adalah jumlah variabel cuaca/iklim harian (sesuai Tabel 3.2).
    *   $\rho$ melambangkan koefisien korelasi Pearson.
    *   $\mathbf{v}_{i, m}$ adalah deret waktu historis variabel $m$ pada node $i$.
    *   $\mathbf{v}_{\text{others}, m}$ adalah profil rata-rata temporal variabel $m$ dari wilayah tersebut tanpa melibatkan node $i$ (untuk mencegah bias diri).

---

### 3. Rumus Transformasi Positif (Shifted Domain) untuk Fitting Fisk
Distribusi Fisk (Log-Logistik) mensyaratkan data input harus bernilai positif ($> 0$), sedangkan deret akumulasi defisit air ($X_t^{(3)}$) sering bernilai negatif. Oleh karena itu, dilakukan pergeseran domain sebelum pencocokan distribusi.

*   **Defisit Air Tergeser ($sm_t$):**
    $$sm_t = X_t^{(3)} + \text{shift}$$

*   **Nilai Pergeseran (Shift Value):**
    $$\text{shift} = \begin{cases} \left| X_{\text{train, min}}^{(3)} \right| + 1.0, & \text{jika } X_{\text{train, min}}^{(3)} \le 0 \\ 0.0, & \text{jika } X_{\text{train, min}}^{(3)} > 0 \end{cases}$$
    *Di mana:*
    *   $X_{\text{train, min}}^{(3)}$ adalah nilai defisit air kumulatif minimum yang tercatat selama masa pelatihan (train period). Batas minimal ini di-lock untuk validasi dan pengujian guna mencegah kebocoran data (*leakage*).

---

### 4. Rumus Novelty: Ekstrapolasi Linier Ekor Distribusi (Deep-Tail Extrapolation)
Ini adalah aspek kebaruan (*novelty*) sistem untuk menangani kondisi kekeringan ekstrem baru yang belum pernah terjadi pada masa pelatihan (nilai defisit melampaui batas historis ekor distribusi).

$$\text{SPEI}_t = \begin{cases} \Phi^{-1}\Big(\text{clip}\big(F_{\text{fisk}}(sm_t), 10^{-6}, 1 - 10^{-6}\big)\Big), & \text{jika } sm_t > x_{\epsilon} \\ z_{\text{floor}} + \dfrac{sm_t - x_{\epsilon}}{\beta}, & \text{jika } sm_t \le x_{\epsilon} \end{cases}$$

*Di mana:*
*   $\Phi^{-1}$ melambangkan *inverse CDF* dari distribusi Normal standar.
*   $F_{\text{fisk}}$ melambangkan fungsi distribusi kumulatif (*CDF*) dari distribusi Fisk yang sudah difit.
*   $\text{clip}(x, a, b)$ adalah fungsi pemotongan nilai $x$ pada rentang $[a, b]$.
*   $z_{\text{floor}} = \Phi^{-1}(10^{-6}) \approx -4.753$ (batas bawah nilai SPEI teoretis).
*   $x_{\epsilon} = F_{\text{fisk}}^{-1}(10^{-6})$ adalah nilai defisit tergeser batas bawah ekor distribusi.
*   $\beta$ adalah parameter skala (scale) dari fungsi distribusi Fisk hasil fitting data training.

---

### 5. Rumus Fitur Hasil Feature Engineering (Transformasi Log & Diferensiasi)
Perubahan matematis pada variabel masukan untuk menstabilkan varians dan menangkap dinamika temporal perubahan SPEI.

*   **Transformasi Logaritma Presipitasi:**
    $$\text{precipitation\_log}_t = \ln(1 + \text{precipitation\_sum}_t)$$
    *Fungsi:* Menstabilkan sebaran nilai curah hujan harian yang memiliki kecondongan (*skewness*) sangat tinggi.

*   **Diferensiasi Temporal SPEI-3 (First-order Difference):**
    $$\Delta \text{SPEI\_3}_t = \text{SPEI\_3}_t - \text{SPEI\_3}_{t-1}$$
    *Fungsi:* Memperkenalkan tren jangka pendek pergerakan indeks kekeringan ke model TFT.

---

## II. DAFTAR 18 VARIABEL DAN TARGET MODEL TFT

Berikut adalah struktur lengkap 18 variabel yang dimuat dalam konfigurasi `TimeSeriesDataSet` pada model Temporal Fusion Transformer:

| No | Nama Variabel pada Kode | Kategori Fitur | Tipe Data | Deskripsi | Peran dalam Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **`SPEI_3`** | **Target Utama** | **Real** | Indeks SPEI skala 3 bulanan | **Output / Target** (sekaligus *unknown input* masa lalu) |
| 2 | `precipitation_log` | Cuaca (API) | Real | Log dari total curah hujan harian | Time-varying Unknown Real (Past Only) |
| 3 | `et0_fao_evapotranspiration` | Cuaca (API) | Real | Laju evapotranspirasi potensial harian | Time-varying Unknown Real (Past Only) |
| 4 | `soil_moisture` | Cuaca (API) | Real | Kelembaban tanah harian kedalaman 0-7 cm | Time-varying Unknown Real (Past Only) |
| 5 | `temperature_2m_max` | Cuaca (API) | Real | Suhu harian maksimum ketinggian 2m | Time-varying Unknown Real (Past Only) |
| 6 | `temperature_2m_min` | Cuaca (API) | Real | Suhu harian minimum ketinggian 2m | Time-varying Unknown Real (Past Only) |
| 7 | `relative_humidity_2m_mean` | Cuaca (API) | Real | Rata-rata kelembaban udara harian | Time-varying Unknown Real (Past Only) |
| 8 | `shortwave_radiation_sum` | Cuaca (API) | Real | Total energi radiasi matahari harian | Time-varying Unknown Real (Past Only) |
| 9 | `wind_speed_10m_mean` | Cuaca (API) | Real | Rata-rata kecepatan angin harian | Time-varying Unknown Real (Past Only) |
| 10 | `water_deficit` | Turunan | Real | Selisih curah hujan dan ET0 ($P - ET_0$) | Time-varying Unknown Real (Past Only) |
| 11 | `SPEI_3_diff` | Turunan | Real | Perubahan harian SPEI-3 ($\Delta \text{SPEI\_3}$) | Time-varying Unknown Real (Past Only) |
| 12 | `time_idx` | Temporal | Real | Indeks urutan waktu kontinu (hari) | Time-varying Known Real (Past & Future) |
| 13 | `month_sin` | Temporal | Real | Representasi siklus bulan (Sinus) | Time-varying Known Real (Past & Future) |
| 14 | `month_cos` | Temporal | Real | Representasi siklus bulan (Cosinus) | Time-varying Known Real (Past & Future) |
| 15 | `elevation` | Spasial | Real | Ketinggian wilayah di atas permukaan laut | Static Real (Kovariat Spasial Konstan) |
| 16 | `lat` | Spasial | Real | Koordinat lintang wilayah (latitude) | Static Real (Kovariat Spasial Konstan) |
| 17 | `lon` | Spasial | Real | Koordinat bujur wilayah (longitude) | Static Real (Kovariat Spasial Konstan) |
| 18 | `super_node_id` | Identitas | Cat | ID penanda entitas super-node kota | Static Categorical / Group ID |
