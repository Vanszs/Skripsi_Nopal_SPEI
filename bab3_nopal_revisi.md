# BAB III
# METODOLOGI PENELITIAN

## 3.1 Tahapan Penelitian
Metodologi penelitian ini dirancang secara sistematis untuk membangun, melatih, dan mengevaluasi sistem peramalan multi-horizon kekeringan lahan pertanian berbasis indeks SPEI-3 menggunakan model Temporal Fusion Transformer (TFT). Tahapan penelitian dirancang secara terstruktur dan terintegrasi untuk menjamin reproduksibilitas hasil, keandalan pengujian, serta keamanan dari kebocoran data (*data leakage*). Secara garis besar, tahapan penelitian ini digambarkan pada Gambar 3.1 yang meliputi:
1. **Pengumpulan Data (Data Ingestion)**: Pengambilan data meteorologi historis harian multi-node dari API eksternal dengan mekanisme pertahanan kegagalan jaringan.
2. **Pra-pemrosesan Data (Data Preprocessing)**: Pembersihan data, pengisian nilai hilang secara temporal-safe, seleksi titik spasial representatif menggunakan pendekatan hibrida (*hybrid scoring*), dan agregasi menjadi entitas spasial super-node per kota.
3. **Komputasi Indeks SPEI-3**: Perhitungan akumulasi defisit air, fitting parameter distribusi statistik pada periode latihan, pergeseran domain, dan transformasi normal standar dengan ekstrapolasi ekor distribusi (*deep-tail extrapolation*).
4. **Pembentukan Dataset Deret Waktu (Time Series Dataset Building)**: Perekayasaan fitur temporal siklikal, diferensiasi temporal, pembuatan variabel kovariat spasial, pembobotan normalisasi, dan pembentukan DataLoader.
5. **Perancangan dan Pelatihan Model TFT**: Arsitektur jaringan seleksi variabel, LSTM encoder-decoder, attention mechanism, optimasi multi-kuantil (P10, P50, P90), dan konfigurasi hyperparameter optimal.
6. **Kalibrasi Interval Konformal (Conformal-style Calibration)**: Penyesuaian lebar interval prediksi probabilistik secara adaptif menggunakan faktor skala multiplikatif per kota untuk mencapai cakupan nominal 80% tanpa penyusutan ekstrem.
7. **Evaluasi dan Pengujian Model**: Pengujian point forecast dan uncertainty forecast menggunakan metrik RMSE, MAE, R², Pearson r, dan PICP, serta perbandingan terhadap model pembanding naive persistence.

---

## 3.2 Pengumpulan Data (Data Ingestion)
Tahap awal penelitian adalah proses pengumpulan data iklim sekunder multivariat dengan resolusi harian untuk wilayah studi sentra padi Jawa Timur yang meliputi 5 kabupaten: Bojonegoro, Lamongan, Nganjuk, Ngawi, dan Tuban. Informasi koordinat geografis pusat wilayah penelitian ditunjukkan pada Tabel 3.1.

### Tabel 3.1 Koordinat Pusat Wilayah Penelitian
| No | Wilayah | Latitude | Longitude |
| :--- | :--- | :--- | :--- |
| 1 | Bojonegoro | -7.155 | 111.880 |
| 2 | Lamongan | -7.128 | 112.316 |
| 3 | Nganjuk | -7.604 | 111.905 |
| 4 | Ngawi | -7.403 | 111.445 |
| 5 | Tuban | -6.895 | 112.045 |

Data meteorologi di-ingest dari layanan **Open-Meteo Archive API** (`https://archive-api.open-meteo.com/v1/archive`) dengan menggunakan zona waktu lokal `Asia/Jakarta`. Rentang waktu pengambilan data dibatasi secara kronologis mulai dari **1 Januari 2005 hingga 1 Januari 2026**. Pemilihan skala harian untuk indeks SPEI ini merujuk pada keunggulan deteksi kekeringan secara cepat (*flash drought*) serta pemantauan dinamika lengas tanah jangka pendek sebagaimana yang direkomendasikan oleh Vicente-Serrano et al. (2020).

Untuk meningkatkan representasi spasial dan menghindari kelemahan data titik tunggal (*point observation*) yang rentan terhadap gangguan (*noise*) lokal, sistem pengumpulan data tidak hanya mengambil data pada koordinat pusat kota saja. Untuk setiap kota, sistem secara otomatis membangun sebuah grid spasial yang terdiri atas **9 kandidat node** menggunakan offset koordinat tetap (`DEFAULT_NODE_OFFSETS`) sebagai berikut:
*   **n00 (Pusat Kota)**: Offset Lat $0.00$, Lon $0.00$
*   **n01 (Utara)**: Offset Lat $+0.12$, Lon $0.00$ (~13 km dari pusat)
*   **n02 (Selatan)**: Offset Lat $-0.12$, Lon $0.00$ (~13 km dari pusat)
*   **n03 (Timur)**: Offset Lat $0.00$, Lon $+0.12$ (~13 km dari pusat)
*   **n04 (Barat)**: Offset Lat $0.00$, Lon $-0.12$ (~13 km dari pusat)
*   **n05 (Timur Laut)**: Offset Lat $+0.08$, Lon $+0.08$
*   **n06 (Barat Laut)**: Offset Lat $+0.08$, Lon $-0.08$
*   **n07 (Tenggara)**: Offset Lat $-0.08$, Lon $+0.08$
*   **n08 (Barat Daya)**: Offset Lat $-0.08$, Lon $-0.08$

Dengan konfigurasi ini, total kandidat node spasial yang di-ingest untuk seluruh wilayah penelitian adalah $5 \text{ kota} \times 9 \text{ node} = 45 \text{ node}$. Setiap node diberi identitas unik yang deterministik dan *collision-safe* dengan skema penamaan:
$$\text{node\_id} = \text{city\_id} \parallel \text{"\_\_"} \parallel \text{node\_local\_id} \parallel \text{"\_\_"} \parallel \text{coord\_token}$$
Di mana $\text{coord\_token}$ adalah 8 karakter pertama dari representasi *hash* SHA-1 terhadap koordinat geografis node tersebut (format: `"{lat:.6f},{lon:.6f}"`).

Proses penarikan data dari API Open-Meteo dilengkapi dengan mekanisme *retry* yang kokoh guna menangani kendala koneksi jaringan atau pembatasan laju data (*rate limiting*). Mekanisme ini dikonfigurasi dengan batas maksimum **6 kali percobaan** (*attempts*):
1.  Jika menerima respons HTTP 429 (*Too Many Requests*), sistem akan membaca *header* respons `Retry-After` dan melakukan penundaan eksekusi (*sleep*) selama $\max(1, \text{Retry-After})$ detik.
2.  Jika menerima kesalahan server internal (HTTP 5xx), sistem menerapkan jeda $3 \times (\text{attempt} + 1)$ detik.
3.  Untuk kegagalan pengecualian (*exception*) umum lainnya, sistem menerapkan jeda $2 \times (\text{attempt} + 1)$ detik.

Variabel meteorologi harian utama yang dikumpulkan ditunjukkan pada Tabel 3.2.

### Tabel 3.2 Parameter Penelitian
| No | Parameter | Deskripsi | Nama Parameter pada Dataset |
| :--- | :--- | :--- | :--- |
| 1 | Presipitasi Harian | Total curah hujan harian (mm) | `precipitation_sum` |
| 2 | Evapotranspirasi | Jumlah air menguap potensial harian (mm/hari) | `et0_fao_evapotranspiration` |
| 3 | Suhu Maksimum | Suhu udara tertinggi harian (°C) | `temperature_2m_max` |
| 4 | Suhu Minimum | Suhu udara terendah harian (°C) | `temperature_2m_min` |
| 5 | Kelembaban Tanah Atas | Kadar air rata-rata tanah kedalaman 0-7 cm (m³/m³) | `soil_moisture` |
| 6 | Kelembaban Udara | Rata-rata kelembaban udara relatif harian (%) | `relative_humidity_2m_mean` |
| 7 | Radiasi Gelombang Pendek | Total energi radiasi matahari harian (MJ/m²) | `shortwave_radiation_sum` |
| 8 | Kecepatan Angin | Rata-rata kecepatan angin pada ketinggian 10 m (km/jam) | `wind_speed_10m_mean` |

---

## 3.3 Pra-pemrosesan Data (Data Preprocessing)
Tahap pra-pemrosesan data dilakukan secara hati-hati pada level node spasial individual sebelum proses penggabungan wilayah dilakukan, untuk menjamin tidak terjadinya kebocoran informasi masa depan (*data leakage*) ke dalam model.

### 3.3.1 Validasi Skema dan Interpolasi Nilai Hilang
Pertama, sistem melakukan verifikasi bahwa seluruh berkas data mentah yang di-ingest memenuhi kriteria skema terpadu (`schema_version = 2`) yang mewajibkan keberadaan kolom koordinat, elevasi, identitas kota, identitas node, serta 8 variabel cuaca utama.

Apabila terdapat nilai yang hilang (*missing values*), penanganan dilakukan menggunakan metode **forward fill** (`ffill` dengan batas `limit=7` hari) secara independen pada setiap `raw_node_id`. Metode ini hanya mengizinkan pengisian data kosong dari nilai valid terakhir pada deret waktu masa lalu, sehingga bebas dari kebocoran data. Persamaan matematis forward fill dirumuskan sebagai:
$$x_t = x_{t-1} \quad \text{jika } x_t \text{ bernilai kosong/NaN}$$
Jika terdapat node yang memiliki nilai kosong berurutan lebih dari 7 hari, sistem akan memicu peringatan (*warning*) kualitas data guna mendeteksi anomali stasiun pengamatan.

### 3.3.2 Seleksi Node Hibrida (Hybrid Selective Node)
Untuk mereduksi noise spasial dan menyaring stasiun cuaca yang kurang representatif, dilakukan seleksi untuk memilih **5 node terbaik dari 9 kandidat** pada setiap kota. Proses seleksi ini bersifat *train-only*, yaitu **hanya menggunakan data historis hingga batas tanggal seleksi 31 Desember 2022 (`SELECTION_END_DATE`)**. Hal ini krusial agar karakteristik data pengujian (tahun 2024 ke atas) tidak memengaruhi proses pemilihan node.

Seleksi node menggunakan kriteria skor hibrida (*hybrid score*) yang menggabungkan kemiripan perilaku data iklim (*behavior similarity*) dengan kedekatan geografis (*distance score*):
1.  **Perhitungan Jarak Geografis (Haversine Formula)**:
    Jarak geografis ($d_i$) antara kandidat node $i$ dengan koordinat pusat wilayah dihitung menggunakan rumus Haversine (Hasanah & Suharso, 2023; Maria et al., 2020):
    $$d_i = 2R \cdot \arcsin\left(\sqrt{\sin^2\left(\frac{\text{lat}_i - \text{lat}_{\text{center}}}{2}\right) + \cos(\text{lat}_{\text{center}}) \cdot \cos(\text{lat}_i) \cdot \sin^2\left(\frac{\text{lon}_i - \text{lon}_{\text{center}}}{2}\right)}\right)$$
    Di mana $R = 6371.0 \text{ km}$ (jari-jari bumi), sedangkan seluruh koordinat telah dikonversi ke dalam satuan radian. Skor jarak kemudian dinormalisasi dalam rentang $(0, 1]$ melalui:
    $$\text{distance\_score}_i = rac{1}{1 + d_i}$$
2.  **Perhitungan Skor Perilaku (Behavior Similarity)**:
    Karakteristik perilaku iklim diukur melalui rata-rata koefisien korelasi Pearson ($\rho$) (Saccenti et al., 2020; Tang & Zhao, 2025) di antara kandidat node $i$ terhadap profil kota pembanding yang dibentuk menggunakan pendekatan *leave-one-node-out* (rata-rata 8 node lain tanpa melibatkan node $i$):
    $$\text{behavior\_score}_i = rac{1}{M} \sum_{m=1}^{M} \rho(\mathbf{v}_{i, m}, \mathbf{v}_{\text{others}, m})$$
    Di mana $M = 8$ (jumlah variabel meteorologi) dan $\mathbf{v}$ merupakan vektor deret waktu harian.
    *Catatan Penanganan Fitur Degenerasi*: Selama proses perhitungan korelasi, apabila ditemukan stasiun yang memiliki fitur bernilai konstan (misalnya elevasi) sehingga simpangan bakunya nol ($\sigma_a = 0$ atau $\sigma_b = 0$), maka perhitungan korelasi variabel tersebut dilewati (*skipped*) untuk mencegah pembagian dengan nol (*division by zero*).
3.  **Skor Akhir Hibrida (Hybrid Score)**:
    Gabungan kedua nilai dihitung dengan bobot $70\%$ untuk pola meteorologi dan $30\%$ untuk kedekatan spasial:
    $$\text{hybrid\_score}_i = 0.7 \cdot \text{behavior\_score}_i + 0.3 \cdot \text{distance\_score}_i$$

Untuk menjaga replikasi hasil secara konsisten pada berbagai sistem operasi, skor-skor tersebut dibulatkan hingga 12 angka di belakang koma (`round(12)`). Pengurutan seleksi dilakukan secara deterministik menggunakan algoritma *mergesort* dengan prioritas: `hybrid_score` menurun, `behavior_score` menurun, `distance_score` menurun, dan alfabetis `raw_node_id` menaik sebagai pemecah nilai kembar (*tie-breaker*).

### 3.3.3 Agregasi Super-Node Wilayah
Lima node spasial terbaik hasil seleksi hibrida kemudian dilebur (*aggregated*) per kota untuk membentuk satu entitas spasial terpadu yang disebut **Super-Node** dengan format nama `SN_{city_id}` (contoh: `SN_Bojonegoro`). Operasi agregasi menggunakan rata-rata aritmatika untuk seluruh variabel cuaca harian dan koordinat statis:
$$\text{SN}_{t, m} = rac{1}{K} \sum_{k=1}^{K} x_{t, m, k}$$
Di mana $K = 5$ (jumlah node terpilih), dan $x_{t, m, k}$ adalah nilai variabel meteorologi $m$ pada hari ke-$t$ untuk node spasial ke-$k$. Agregasi super-node ini secara efektif mereduksi derau lokal (*microclimate noise*) tanpa memerlukan pemodelan graf spasial yang rumit.

### 3.3.4 Standardisasi Input Fitur (Z-Score Scaling)
Seluruh fitur meteorologi kontinu yang telah didefinisikan pada super-node ditransformasikan ke skala terstandardisasi (Z-score, mean = 0, std = 1) sebelum diumpankan ke model TFT. Metode standardisasi ini dirumuskan sebagai:
$$z_t = \frac{x_t - \mu}{\sigma}$$
Di mana $\mu$ dan $\sigma$ berturut-turut merupakan nilai rata-rata dan deviasi standar yang dihitung **secara eksklusif hanya pada data periode pelatihan (sebelum tahun 2023)** untuk mencegah kebocoran informasi.

*Justifikasi Pemilihan Z-Score*:
1.  **Preservasi Kejadian Ekstrem**: Berbeda dengan Min-Max scaling yang membatasi nilai pada rentang kaku $[0, 1]$, Z-score tidak membatasi batas atas atau bawah. Fitur anomali cuaca ekstrem (seperti curah hujan badai atau suhu ekstrem) tidak terkompresi secara artifisial, melainkan tetap terepresentasi sebagai nilai deviasi standar yang tinggi.
2.  **Distribusi Normal Alami**: Karakteristik variabel meteorologi kontinu (seperti suhu dan tekanan udara) secara alami cenderung mengikuti distribusi normal/Gaussian, sehingga Z-score mempertahankan bentuk distribusi data asli.
3.  **Kestabilan Gradien**: Fitur masukan dengan skala $[-3, +3]$ terbukti meminimalkan risiko saturasi gradien pada fungsi aktivasi model deep learning (seperti Gated Linear Units / GLU dan SiLU) sehingga mempercepat konvergensi pelatihan.

Dalam implementasi kode, standardisasi ini dikemas dalam kelas kustom `ArrayStandardScaler` (turunan dari `StandardScaler` scikit-learn). Kelas ini dimodifikasi agar selalu melakukan transformasi pada level *array* numpy 2D (`_to_2d_array`), guna menghindari peringatan tidak validnya nama fitur (*feature name warnings*) ketika model memproses *batch* data tensor PyTorch.

---

## 3.4 Komputasi Indeks SPEI-3
Tahap komputasi indeks SPEI-3 dilakukan untuk membentuk variabel target utama peramalan. Indeks SPEI dirancang dengan skala 3 bulan (SPEI-3) untuk memantau kekeringan lahan pertanian jangka pendek berdasarkan kerangka kerja Vicente-Serrano et al. (2020) dan dataset global terstandardisasi Beguería et al. (2020).

1.  **Kalkulasi Defisit Air Harian ($D_t$)**:
    Defisit air dihitung sebagai selisih antara presipitasi harian ($P_t$) dan evapotranspirasi potensial ($ET0_t$):
    $$D_t = P_t - ET0_t$$
    *Catatan*: Nilai $ET0_t$ diperoleh langsung dari Open-Meteo API yang dihitung secara standar menggunakan formula **FAO-56 Penman-Monteith** sebagaimana diuraikan dalam pembaruan komprehensif Pereira et al. (2025), bukan dihitung secara manual di dalam kode.
2.  **Akumulasi Jendela Rolling ($X_t^{(3)}$)**:
    Karena SPEI-3 berbasis skala 3 bulan, nilai defisit harian diakumulasikan sepanjang jendela waktu 90 hari ke belakang:
    $$X_t^{(3)} = \sum_{i=0}^{89} D_{t-i}$$
3.  **Pergeseran Domain Positif (Shifted Domain)**:
    Pemodelan SPEI menggunakan fitting distribusi kumulatif Log-Logistic (Fisk). Karena distribusi Fisk hanya terdefinisi pada domain positif ($> 0$), sedangkan akumulasi defisit air $X_t^{(3)}$ sering bernilai negatif, maka dilakukan pergeseran nilai menggunakan faktor *shift*:
    $$sm_t = X_t^{(3)} + \text{shift}$$
    Di mana nilai pergeseran ditentukan berdasarkan nilai minimum pada masa pelatihan saja untuk menghindari kebocoran data:
    $$\text{shift} = \begin{cases} \left| X_{\text{train, min}}^{(3)} \right| + 1.0, & \text{jika } X_{\text{train, min}}^{(3)} \le 0 \\ 0.0, & \text{jika } X_{\text{train, min}}^{(3)} > 0 \end{cases}$$
4.  **Fitting Parameter Distribusi Fisk**:
    Parameter skala ($\beta$), bentuk ($\alpha$), dan lokasi ($floc=0$) dari distribusi Fisk diestimasi **secara terpisah untuk setiap bulan kalender (Januari s.d. Desember)** menggunakan metode Maximum Likelihood Estimation (MLE) yang hanya diterapkan pada subset data periode latihan (sebelum tahun 2023). Penggunaan distribusi Fisk (Log-Logistic) ini secara empiris terbukti memberikan representasi ekor probabilitas yang lebih seimbang untuk indikator kekeringan dibanding fungsi distribusi alternatif lainnya (Vicente-Serrano et al., 2021).
5.  **Ekstrapolasi Linier Ekor Distribusi (Deep-Tail Extrapolation)**:
    Apabila terjadi anomali kekeringan ekstrem baru pada data uji yang nilainya melampaui batas historis data latih ($sm_t \le x_{\epsilon}$), fungsi CDF Fisk standar ($F_{\text{fisk}}$) akan mengalami saturasi mendekati 0. Hal ini menyebabkan kegagalan perhitungan Z-score normal standar (nilai mendekati $-\infty$), yang merupakan masalah numerik umum pada kalkulasi SPEI konvensional saat menghadapi *out-of-bounds data* akibat parameter distribusi yang kaku (Stankeviciute & Alaa, 2021; Vicente-Serrano et al., 2021). Untuk mengatasi masalah kestabilan matematis ini, sistem menerapkan ekstrapolasi linier pada ekor distribusi (*deep-tail extrapolation*) sebagai berikut:
    $$\text{SPEI}_t = \begin{cases} \Phi^{-1}\Big(\text{clip}\big(F_{\text{fisk}}(sm_t), 10^{-6}, 1 - 10^{-6}\big)\Big), & \text{jika } sm_t > x_{\epsilon} \\ z_{\text{floor}} + \dfrac{sm_t - x_{\epsilon}}{\beta}, & \text{jika } sm_t \le x_{\epsilon} \end{cases}$$
    Di mana $\Phi^{-1}$ adalah invers CDF normal standar, $z_{\text{floor}} = \Phi^{-1}(10^{-6}) \approx -4.753$ (batas bawah teoretis), $\beta$ adalah parameter skala Fisk hasil fitting, dan $x_{\epsilon} = F_{\text{fisk}}^{-1}(10^{-6})$ adalah nilai batas bawah ekor distribusi.

Interpretasi nilai hasil komputasi SPEI-3 diselaraskan secara konsisten dengan kategori resmi standar WMO yang divalidasi kembali untuk pemantauan kekeringan dalam Zhang & Li (2020) serta Mehta et al. (2025) yang ditunjukkan pada Tabel 3.3.

### Tabel 3.3 Klasifikasi Kategori Indeks SPEI
| Batas Ambang Nilai SPEI | Kategori Kondisi |
| :--- | :--- |
| $\text{SPEI} \le -2.0$ | Kekeringan Ekstrem |
| $-2.0 < \text{SPEI} \le -1.5$ | Kekeringan Parah |
| $-1.5 < \text{SPEI} \le -1.0$ | Kekeringan Sedang |
| $-1.0 < \text{SPEI} < -0.5$ | Kekeringan Ringan |
| $-0.5 \le \text{SPEI} \le 0.5$ | Normal |
| $0.5 < \text{SPEI} < 1.0$ | Basah Ringan |
| $1.0 \le \text{SPEI} < 1.5$ | Basah Sedang |
| $1.5 \le \text{SPEI} < 2.0$ | Basah Parah |
| $\text{SPEI} \ge 2.0$ | Basah Ekstrem |

---

## 3.5 Pembentukan Dataset Time Series
Data harian yang telah diproses kemudian disusun ke dalam format dataset sekuensial deret waktu (*time series dataset*) menggunakan representasi sliding window.

### 3.5.1 Perekayasaan Fitur Temporal dan Spasial
Sebelum diumpankan ke model, sistem melakukan rekayasa fitur (*feature engineering*) untuk membekali model dengan informasi siklikal temporal dan tren perubahan nilai:
1.  **Diferensiasi Temporal SPEI-3 (First-order Difference)**:
    Menambahkan fitur perubahan harian SPEI-3 untuk menangkap kecepatan dinamika perkembangan kebasahan/kekeringan lahan:
    $$\Delta \text{SPEI\_3}_t = \text{SPEI\_3}_t - \text{SPEI\_3}_{t-1}$$
2.  **Transformasi Logaritma Presipitasi**:
    Menghitung logaritma dari curah hujan harian untuk menstabilkan sebaran nilai presipitasi yang memiliki tingkat kecondongan (*skewness*) ekstrem:
    $$\text{precipitation\_log}_t = \ln(1 + \text{precipitation\_sum}_t)$$
3.  **Representasi Temporal Siklikal**:
    Fitur bulan kalender ($m \in [1, 12]$) dikonversi menjadi representasi koordinat sinus dan cosinus 2D untuk mempertahankan hubungan kedekatan siklus tahunan (misalnya Desember dekat dengan Januari):
    $$\text{month\_sin}_t = \sin\left(\frac{2\pi \cdot \text{month}_t}{12}\right)$$
    $$\text{month\_cos}_t = \cos\left(\frac{2\pi \cdot \text{month}_t}{12}\right)$$
4.  **Indeks Waktu Kontinu**:
    Membuat fitur `time_idx` yang menghitung jarak hari sejak tanggal awal dataset ($t_{min}$):
    $$\text{time\_idx}_t = t - t_{min}$$
    Fitur ini bertindak sebagai penunjuk posisi temporal bagi model.

---

## 3.6 Perancangan Model Temporal Fusion Transformer (TFT)
Model utama yang dirancang untuk peramalan multi-horizon SPEI-3 ini adalah Temporal Fusion Transformer (TFT) berbasis arsitektur orisinal Lim et al. (2021). Model TFT menggabungkan keunggulan jaringan rekuren untuk pemrosesan lokal jangka pendek dengan mekanisme *self-attention* untuk menangkap pola jangka panjang, serta dilengkapi jaringan interpretasi kontribusi fitur. TFT dirancang secara modular dengan spesifikasi jumlah lapisan (*layer*) pada masing-masing sub-jaringannya sebagai berikut:
1. **Variable Selection Network (VSN)**: Terdiri atas 1 lapisan VSN untuk setiap kategori variabel input (statis, masa lalu, dan masa depan) yang berfungsi untuk menyaring fitur secara adaptif. Di dalam VSN, setiap fitur diproses oleh satu blok *Gated Residual Network* (GRN) yang memiliki **2 lapisan linier padat (dense layers)** dengan aktivasi *Exponential Linear Unit* (ELU), diikuti oleh lapisan *Gated Linear Unit* (GLU) (Shazeer, 2020) dan *Layer Normalization*.
2. **LSTM Encoder-Decoder**: Menggunakan **1 lapisan LSTM** (recurrent network) pada bagian encoder (jendela historis 90 hari) dan **1 lapisan LSTM** pada bagian decoder (horizon prediksi 30 hari) untuk menangkap ketergantungan temporal jangka pendek dan menengah secara berurutan.
3. **Multi-Head Self-Attention**: Menggunakan **1 lapisan attention** dengan **1 kepala atensi (1 head)** untuk mempelajari hubungan temporal jangka panjang di sepanjang jendela waktu 90 hari tanpa menambah memori berlebih.
4. **Quantile Output Layer**: Menggunakan **1 lapisan proyeksi linier (dense output layer)** untuk memetakan representasi laten akhir menjadi **3 dimensi keluaran** yang bersesuaian dengan estimasi kuantil target (P10, P50, P90).

### 3.6.1 Definisi 18 Variabel Model TFT
Seluruh fitur yang digunakan oleh model TFT dikonfigurasi secara ketat di dalam objek `TimeSeriesDataSet` dari PyTorch Forecasting dengan pembagian peran yang ditunjukkan pada Tabel 3.4.

### Tabel 3.4 Struktur 18 Variabel Input Model TFT
| No | Nama Fitur di Kode | Kategori Fitur | Tipe Data | Deskripsi | Peran dalam Jaringan TFT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `SPEI_3` | Target Utama | Real | Indeks SPEI skala 3 bulanan | Target Output & Time-varying Unknown Real (Past) |
| 2 | `precipitation_log` | Cuaca | Real | Log dari curah hujan harian | Time-varying Unknown Real (Past Only) |
| 3 | `et0_fao_evapotranspiration` | Cuaca | Real | Evapotranspirasi potensial harian | Time-varying Unknown Real (Past Only) |
| 4 | `soil_moisture` | Cuaca | Real | Kelembaban tanah harian | Time-varying Unknown Real (Past Only) |
| 5 | `temperature_2m_max` | Cuaca | Real | Suhu harian maksimum | Time-varying Unknown Real (Past Only) |
| 6 | `temperature_2m_min` | Cuaca | Real | Suhu harian minimum | Time-varying Unknown Real (Past Only) |
| 7 | `relative_humidity_2m_mean` | Cuaca | Real | Rata-rata kelembaban relatif | Time-varying Unknown Real (Past Only) |
| 8 | `shortwave_radiation_sum` | Cuaca | Real | Total radiasi matahari harian | Time-varying Unknown Real (Past Only) |
| 9 | `wind_speed_10m_mean` | Cuaca | Real | Rata-rata kecepatan angin | Time-varying Unknown Real (Past Only) |
| 10 | `water_deficit` | Turunan | Real | Selisih presipitasi dan ET0 | Time-varying Unknown Real (Past Only) |
| 11 | `SPEI_3_diff` | Turunan | Real | Perubahan harian SPEI-3 | Time-varying Unknown Real (Past Only) |
| 12 | `time_idx` | Temporal | Real | Indeks hari kontinu | Time-varying Known Real (Past & Future) |
| 13 | `month_sin` | Temporal | Real | Siklus bulan (Sinus) | Time-varying Known Real (Past & Future) |
| 14 | `month_cos` | Temporal | Real | Siklus bulan (Cosinus) | Time-varying Known Real (Past & Future) |
| 15 | `elevation` | Spasial | Real | Ketinggian wilayah rata-rata | Static Real (Kovariat Spasial Konstan) |
| 16 | `lat` | Spasial | Real | Lintang wilayah rata-rata | Static Real (Kovariat Spasial Konstan) |
| 17 | `lon` | Spasial | Real | Bujur wilayah rata-rata | Static Real (Kovariat Spasial Konstan) |
| 18 | `super_node_id` | Identitas | Cat | ID super-node kota pengamatan | Static Categorical / Group ID |

*Catatan Target Normalization*: Untuk menstabilkan proses pelatihan target `SPEI_3`, model menggunakan objek `EncoderNormalizer(transformation=None)` dari PyTorch Forecasting. Modul ini melakukan standardisasi dinamis pada target secara adaptif per sekuens berdasarkan histori nilai pada jendela encoder masing-masing sampel.

### 3.6.2 Konfigurasi Hyperparameter dan Justifikasi Desain
Model TFT diimplementasikan menggunakan kombinasi hyperparameter optimal hasil eksperimen pengujian parameter (*ablation study*) untuk menjamin konvergensi yang sehat pada dataset kecil (5 entitas super-node). Konfigurasi parameter model ditunjukkan pada Tabel 3.5.

### Tabel 3.5 Konfigurasi Arsitektur dan Hyperparameter TFT
| Parameter | Nilai | Justifikasi Desain Teknis |
| :--- | :--- | :--- |
| **Encoder Length** | 90 hari | Jendela input historis sepanjang 90 hari dipilih agar model dapat merangkum seluruh informasi meteorologi harian yang menyusun indeks target SPEI-3 (3 bulan). |
| **Prediction Length** | 30 hari | Horizon peramalan multi-horizon selama 30 hari ke depan secara simultan untuk pemantauan jangka pendek. |
| **Quantile Output** | P10, P50, P90 | Output multi-kuantil untuk menghasilkan titik prediksi median (P50) serta interval ketidakpastian probabilistik (P10 - P90). |
| **Hidden Size** | 48 | Kapasitas dimensi tersembunyi berukuran 48 terbukti optimal untuk representasi fitur 5 wilayah stasiun. Ukuran yang lebih besar (seperti 64) memicu *overfitting* instan sejak epoch awal karena keterbatasan jumlah wilayah. |
| **Dropout** | 0.40 | Nilai regularisasi dropout tinggi (0.40) dipilih sebagai pelindung utama dari penyempitan jurang *train-val loss gap* tanpa merusak kelancaran konvergensi model. |
| **Attention Heads** | 1 | Kepala atensi tunggal (1 head) dinilai cukup untuk melacak dependensi temporal 90 hari tanpa menambah kompleksitas jalur penghafalan pola (*memorization paths*). |
| **Hidden Continuous Size** | 8 | Dimensi representasi untuk variabel kontinu, dikonfigurasi proporsional terhadap ukuran hidden ($\text{hidden\_size} // 6 = 8$). |
| **Learning Rate** | 0.0003 | Jembatan laju pembaruan bobot optimal. Nilai $10^{-3}$ menyebabkan gradien meledak (*diverge*), sedangkan $10^{-4}$ terlalu lambat mencapai titik optimum. |
| **Reduce LR Patience** | 8 epoch | Batas toleransi validator `ReduceLROnPlateau`. Jika loss validasi tidak membaik dalam 8 epoch berturut-turut, laju pembelajaran dipangkas setengahnya ($0.5\times$) untuk keluar dari area lokal datar. |
| **Weight Decay** | 0.0001 | Regularisasi L2 sebesar $10^{-4}$ disematkan pada bobot untuk membatasi pertumbuhan nilai magnitudo parameter model. |
| **Gradient Clipping** | 0.5 | Batas atas norma gradien maksimum sebesar 0.5 untuk mengontrol kestabilan *backpropagation* dari recurrent layer. |

Fungsi kerugian (*loss function*) yang dioptimalkan oleh model TFT selama pelatihan adalah **Quantile Loss (Pinball Loss)** untuk tiga kuantil target $q \in [0.1, 0.5, 0.9]$:
$$\mathcal{L}_q(y, \hat{y}_q) = \max\Big(q(y - \hat{y}_q), (q - 1)(y - \hat{y}_q)\Big)$$
Di mana $y$ adalah nilai aktual SPEI-3 target, dan $\hat{y}_q$ adalah nilai prediksi kuantil ke-$q$. Total loss merupakan akumulasi rata-rata pinball loss dari ketiga kuantil tersebut di seluruh horizon prediksi (1 hingga 30 hari).

---

## 3.7 Prosedur Pelatihan dan Validasi Model
Prosedur pelatihan model dirancang secara ketat untuk membagi data berdasarkan sumbu waktu kronologis (*chronological split*) guna mensimulasikan skenario peramalan operasional nyata di masa depan:
*   **Dataset Pelatihan (Training Set)**: Menggunakan seluruh data dari periode **29 Juni 2005 hingga 31 Desember 2022**.
*   **Dataset Validasi (Validation Set)**: Menggunakan data sepanjang tahun **2023 (1 Januari 2023 hingga 31 Desember 2023)**.
*   **Dataset Pengujian (Testing Set)**: Menggunakan data dari **1 Januari 2024 hingga 1 Januari 2026** (periode di mana performa diuji secara independen).

Proses pelatihan dijalankan menggunakan pustaka **PyTorch Lightning** dengan ukuran *mini-batch* sebesar **32 sekuens** per iterasi. Optimasi parameter dilakukan menggunakan algoritma Adam. Kriteria *Early Stopping* diatur pada monitor `val_loss` dengan nilai toleransi perbaikan minimum `min_delta = 1e-4` dan tingkat kesabaran `patience = 10` epoch.

### 3.7.1 Konfigurasi Akselerasi dan Kestabilan Komputasi
Pelatihan model dilakukan pada perangkat keras GPU (CUDA) dengan konfigurasi akselerasi spesifik:
1.  **Presisi Campuran Bfloat16 (`precision="bf16-mixed"`)**:
    Pelatihan model menggunakan presisi `bf16-mixed`, bukan FP16 (`16-mixed`). Pada arsitektur TFT, nilai bias atensi untuk sekuens masking dihitung menggunakan konstanta negatif yang sangat besar ($<-10^4$). Penggunaan presisi FP16 standar akan memicu kondisi *underflow/overflow* matematis karena keterbatasan rentang eksponen FP16 (maksimum $\approx 6.5 \times 10^4$). Presisi Bfloat16 (BF16) memiliki rentang eksponen yang identik dengan FP32 (Kalamkar et al., 2021), sehingga memberikan keuntungan kompresi VRAM GPU dan kecepatan operasi *tensor core* tanpa mengalami resiko ketidakstabilan numerik.
2.  **Akselerasi Perkalian Matriks**:
    Sistem mengaktifkan akselerasi perkalian matriks menggunakan perintah `torch.set_float32_matmul_precision("medium")` untuk memanfaatkan arsitektur Tensor Cores pada GPU modern tanpa mengorbankan konvergensi model.
3.  **Workaround Kompatibilitas Pemuatan Checkpoint (PyTorch 2.6+)**:
    Pada versi PyTorch 2.6 ke atas, fungsi `torch.load` secara bawaan mengaktifkan parameter keamanan `weights_only=True`. Kebijakan ini secara otomatis akan menolak pemuatan modul scaler `ArrayStandardScaler` dan objek metadata normalisasi `EncoderNormalizer` yang tersimpan di dalam berkas checkpoint model (`.ckpt`). Untuk menjamin pipeline evaluasi berjalan mulus di berbagai versi PyTorch, sistem menerapkan fungsi pemuat kustom `load_tft_checkpoint` yang secara dinamis mengabaikan pembatasan tersebut dengan menyetel `weights_only=False` hanya selama proses instansiasi model berlangsung.

---

### 3.7.2 Kalibrasi Interval Konformal (Conformal-style Calibration)
Sebagai salah satu kontribusi metodologis untuk meningkatkan keandalan (*reliability*) interval prediksi probabilistik, model peramalan dilengkapi dengan modul **Conformal-style Interval Calibration** per wilayah super-node. Pendekatan kalibrasi ini diturunkan dari kerangka kerja *Conformalized Quantile Regression* (CQR) (Stankeviciute & Alaa, 2021; Zaffran et al., 2022) guna memberikan jaminan cakupan nominal probabilistik tanpa bergantung pada asumsi distribusi tertentu (*distribution-free coverage guarantee*).

Prediksi kuantil mentah yang dihasilkan model TFT (P10 dan P90) seringkali mengalami deviasi cakupan (*coverage gap*) pada data uji akibat heterogenitas spasial antar kota dan sifat deret waktu non-stasioner, yang melanggar asumsi pertukaran data (*data exchangeability*) (Zaffran et al., 2022). Untuk menyelaraskan cakupan interval agar konsisten mencapai target nominal coverage **80%** (interval P10 s.d. P90) secara adaptif di setiap wilayah, sistem menghitung faktor skala multiplikatif $s_c$ untuk setiap super-node kota $c$ menggunakan data periode validasi (tahun 2023) secara eksklusif. Konsep penskalaan lokal ini merujuk pada prinsip *Normalized Conformal Prediction* (Angelopoulos & Bates, 2021; Dewolf et al., 2025) dan adaptasi spasial lokal (Jiang & Xie, 2024):

1.  **Kalkulasi Deviasi Residu Ternormalisasi**:
    Untuk setiap sampel sekuens validasi, dihitung nilai residu absolut antara nilai SPEI-3 aktual ($y_t$) dengan nilai prediksi median P50 ($\hat{y}_{t, P50}$), dibagi dengan setengah lebar interval prediksi mentah:
    $$e_t = \frac{|y_t - \hat{y}_{t, P50}|}{0.5 \cdot (\hat{y}_{t, P90} - \hat{y}_{t, P10})}$$
2.  **Penentuan Faktor Skala Kota ($s_c$)**:
    Faktor skala $s_c$ ditentukan sebagai nilai kuantil ke-0.80 (nominal $80\%$) dari seluruh himpunan residu $e_t$ pada kota $c$ tersebut:
    $$s_c = \text{Quantile}\Big( \{e_t\}_{t \in \text{Val}}, 0.80 \Big)$$
    Untuk mencegah penyusutan interval prediksi secara ekstrem yang berlebihan saat model terlalu percaya diri (*overconfident*), sistem menerapkan batas bawah (*floor limit*) sebagai pelindung:
    $$s_c = \max(s_c, 0.5)$$
3.  **Aplikasi Kalibrasi pada Data Uji**:
    Pada tahap pengujian (inferensi), interval prediksi P10 dan P90 dikalibrasi ulang secara dinamis menggunakan faktor skala $s_c$ yang bersesuaian dengan kota tersebut:
    $$\hat{y}_{t, P10}^{\text{calibrated}} = \hat{y}_{t, P50} - 0.5 \cdot s_c \cdot \left(\hat{y}_{t, P90} - \hat{y}_{t, P10}\right)$$
    $$\hat{y}_{t, P90}^{\text{calibrated}} = \hat{y}_{t, P50} + 0.5 \cdot s_c \cdot \left(\hat{y}_{t, P90} - \hat{y}_{t, P10}\right)$$
    Sedangkan nilai prediksi median P50 tetap dipertahankan tanpa perubahan.

---

## 3.8 Peramalan Multi-Horizon
Setelah model Temporal Fusion Transformer selesai dilatih, divalidasi, dan modul kalibrasi disiapkan, tahap berikutnya adalah melakukan inferensi peramalan multi-horizon untuk memproyeksikan nilai SPEI-3 hingga 30 hari ke depan secara simultan. Model menerima input sequence historis sepanjang 90 hari ($t-89$ hingga $t$) dan menghasilkan prediksi sekuensial sepanjang 30 hari ke depan ($t+1$ hingga $t+30$).

Formulasi peramalan multi-horizon ini dinyatakan sebagai:
$$\hat{Y}_{t+1:t+30} = \Big\{\hat{y}_{t+1}, \hat{y}_{t+2}, \dots, \hat{y}_{t+30}\Big\}$$
Pendekatan ini jauh lebih unggul dibandingkan metode peramalan rekursif satu-langkah (*recursive single-step forecasting*) karena meminimalkan akumulasi kesalahan perambatan (*error propagation*) dan memanfaatkan dependensi temporal antar-horizon secara langsung di dalam struktur decoder.

---

## 3.9 Evaluasi Model
Evaluasi model dirancang untuk menguji keandalan hasil peramalan pada dataset pengujian (2024–2026). Skenario evaluasi disusun secara terperinci untuk mengukur performa point forecast (median P50), interval probabilistik, perbandingan baseline, degradasi horizon, serta visualisasi kontribusi fitur. Rincian skenario evaluasi ditunjukkan pada Tabel 3.6.

### Tabel 3.6 Skenario Evaluasi Model
| Skenario Pengujian | Periode Data Uji | Parameter / Output yang Diamati | Metrik Evaluasi |
| :--- | :--- | :--- | :--- |
| **Evaluasi Point Forecast Agregat** | Data uji 2024–2026 | Prediksi median P50 SPEI-3 seluruh data uji | RMSE, MAE, R², Pearson r |
| **Evaluasi Point Forecast per Kota** | Data uji 2024–2026 | Prediksi median P50 SPEI-3 untuk setiap super-node | RMSE, MAE, R², Pearson r per super-node |
| **Evaluasi Probabilistik Agregat** | Data uji 2024–2026 | Lebar dan coverage interval prediksi P10-P90 | PICP (Target nominal coverage 80%) |
| **Evaluasi Probabilistik per Kota** | Data uji 2024–2026 | Interval prediksi P10-P90 setiap super-node | PICP per super-node |
| **Perbandingan Model Baseline** | Data uji 2024–2026 | Perbandingan P50 TFT vs Naive Persistence | RMSE, MAE, persentase kemenangan per horizon |
| **Evaluasi Degradasi antar-Horizon**| Data uji 2024–2026 | Prediksi P50 pada horizon hari ke-1 s.d. hari ke-30 | RMSE dan MAE per langkah horizon (1-30) |
| **Evaluasi Dampak Kalibrasi** | Data uji 2024–2026 | Interval P10-P90 sebelum vs sesudah kalibrasi | PICP sebelum dan sesudah kalibrasi per kota |
| **Analisis Interpretabilitas** | Data uji 2024–2026 | Bobot atensi VSN dan Temporal Attention | Peringkat kepentingan fitur dan atensi temporal |

Metrik evaluasi numerik dihitung menggunakan rumus-rumus berikut:
1.  **Root Mean Squared Error (RMSE)**:
    $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
2.  **Mean Absolute Error (MAE)**:
    $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
3.  **Koefisien Determinasi ($R^2$)**:
    $$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$
4.  **Koefisien Korelasi Pearson ($r$)**:
    $$r = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{N} (x_i - \bar{x})^2 \sum_{i=1}^{N} (y_i - \bar{y})^2}}$$
5.  **Prediction Interval Coverage Probability (PICP)**:
    $$\text{PICP} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\Big(y_i \in [P_{10, i}, P_{90, i}]\Big)$$
    Di mana $\mathbb{I}$ merupakan fungsi indikator bernilai 1 jika nilai aktual berada di dalam interval kuantil, dan 0 jika sebaliknya.

Sebagai model pembanding sederhana, digunakan pendekatan **Naive Persistence Baseline** yang merumuskan bahwa nilai SPEI-3 pada horizon ke-$h$ di masa depan diasumsikan sama dengan nilai observasi SPEI-3 terakhir yang teramati pada waktu ke-$t$:
$$\hat{y}_{t+h} = y_t$$

---

## 3.10 Analisis Interpretabilitas Model
Guna membongkar sifat kotak hitam (*black-box*) yang umum pada model deep learning, penelitian memanfaatkan kemampuan interpretasi bawaan (*built-in interpretability*) dari model TFT melalui analisis bobot Variable Selection Network (VSN) dan matriks *Self-Attention*.
1.  **Variable Importance**: Bobot VSN dianalisis untuk menentukan peringkat kontribusi variabel input, baik variabel statis (seperti koordinat spasial), variabel temporal masa lalu, maupun variabel temporal yang diketahui di masa depan.
2.  **Temporal Attention Weights**: Pola perhatian temporal dianalisis menggunakan rata-rata bobot atensi dari *self-attention layer* pada decoder untuk mengidentifikasi hari keberapa pada jendela historis 90 hari yang paling berpengaruh terhadap peramalan.

Analisis ini murni bersifat eksploratif-deskriptif untuk memahami pola hubungan statistik yang ditangkap model, bukan untuk menyimpulkan hubungan kausalitas fisik secara mutlak.

---

## 3.11 Perancangan Web Visualisasi Hasil Peramalan
Untuk mempermudah pemantauan dan analisis kekeringan lahan pertanian oleh pengguna akhir, sistem peramalan ini diintegrasikan ke dalam sebuah platform web dashboard interaktif. Web visualisasi dirancang sebagai sistem penyajian (*visualizer layer*) hasil inferensi model tanpa melakukan pelatihan ulang model di sisi klien.

Arsitektur aplikasi web dirancang dengan memisahkan sisi depan (*frontend*) dan sisi belakang (*backend*) untuk memudahkan pemeliharaan:
*   **Frontend (Vue.js)**: Membangun antarmuka pengguna yang dinamis dan reaktif. Komponen antarmuka meliputi modul pemilih kota (Bojonegoro, Lamongan, Nganjuk, Ngawi, Tuban), ringkasan status kelas kekeringan SPEI-3 terkini, grafik deret waktu peramalan 30 hari ke depan yang dilengkapi pita area ketidakpastian probabilistik kuantil (calibrated P10 dan P90), serta visualisasi representatif bobot kepentingan variabel (*variable importance*).
*   **Backend (FastAPI)**: Menyediakan layanan Web API berbasis JSON untuk mentransmisikan data prediksi, status klasifikasi kekeringan (sesuai Tabel 3.3), dan nilai metrik evaluasi dari berkas penyimpanan lokal (.json dan .parquet hasil evaluasi model).
*   **Alur Integrasi**: Ketika pengguna memilih kota tertentu pada antarmuka Vue.js, aplikasi akan mengirimkan permintaan (*request*) HTTP ke endpoint FastAPI. Backend akan mengambil data hasil peramalan terkalibrasi yang bersesuaian, menghitung kelas kekeringan berdasarkan nilai SPEI, dan mengembalikan respons JSON. Antarmuka Vue.js kemudian menyajikan visualisasi grafik secara reaktif.

---

## DAFTAR PUSTAKA
1. Angelopoulos, A. N., & Bates, S. (2021). *A gentle introduction to conformal prediction and distribution-free uncertainty quantification*. **arXiv preprint arXiv:2107.07511**.
2. Beguería, S., Vicente-Serrano, S. M., Reig-Gracia, F., & Latorre Garcés, B. (2020). *SPEIbase v.2.6 [Dataset]*. **DIGITAL.CSIC**. https://doi.org/10.20350/digitalCSIC/15555.
3. Dewolf, N., De Baets, B., & Waegeman, W. (2025). *Conditional validity of heteroskedastic conformal regression*. **Information and Inference: A Journal of the IMA**, 14(2), iaaf013.
4. Hasanah, M., & Suharso, A. (2023). *Algoritma Haversine pada Sistem Informasi Geografis: Tinjauan Literatur Sistematis*. **Nuansa Informatika**, 17(2), 135-143.
5. Jiang, H., & Xie, Y. (2024). *Spatial Conformal Inference through Localized Quantile Regression*. **arXiv preprint arXiv:2412.01098**.
6. Kalamkar, D., et al. (2021). *BFLOAT16: A numerical format for deep learning training*. **IEEE 26th Symposium on Computer Arithmetic (ARITH)**, 2021, pp. 23-30.
7. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*. **International Journal of Forecasting**, 37(4), 1748-1764.
8. Mehta, D., Caloiero, T., Yadav, S., & Kumar, V. (2025). *Rainfall temporal variability and drought analysis by means of the Standardized Precipitation Index in Ganganagar District, Rajasthan, India*. **Frontiers in Climate**, 7, 1702356.
9. Maria, E., Budiman, E., Haviluddin, & Taruk, M. (2020). *Measure distance locating nearest public facilities using Haversine and Euclidean Methods*. **Journal of Physics: Conference Series**, 1450, 012080.
10. Pereira, L. S., Allen, R. G., Paredes, P., López-Urrea, R., Raes, D., Smith, M., Kilic, A., & Salman, M. (2025). *Crop evapotranspiration – Guidelines for computing crop water requirements. Second edition, revised 2025*. **FAO Irrigation and Drainage Paper, No. 56 Rev.1. Rome, FAO**. https://doi.org/10.4060/cd6621en.
11. Saccenti, E., Hendriks, M. H., & Smilde, A. K. (2020). *Corruption of the Pearson correlation coefficient by measurement error and its estimation, bias, and correction under different error models*. **Scientific Reports**, 10, 438.
12. Shazeer, N. (2020). SwiGLU: *GLU Variants Improve Transformer*. **arXiv preprint arXiv:2002.05202**.
13. Stankeviciute, K., & Alaa, A. M. (2021). *Conformal Time-series Forecasting*. **Advances in Neural Information Processing Systems (NeurIPS)**, 34, 312f1ba2.
14. Tang, S., & Zhao, X. (2025). *Limitations of Correlation Coefficients in Research on Functional Connectomes and Psychological Processes*. **Human Brain Mapping**, 46(10), e70287.
15. Vicente-Serrano, S. M., et al. (2020). *A multi-scalar daily standardized precipitation evapotranspiration index (SPEI) for drought monitoring*. **Agricultural and Forest Meteorology**, 290, 108031.
16. Vicente-Serrano, S. M., et al. (2021). *Evaluating candidate distributions for the standardized precipitation evapotranspiration index (SPEI)*. **International Journal of Climatology**, 41(S1), E1-E21.
17. Zaffran, M., Dieuleveut, A., Feron, O., Goude, Y., & Julie, J. (2022). *Adaptive Conformal Predictions for Time Series*. **Proceedings of the 39th International Conference on Machine Learning (ICML)**, PMLR 162:25834-25866.
18. Zhang, Y., & Li, Z. (2020). *Uncertainty Analysis of Standardized Precipitation Index Due to the Effects of Probability Distributions and Parameter Errors*. **Frontiers in Earth Science**, 8, 76.
