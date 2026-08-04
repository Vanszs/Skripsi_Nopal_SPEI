# 📑 Dokumentasi & Overview Proyek: NusaPantau Kekeringan Indonesia (TFT SPEI Drought Forecasting)

---

## 📌 1. Tentang Proyek (Project Overview)

Proyek ini merupakan **sistem peramalan (forecasting) dan pemantauan kekeringan spasial-temporal** di seluruh wilayah Kabupaten/Kota Indonesia. Proyek riset skripsi ini mengintegrasikan pemodelan iklim hidrometeorologi berbasis indikator **SPEI (Standardized Precipitation Evapotranspiration Index)** dengan arsitektur deep learning **Temporal Fusion Transformer (TFT)**.

Sistem ini mengambil data parameter cuaca historis dan perkiraan dari **OpenMeteo SDK**, melakukan kalkulasi neraca air tanah (presipitasi dikurangi evapotranspirasi potensial), dan memprediksi tingkat kekeringan hingga 12 bulan ke depan beserta rentang ketidakpastian kuantilnya ($q_{0.10}, q_{0.50}, q_{0.90}$).

### A. Detail 5 Kota/Kabupaten Asli Studi (`city_id`) & Resolusi Spasial Grid
1. **Daftar 5 Wilayah Studi Asli (Jawa Timur)**:
   - Data diidentifikasi melalui variabel `city_id` yang mencakup 5 Kabupaten asli di Jawa Timur:
     - **Bojonegoro**: Lat `-7.155`, Lon `111.880` (Elevasi 18 MDPL)
     - **Lamongan**: Lat `-7.128`, Lon `112.316`
     - **Nganjuk**: Lat `-7.604`, Lon `111.905`
     - **Ngawi**: Lat `-7.403`, Lon `111.445`
     - **Tuban**: Lat `-6.895`, Lon `112.045`

2. **Arsitektur Grid Spasial 0.1° (`node_local_id` & `super_node_id`)**:
   - **Resolusi Grid**: Menggunakan resolusi spasial **0.1° ($\approx 11 \text{ km} \times 11 \text{ km}$ per sel grid)**.
   - **Sub-Node Grid**: Di sekitar titik pusat dari 5 kota di atas, sistem melakukan *sampling* 9 kandidat sub-node grid lokal per kota (total **45 raw node**) dengan format `node_local_id` (`n00` s/d `n08`):
     - `n00` (Pusat Kota), `n01` (Utara +13km), `n02` (Selatan -13km), `n03` (Timur +13km), `n04` (Barat -13km), `n05` (Timur Laut), `n06` (Barat Laut), `n07` (Tenggara), `n08` (Barat Daya).
   - **Penyintesisan Super-Node (`SN_<city_id>`)**: Dari 9 kandidat sub-node per kota, dipilih 5 node paling representatif (total **25 node terpilih**) yang merekam 5 variabel meteorologi harian (curah hujan, evapotranspirasi $ET_0$, kelembapan tanah, suhu, radiasi matahari). Lima node tersebut diamati dan digabungkan menjadi 5 **Super-Node** (`SN_Bojonegoro`, `SN_Lamongan`, `SN_Nganjuk`, `SN_Ngawi`, `SN_Tuban`) sebagai input utama model **Temporal Fusion Transformer (TFT)** untuk memprediksi indeks **SPEI**.

---

## 🎯 2. Tujuan Proyek (Project Goals & Purpose)

1. **Akurasi & Pemodelan Multi-Horizon**: Menghasilkan estimasi indeks kekeringan SPEI dengan tingkat presisi tinggi pada horizon 1, 3, 6, hingga 12 bulan menggunakan keunggulan arsitektur Transformer untuk time-series.
2. **Estimasi Ketidakpastian Kuantil (Quantile Uncertainty)**: Mengkuantifikasi risiko skenario terburuk ($q_{0.10}$), median ($q_{0.50}$), dan terbasah ($q_{0.90}$) sehingga pengambil kebijakan tidak hanya mengandalkan satu nilai tunggal.
3. **Penyampaian Informasi Ramah Publik**: Menyajikan hasil riset ilmiah yang rumit ke dalam antarmuka web publik yang mudah dipahami oleh masyarakat umum, petani, media, dan pemerintah daerah, tanpa terlihat seperti dashboard admin internal yang kaku.
4. **Keandalan & Standar Produksi (Robustness)**: Mengimplementasikan pipeline yang terverifikasi multi-seed, bebas dari flaw arsitektur via *Sonar Audit*, serta siap di-deploy secara publik menggunakan FastAPI, PM2, dan Nginx.

---

## 🏗️ 3. Arsitektur Proyek (System Architecture & Pipeline)

Sistem ini dibangun dengan arsitektur modular yang terbagi menjadi 6 layer utama:

```
[ OpenMeteo API ] ──► [ Data Ingestion & SPEI Calc ] ──► [ Preprocessing & TimeSeriesDataSet ]
                                                                   │
[ Vite.js Frontend ] ◄── [ FastAPI REST & SSE ] ◄── [ TFT Model (PyTorch Lightning) ]
```

### A. Data Ingestion & SPEI Engine (`src/data/`)
- `ingest.py`: Mengambil data cuaca historis dan perkiraan (curah hujan, suhu, kelembapan, radiasi matahari, kecepatan angin) dari OpenMeteo API.
- `spei.py`: Menghitung indeks SPEI skala 1, 3, 6, 12 bulan berbasis metode Thornthwaite/Penman-Monteith.
- `preprocess.py`: Penyeimbangan deret waktu, pembentukan feature lag, dan normalisasi spatial-temporal grid per wilayah.

### B. TFT Model Layer (`src/models/` & `src/training/`)
- `tft.py`: Arsitektur Temporal Fusion Transformer dengan *Variable Selection Networks*, *Gated Residual Networks (GRN)*, dan *Multi-Head Attention*.
- `dataset.py`: Konstruktor `TimeSeriesDataSet` dari PyTorch Forecasting untuk pemetaan kovariat statis (wilayah) dan dinamis (cuaca).
- `train.py`: Training loop PyTorch Lightning dengan *Pinball/Quantile Loss Minimization* dan *Learning Rate Finder*.

### C. Evaluation & Quality Assurance Layer (`src/evaluation/` & `sonar_audit/`)
- `metrics.py` & `calibration.py`: Perhitungan Quantile Loss, RMSE, MAE, Skill Score (86.5%), serta kurva kalibrasi.
- `run_audit.py` & `test_pipeline.py`: Audit kualitas kode static Sonar dan pengujian integrasi end-to-end.

### D. Backend API Service (`app/`)
- **FastAPI Async Engine** (`app/main.py` & `app/api/v1/endpoints/predict.py`): Menyediakan REST API endpoint untuk kalkulasi SPEI, inferensi model TFT, serta Server-Sent Events (SSE) / WebSocket untuk streaming status real-time.

### E. Frontend Application (`frontend/`)
- **Vite 6 + React 19 + TypeScript + Tailwind v4**: Aplikasi Single Page Application (SPA) publik yang cepat dan responsif.
- **Leaflet & Recharts**: Peta interaktif GeoJSON spasial dan grafik proyeksi kuantil *Fan-Chart*.

### F. Operations & Deployment Infrastructure
- **PM2 Ecosystem Manager** (`ecosystem.config.js`): Mengelola eksekusi paralel FastAPI backend (port 8005) dan Vite frontend (port 3005) dengan proteksi memory auto-restart.
- **Nginx Reverse Proxy** (`drought-monitor.conf`): Menangani enkripsi SSL Certbot, HTTP/2, Gzip/Brotli compression, rate limiting, serta header keamanan HSTS dan CSP.

---

## 🎨 4. Filosofi Desain Antarmuka: Anti-AI UI (Anti-AI-Slop & Taste Skill)

Salah satu pilar utama proyek ini adalah penolakan terhadap **"AI Slop UI"**—yaitu gaya antarmuka buatan AI/LLM yang homogen dan klise. 

### 🚫 Kenapa Menolak Default AI UI (AI Slop)?
LLM secara statistik cenderung menghasilkan UI dengan template seragam:
- Gradien warna ungu/indigo (`indigo-500`, `#6366f1`).
- Menggunakan font *Inter* atau *Geist* secara berlebihan tanpa variasi.
- Hero section terpusat dengan badge bertuliskan *"✨ Powered by AI"*.
- Layout 3 kartu sejajar yang kaku.
- Efek *glassmorphism* (`backdrop-blur-lg`) di semua elemen.
- Penggunaan tanda hubung em-dash (`—`) yang berlebihan pada teks.

### ✅ Penerapan Standar Design Taste & Anti-AI-Slop:

1. **Skema Warna Berbasis Karakter Iklim (Cold Meteorological Precision)**:
   - Warna latar belakang menggunakan *Dark Obsidian Navy* (`#0c1017`) yang tenang, bukan hitam pekat `#000000`.
   - Warna indikator SPEI menggunakan skala warna fungsional: *Emerald* (Normal/Aman), *Amber* (Waspada), *Orange* (Siaga), dan *Rose Red* (Awas/Ekstrem).
2. **Sistem Tipografi yang Terarah**:
   - Teks antarmuka menggunakan **Geist** untuk keterbacaan tinggi.
   - Seluruh data angka kuantil, koordinat, dan matriks ilmiah menggunakan **JetBrains Mono**.
3. **Penyampaian Bahasa Populer (Human-Centered Copywriting)**:
   - Istilah statistik rumit diterjemahkan ke bahasa yang ramah publik:
     - $q_{0.10} \rightarrow$ **Skenario Paling Kering** (Batas Terburuk)
     - $q_{0.50} \rightarrow$ **Proyeksi Utama** (Estimasi Median)
     - $q_{0.90} \rightarrow$ **Skenario Paling Basah** (Batas Terbaik)
   - Dilengkapi **Rekomendasi Aksi Nyata** untuk petani, BPBD/Pemda, dan warga umum.
4. **Larangan Keras Em-Dash (`—`) & Buzzword Hype**:
   - Seluruh teks antarmuka bersih dari em-dash (`—`) dan kata-kata klise buatan AI seperti *"supercharge"*, *"revolutionize"*, atau *"seamless"*.
5. **Layout Asimetris (7:5 Split Screen)**:
   - Menggabungkan Peta Spasial Indonesia (7 Kolom) dengan Kartu Proyeksi Kuantil & Rekomendasi Wilayah (5 Kolom) dalam komposisi bernapas.
6. **Aksesibilitas & Kepatuhan WCAG 2.1 AA**:
   - Memiliki kontras rasio tinggi (> 4.5:1), navigasi keyboard lengkap (`tabIndex={0}`), serta dukungan pembaca layar (`aria-label`).

---

## 🎨 5. Spesifikasi Komponen & Panduan Desain Frontend (UI Checklist)

Setiap pengembang/AI lain yang akan melanjutkan pengembangan atau merancang ulang UI **WAJIB** memasukkan elemen-elemen berikut ke dalam antarmuka:

### A. Elemen & Fitur Utama Antarmuka
1. **Header & Portal Identity**:
   - Nama Portal: **NusaPantau Kekeringan Indonesia** (Bahasa publik bersahabat, bukan Admin Dashboard).
   - Menu Navigasi Tab: `Peta & Prediksi`, `Panduan Warga`, dan `Metode AI`.
   - Tombol Aksi Utama: `Unduh Laporan` (Membuka modal ekspor PDF, CSV, GeoJSON).

2. **Hero Section & Search Bar**:
   - Spanduk Peringatan Nasional (misal: *Waspada Kemarau*).
   - **Search Input Universal**: Pengguna dapat mencari nama Kabupaten/Kota mereka secara langsung (`Bojonegoro`, `Lamongan`, `Nganjuk`, `Ngawi`, `Tuban`).
   - Chip Pencarian Populer & 4 Rangkuman Metrik (Wilayah Terpantau, Skill Score 86.5%, Status Nasional, Sync Real-time).

3. **Peta Spasial Grid Interaktif (`DroughtMap`)**:
   - Map Basemap Dark Vector Tile (`CartoDB Dark Matter`).
   - Pewarnaan Choropleth sesuai skala status SPEI (-3.0 s/d +3.0).
   - Tooltip hover & interaksi klik titik lokasi untuk memperbarui detail wilayah terpilih.

4. **Visualisasi Fan-Chart Kuantil SPEI (`TFTFanChart`)**:
   - Grafik area bertumpuk Recharts yang menampilkan tren 6 bulan historis dan pita ketidakpastian prediksi (+1M s/d +12M).
   - Penjelasan Skenario Kuantil:
     - $q_{0.10}$: **Skenario Paling Kering** (Batas Terburuk)
     - $q_{0.50}$: **Proyeksi Utama** (Estimasi Median)
     - $q_{0.90}$: **Skenario Paling Basah** (Batas Terbaik)

5. **Kartu Rekomendasi Aksi Nyata (Langkah Publik & Pemda)**:
   - Menyajikan saran konkret untuk **Sektor Pertanian** (pemilihan varietas palawija), **BPBD/Pemda** (kesiapan embung & pompa air), dan **Masyarakat** (hemat air).

6. **Panduan Warga & Transparansi Metode AI**:
   - Penjelasan istilah SPEI dengan bahasa populer.
   - Transparansi metrik riset skripsi (Pinball Loss 0.142, Dropout Rate 0.40, Evaluasi 3-Seed Verified `87eae91`).

---

## 🛠️ Perintah Utama Operasional Proyek

```bash
# 1. Jalankan Backend REST API FastAPI (Port 8005)
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8005

# 2. Build & Jalankan Frontend Vite Web Portal (Port 3005)
cd frontend
pnpm run build
pnpm run preview --port 3005 --host 0.0.0.0

# 3. Jalankan Seluruh Layanan via PM2 Produksi
pm2 start ecosystem.config.js
```
