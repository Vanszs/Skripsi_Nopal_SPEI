# REVISI V5 — Audit Rumus, Sitasi, dan Flow End-to-End TFT–SPEI

## Status dokumen

Dokumen ini adalah bahan revisi siap-tempel untuk Bab II dan Bab III. Isinya disusun dengan membandingkan:

1. Bab II–III pada `Draft PraSkripsi Nopal.pdf`;
2. `bab3_nopal_revisi.md`;
3. `REVISI_RUMUS.md`;
4. implementasi aktual pada `src/data/spei.py`, `src/models/tft.py`, dan `src/evaluation/calibration.py`.

**Batas penting:** arsitektur TFT, QuantileLoss, LSTM, attention, dan optimizer bukan kontribusi algoritmik baru dalam penelitian ini. Kontribusi penelitian ditulis sebagai penerapan dan integrasi pipeline pada SPEI-3, lima wilayah Jawa Timur, multi-horizon 30 hari, fitur spasial-temporal, serta kalibrasi interval per lokasi. Klaim “novel” hanya boleh digunakan jika ada eksperimen pembanding yang membuktikannya.

---

# 1. Jawaban audit utama

## 1.1 Apakah flow TFT pada draft sudah end-to-end?

Belum sepenuhnya. Draft sudah menyebut komponen utama TFT, yaitu Variable Selection Network (VSN), Gated Residual Network (GRN), LSTM encoder-decoder, attention, dan quantile output. Namun, urutan aliran data dari input sampai keluaran belum dijelaskan secara operasional.

Kekurangan utama draft:

- belum memetakan setiap variabel ke kategori `static`, `time-varying known`, dan `time-varying unknown`;
- belum menjelaskan bagaimana data encoder 90 hari dan decoder 30 hari dibentuk;
- belum menjelaskan bahwa fitur kategorikal dan fitur kontinu diproses melalui mekanisme input/normalisasi dataset TFT;
- belum menjelaskan urutan VSN, static context, recurrent processing, temporal attention, gating/residual, dan output projection;
- belum menghubungkan keluaran model berbentuk tiga kuantil pada setiap horizon dengan kalibrasi per kota;
- belum membedakan komponen TFT standar dari keputusan konfigurasi penelitian;
- belum mendokumentasikan formula kalibrasi yang benar-benar dipakai di kode.

## 1.2 Prinsip penulisan rumus

Algoritma standar tidak perlu diturunkan menjadi seluruh persamaan internalnya. Cukup:

1. sebutkan nama algoritma;
2. jelaskan fungsinya dalam pipeline;
3. berikan satu rumus inti jika rumus tersebut membantu pembaca memahami input/output;
4. berikan sitasi primer atau dokumentasi resmi.

Rumus wajib ditulis apabila rumus tersebut merupakan definisi target, aturan seleksi, transformasi data, konfigurasi khusus, atau prosedur yang diperlukan untuk mereproduksi penelitian.

---

# 2. Flow penelitian end-to-end

```text
Data Open-Meteo harian, 9 node/kota
        |
        v
Validasi skema, urutan waktu, forward fill maksimum 7 hari
        |
        v
Seleksi 5 node dari 9 node, train-only sampai 31-12-2022
        |
        v
Agregasi 5 node terpilih menjadi 1 super-node/kota
        |
        v
Defisit air D_t = P_t - ET0_t
        |
        v
Rolling accumulation 90 hari, fitting Fisk per bulan dengan `fit_mask` train-only jika dipasok oleh pemanggil
        |
        v
Transformasi ke SPEI, termasuk deep-tail extrapolation pada implementasi
        |
        v
Feature engineering: log presipitasi, SPEI difference, month sin/cos
        |
        v
Z-score scaling menggunakan statistik train-only
        |
        v
TimeSeriesDataSet: encoder 90 hari, decoder/prediction length 30 hari
        |
        v
Pemisahan fitur: static / known future / observed past
        |
        v
TFT standar dari PyTorch Forecasting
  - input embedding/normalisasi
  - Variable Selection Network
  - static context dan Gated Residual Network
  - LSTM encoder-decoder
  - interpretable temporal self-attention
  - gating/residual connections
  - quantile output layer
        |
        v
Output: 30 horizon x {q=0.10, 0.50, 0.90}
        |
        v
Kalibrasi lebar interval per kota memakai validation 2023
        |
        v
Evaluasi test: RMSE, MAE, R2, Pearson r, PICP
        |
        v
Artefak prediksi/evaluasi -> FastAPI JSON -> frontend visualization
```

## 2.1 Redaksi siap-tempel untuk Bab III

> Pipeline penelitian dimulai dari pengumpulan data meteorologi harian pada sembilan kandidat node untuk setiap wilayah. Setelah validasi skema dan penanganan nilai hilang, lima node dipilih menggunakan skor gabungan kemiripan perilaku iklim dan kedekatan spasial. Proses seleksi hanya menggunakan data sampai 31 Desember 2022 agar data validasi dan pengujian tidak memengaruhi pembentukan super-node. Lima node terpilih kemudian diagregasikan menjadi satu super-node untuk setiap wilayah.
>
> Pada setiap super-node, ET0 FAO Penman–Monteith digunakan sebagai proksi PET dalam neraca air. Defisit air dihitung sebagai selisih antara presipitasi dan ET0. Defisit tersebut diakumulasikan dalam rolling window 90 hari untuk membentuk target SPEI-3 pada resolusi harian. Selanjutnya dibentuk fitur logaritmik presipitasi, perubahan harian SPEI-3, serta representasi musiman sinus-kosinus. Parameter standardisasi kovariat kontinu dihitung hanya dari periode pelatihan; target `SPEI_3` memakai `EncoderNormalizer` berbasis histori encoder.
>
> Dataset forecasting menggunakan 90 hari historis sebagai encoder dan 30 hari masa depan sebagai prediction window. Input dibagi menjadi fitur statis, fitur yang diketahui pada masa depan, dan fitur yang hanya tersedia pada masa lalu. Dataset tersebut diberikan kepada implementasi Temporal Fusion Transformer dari PyTorch Forecasting. Secara konseptual, TFT melakukan seleksi variabel, pemrosesan temporal melalui LSTM encoder-decoder, pemodelan dependensi temporal melalui interpretable self-attention, pemfilteran informasi melalui gating dan residual connection, kemudian memproyeksikan representasi decoder menjadi tiga keluaran kuantil pada setiap horizon. Pada konfigurasi penelitian, kuantil yang digunakan adalah 0,10, 0,50, dan 0,90, sehingga keluaran model berisi 30 horizon dan tiga kuantil per horizon [S1][S2].
>
> Setelah inferensi, P50 digunakan sebagai prediksi median. Interval awal dibentuk dari P10 dan P90, kemudian lebarnya disesuaikan secara terpisah untuk setiap kota menggunakan prediksi validasi tahun 2023. Penyesuaian ini disebut *conformal-style per-city interval calibration*, bukan conformal prediction formal, karena implementasi yang digunakan adalah faktor pengali lebar interval berbasis kuantil residual ternormalisasi dan belum mendemonstrasikan jaminan coverage conformal formal [S3].

---

# 3. Tabel audit rumus dan sitasi

| Komponen | Status pada PDF | Status implementasi | Keputusan rumus | Rekomendasi penulisan dan sitasi |
|---|---|---|---|---|
| Forward fill | Sudah, Pers. 3.1 | `ffill(limit=7)` per node | Cukup satu baris | Jelaskan batas 7 hari dan train/temporal safety; tidak perlu teori panjang |
| Haversine | Belum konsisten | Ada di `ingest.py`/`preprocess.py` | Tulis | Tulis rumus karena dipakai dalam skor spasial. Sitasi [S4] |
| Behavior similarity | Naratif | Ada pada revisi metode | Tulis | Definisikan korelasi Pearson rata-rata, leave-one-node-out, jumlah variabel, dan penanganan varians nol |
| Hybrid score | Sebagian/naratif | Bobot 0,7 dan 0,3 | **Wajib tulis** | Ini keputusan pipeline: `0,7 behavior + 0,3 distance`; jangan menyebut bobot sebagai standar literatur tanpa sumber eksperimen |
| Seleksi train-only | Disebut | Sampai 31-12-2022 | Tidak perlu rumus | Wajib tulis batas tanggal dan alasan mencegah leakage |
| Agregasi super-node | Sudah, Pers. 3.2 | Rata-rata lima node | Tulis | Rumus reproduktif; jelaskan `K=5` |
| Z-score | Tidak lengkap | Train-only | Tulis | `z=(x−μ_train)/σ_train`; jelaskan parameter tidak dihitung dari validasi/test |
| Defisit air | Sudah, Pers. 3.3 | `P−ET0` | Tulis | Rumus definisi SPEI [S5] |
| Rolling SPEI-3 | Sudah, Pers. 3.4 | `scale*30`, jadi 90 hari | Tulis dan beri batasan | Nyatakan ini resolusi harian pendekatan 30 hari/bulan, bukan klaim bahwa semua definisi SPEI-3 harus 90 hari |
| Fisk/log-logistic | Sebagian | `scipy.stats.fisk.fit(..., floc=0)` per bulan | Tulis ringkas | Sitasi metode SPEI [S5][S6]; detail API `scipy.stats` cukup di metode implementasi |
| Shift domain Fisk | Ada di revisi rumus | Shift dari minimum data fitting | Tulis | Jelaskan shift train-only dan konsistensi transformasi; jangan menyatakan ini formula asli SPEI |
| Deep-tail extrapolation | Ada di kode/revisi rumus | Ada di `src/data/spei.py` | **Wajib tulis sebagai modifikasi implementasi** | Jelaskan sebagai ekstrapolasi heuristik monotonik pada ekor di bawah ambang kuantil `eps=10^-6`; jangan menyebutnya bagian standar SPEI tanpa sitasi khusus |
| Kategori SPEI | Sudah | Ada `classify_spei` | Cukup tabel | Sitasi WMO/McKee hanya jika sumber benar-benar dimasukkan; konsistenkan interval batas |
| Log presipitasi | Disebut tanpa rumus | Ada pada revisi metode | Tulis | `ln(1+P)`; jelaskan alasan transformasi, tanpa klaim universal |
| SPEI difference | Disebut tanpa rumus | Ada pada revisi metode | Tulis | `ΔSPEI_t=SPEI_t−SPEI_{t−1}` |
| Month sin/cos | Sudah | Ada | Tulis | Rumus sederhana dan sumber opsional; yang penting definisi periode 12 bulan |
| Sequence | Sudah, Pers. 3.8–3.9 | Encoder 90, prediction 30 | Tulis | Hubungkan langsung dengan konfigurasi dataset |
| Input categories | Naratif | Ada dataset schema | Tabel wajib, rumus tidak | Cantumkan nama fitur dan ketersediaan masa depan |
| Feature embedding/normalisasi TFT | Belum end-to-end | Bawaan `from_dataset` | Cukup mention | Jangan menulis ulang seluruh persamaan embedding; rujuk [S1][S2] |
| VSN | Disebut; formula terlalu umum | Bawaan TFT | Satu formula opsional | Jelaskan fungsi weighted feature selection; sitasi [S1] |
| GRN | Disebut; formula terlalu sederhana | Bawaan TFT | Satu formula opsional | Nyatakan formula berasal dari arsitektur TFT, bukan novelty penelitian [S1] |
| LSTM encoder-decoder | Disebut | Bawaan TFT | Cukup mention atau satu formula | Tidak perlu menurunkan semua gate LSTM; sitasi [S1] |
| Self-attention | Sudah dua kali | Bawaan TFT | Pertahankan satu formula | Hapus duplikasi Pers. 2.4/3.11; sitasi [S1] |
| Gating/residual | Naratif | Bawaan TFT | Cukup mention | Jelaskan fungsinya, bukan mengeklaim modifikasi |
| Quantile output | Sudah | `output_size=3`, quantiles `[.1,.5,.9]` | Tulis output shape | Jelaskan konfigurasi implementasi, sitasi [S1][S2] |
| Quantile loss | Sudah dua kali | `QuantileLoss` | Pertahankan satu formula | Hapus duplikasi Pers. 2.1/3.12; sitasi [S1][S2] |
| Adam, dropout, weight decay | Naratif | Konfigurasi kode | Tidak perlu rumus | Cukup tabel hyperparameter |
| Early stopping/gradient clipping | Naratif | Training pipeline | Tidak perlu rumus | Jelaskan parameter aktual jika tersedia |
| Direct multi-horizon | Sudah | Prediction length 30 | Cukup formula output | Fokus pada `ŷ_{t+1:t+30}` |
| Kalibrasi interval per kota | Belum memadai | Ada `calibration.py` | **Wajib tulis** | Tulis formula residual ternormalisasi dan faktor kuantil; sebut *conformal-style*, bukan CQR formal [S3] |
| PICP | Sudah | Ada evaluasi | Cukup satu formula | Definisikan nominal 80%; jangan menyamakan PICP validasi dengan jaminan coverage populasi |
| RMSE/MAE/R²/Pearson | Sudah | Ada | Cukup mention/formula ringkas | Tidak perlu memenuhi Bab II dengan rumus metrik standar |
| Naive persistence | Sudah | Ada pembanding | Satu formula | `ŷ_{t+h}=y_t`; cukup mention |
| VSN/attention interpretability | Naratif | Output interpretability | Tidak perlu rumus tambahan | Wajib jelaskan cara bobot diekstrak/diagregasi pada eksperimen |
| Vue/FastAPI | Sudah | Backend aktual menyebut FastAPI dan frontend aktual React/Vite | Tidak perlu rumus | **Koreksi:** jangan menulis Vue.js jika frontend aktual React/Vite; gunakan dokumentasi kode |

---

# 4. Rumus yang disarankan masuk ke Bab III

## 4.1 Seleksi node

### Jarak Haversine

Untuk kandidat node `i` dan pusat wilayah, setelah lintang/bujur dikonversi ke radian:

\[
 a_i=\sin^2\left(\frac{\Delta\varphi_i}{2}\right)+\cos(\varphi_c)\cos(\varphi_i)\sin^2\left(\frac{\Delta\lambda_i}{2}\right)
\]
\[
 d_i=2R\arcsin(\sqrt{a_i})
\]

dengan `R=6371 km`. Skor jarak implementasi:

\[
\operatorname{distance\_score}_i=\frac{1}{1+d_i}.
\]

Rumus jarak adalah rumus standar, bukan novelty. Sitasi [S4].

### Behavior score

\[
\operatorname{behavior\_score}_i
=\frac{1}{M}\sum_{m=1}^{M}
\rho\left(\mathbf v_{i,m},\mathbf v_{-i,m}\right)
\]

`M=8` adalah jumlah variabel meteorologi. `v_{-i,m}` adalah profil rata-rata delapan node lain. Jika salah satu simpangan baku nol, korelasi variabel tersebut dilewati agar tidak terjadi pembagian dengan nol. Jika seluruh variabel menghasilkan korelasi tidak valid, implementasi menetapkan `behavior_score=-1.0`.

### Hybrid score

\[
\operatorname{hybrid\_score}_i
=0.7\operatorname{behavior\_score}_i
+0.3\operatorname{distance\_score}_i.
\]

Lima kandidat dengan skor tertinggi dipilih secara deterministik. Bobot 0,7/0,3 adalah konfigurasi penelitian, bukan konstanta universal. Jika bobot tidak berasal dari optimasi/ablation study, tulis “ditetapkan sebagai konfigurasi penelitian”, bukan “bobot optimal”.

## 4.2 Agregasi super-node

\[
SN_{t,m}=\frac{1}{K}\sum_{k=1}^{K}x_{t,m,k},\qquad K=5.
\]

Operasi ini adalah rata-rata aritmetika node terpilih pada waktu `t` untuk variabel `m`.

## 4.3 Standardisasi train-only

\[
z_t=\frac{x_t-\mu_{train}}{\sigma_{train}}.
\]

Untuk kovariat kontinu, `μ_train` dan `σ_train` dihitung hanya pada data pelatihan. Statistik tersebut kemudian digunakan tanpa dihitung ulang pada validasi dan pengujian. Target `SPEI_3` menggunakan `EncoderNormalizer` berbasis histori encoder pada `TimeSeriesDataSet`, bukan scaler global kovariat.

## 4.4 SPEI-3 resolusi harian

Defisit air:

\[
D_t=P_t-ET0_t.
\]

Akumulasi rolling:

\[
X_t^{(3)}=\sum_{j=0}^{89}D_{t-j}.
\]

Implementasi mengonversi skala `3` menjadi `3×30=90` hari. Nyatakan secara eksplisit bahwa ini adalah operasionalisasi resolusi harian penelitian. Metode SPEI asli menggunakan neraca air dan standardisasi distribusi pada skala waktu tertentu [S5][S6].

Fungsi `calculate_spei()` mendukung fitting train-only melalui parameter `fit_mask`; penggunaan train-only harus dipastikan pada pemanggil fungsi dan dicatat dalam metode run final. Distribusi Fisk dipasang per bulan kalender pada data fitting. Secara umum:

\[
SPEI_t=\Phi^{-1}\left(F_{Fisk}(X_t^{(3)})\right),
\]

setelah transformasi domain yang diperlukan oleh implementasi. Jangan menyatakan bahwa persamaan ringkas tersebut sudah menjelaskan seluruh `scipy.stats.fisk.fit`; parameter fitting dan periode fitting harus disebutkan dalam narasi.

### Shift domain dan deep tail

Jika data fitting memiliki nilai nonpositif, implementasi menggunakan shift yang dihitung terpisah untuk setiap bulan kalender `m`:

\[
sm_t=X_t^{(3)}+s_m,
\]

\[
s_m=\begin{cases}
|\min(X_{fit,m}^{(3)})|+1,&\min(X_{fit,m}^{(3)})\le0\\
0,&\text{lainnya.}
\end{cases}
\]

dengan `X_{fit,m}^{(3)}` sebagai nilai rolling yang valid pada bulan `m` dan subset fitting yang digunakan implementasi. Parameter Fisk juga difit terpisah untuk setiap bulan dengan `fisk.fit(..., floc=0)`; parameter skala yang dipakai pada ekstrapolasi adalah `β_m = params[-1]`.

Untuk nilai pada ekor bawah di bawah ambang kuantil `eps=10^{-6}`, ketika CDF Fisk mengalami saturasi numerik, implementasi mempertahankan urutan nilai dengan ekstrapolasi linear:

\[
SPEI_t=z_{floor}+\frac{sm_t-x_{\epsilon,m}}{\beta_m},
\qquad sm_t\le x_{\epsilon,m}.
\]

Bagian ini adalah keputusan implementasi penelitian. Jangan menyebut deep-tail extrapolation sebagai formula SPEI standar dari [S5].

## 4.5 Feature engineering

\[
precipitation\_log_t=\ln(1+P_t)
\]

\[
\Delta SPEI_t=SPEI_t-SPEI_{t-1}
\]

\[
month\_sin_t=\sin\left(\frac{2\pi m_t}{12}\right),\qquad
month\_cos_t=\cos\left(\frac{2\pi m_t}{12}\right).
\]

## 4.6 Pembentukan sequence

\[
X_t=\{x_{t-89},x_{t-88},\ldots,x_t\}
\]
\[
Y_t=\{y_{t+1},y_{t+2},\ldots,y_{t+30}\}.
\]

Dengan demikian, satu sampel memuat 90 hari historis dan target 30 hari berikutnya.

## 4.7 Quantile output dan loss

Model dibangun dengan `output_size=3` dan kuantil `[0.10,0.50,0.90]` pada `src/models/tft.py`. Quantile loss dapat ditulis sekali:

\[
L_q(y,\hat y_q)=\max\left(q(y-\hat y_q),(q-1)(y-\hat y_q)\right).
\]

Persamaan ini adalah loss standar, bukan novelty penelitian. Detail internal output TFT cukup dijelaskan sebagai tiga prediksi kuantil untuk setiap dari 30 horizon [S1][S2].

**Koreksi interpretasi:** untuk target SPEI, nilai yang lebih kecil berarti kondisi lebih kering. Karena itu, jangan menulis secara otomatis bahwa “P10 adalah skenario basah dan P90 skenario kering” atau sebaliknya tanpa memeriksa orientasi target dan hasil model. Tulis lebih aman: “P10, P50, dan P90 adalah kuantil distribusi prediksi; implikasi basah/kering ditentukan dari nilai SPEI dan arah ambang kekeringan.”

## 4.8 Kalibrasi interval aktual

Kode aktual menggunakan data validasi per kota. Faktor dihitung secara pooled per kota, bukan terpisah untuk setiap horizon. Untuk observasi `j`:

\[
h_j=\frac{\hat y_{0.90,j}-\hat y_{0.10,j}}{2}
\]

\[
r_j=\frac{|y_j-\hat y_{0.50,j}|}{h_j},\qquad h_j>0.
\]

Faktor kota `c_g` dihitung sebagai kuantil 0,80 dari `r_j` pada kota `g`:

\[
c_g=\max\left(\operatorname{Quantile}_{0.80}\{r_j:j\in g\},0.5\right).
\]

Interval terkalibrasi:

\[
\hat y^{cal}_{0.10}=\hat y_{0.50}-c_g h,
\qquad
\hat y^{cal}_{0.90}=\hat y_{0.50}+c_g h.
\]

`P50` tidak diubah. Jika jumlah observasi valid kurang dari lima, kode menggunakan `c_g=1.0`.

Penulisan yang benar:

> Penelitian menerapkan kalibrasi interval bergaya conformal (*conformal-style*) secara per kota. Faktor kalibrasi dihitung dari kuantil residual absolut yang dinormalisasi oleh setengah lebar interval pada data validasi. Prosedur ini digunakan untuk menyesuaikan lebar interval terhadap heterogenitas lokasi. Faktor dihitung secara pooled per kota, bukan per horizon, dan kuantil yang digunakan adalah kuantil empiris biasa. Karena prediksi deret waktu dan horizon yang tumpang tindih tidak otomatis independen atau *exchangeable*, PICP pada data uji diperlakukan sebagai evaluasi empiris, bukan jaminan coverage conformal. Prosedur ini tidak disebut sebagai Conformalized Quantile Regression formal karena implementasi tidak menggunakan skor CQR standar [S3].
>
> QuantileLoss tidak dengan sendirinya menjamin `P10 ≤ P50 ≤ P90` pada setiap sampel. Oleh karena itu, pipeline perlu memeriksa quantile crossing; baris dengan setengah lebar interval tidak positif tidak digunakan dalam fitting kalibrasi.

---

# 5. Rekomendasi penulisan per section

## 5.1 Bab II — Landasan Teori

### Subbab SPEI

Pertahankan:

- definisi SPEI sebagai indeks berbasis neraca `P−PET`;
- keunggulan memasukkan evapotranspirasi;
- distribusi log-logistic/Fisk;
- alasan penggunaan skala tiga bulan.

Tambahkan batasan:

> Pada penelitian ini, SPEI-3 dihitung dari data harian dengan rolling window 90 hari. Operasionalisasi tersebut digunakan untuk kebutuhan forecasting harian, sehingga disebut sebagai SPEI-3 resolusi harian dan tidak disamakan begitu saja dengan seluruh implementasi SPEI bulanan pada literatur [S5][S6].

### Subbab TFT

Gunakan urutan berikut:

1. tujuan desain TFT;
2. tipe input TFT;
3. VSN;
4. static context dan GRN;
5. LSTM encoder-decoder;
6. interpretable attention;
7. gating/residual;
8. quantile output;
9. interpretability output.

Tambahkan kalimat batas kontribusi:

> Komponen-komponen tersebut merupakan bagian dari arsitektur TFT yang dirujuk dari Lim et al. [S1]. Penelitian ini tidak mengusulkan arsitektur TFT baru, melainkan mengonfigurasi dan menerapkannya pada peramalan SPEI-3 multi-horizon dengan fitur spasial-temporal dan keluaran kuantil.

### Subbab probabilistic forecasting

Hapus penyebutan “P10 skenario basah” dan “P90 skenario kering” yang tidak diberi syarat. Gunakan istilah kuantil bawah, median, dan kuantil atas. Interpretasi kekeringan dilakukan setelah membandingkan nilai output dengan ambang SPEI.

### Subbab kalibrasi

Jelaskan perbedaan:

- quantile forecasting: menghasilkan kuantil dari model;
- calibration: menyesuaikan interval agar coverage validasi lebih dekat dengan nominal;
- conformal prediction formal: memerlukan prosedur dan asumsi khusus.

Rujukan CQR [S3] dipakai sebagai pembanding teori, bukan sebagai klaim bahwa kode penelitian telah mengimplementasikan CQR formal.

## 5.2 Bab III — Metode Penelitian

Tambahkan subbab **Perbedaan algoritma standar dan keputusan penelitian** setelah uraian model TFT:

| Jenis | Komponen | Cara tulis |
|---|---|---|
| Standar referensi | TFT, VSN, GRN, LSTM, attention, QuantileLoss | Nama, fungsi, satu rumus inti bila perlu, [S1][S2] |
| Pipeline penelitian | 9 node → 5 node, hybrid score 0,7/0,3, super-node | Rumus lengkap dan parameter |
| Target penelitian | rolling 90 hari, Fisk per bulan, train-only fitting | Rumus dan periode fitting |
| Feature engineering | log presipitasi, difference, sin/cos | Rumus singkat |
| Kalibrasi penelitian | faktor residual per kota | Rumus lengkap dan kode aktual |
| Evaluasi standar | RMSE, MAE, R2, Pearson, PICP | Definisi singkat; tidak perlu derivasi panjang |

Tambahkan subbab **Pemetaan input TFT**:

| Kelompok input | Variabel | Tersedia pada decoder? |
|---|---|---|
| Static categorical / group ID | `super_node_id` saja | Ya, konstan |
| Static real | `elevation`, `lat`, `lon` | Ya, konstan |
| Time-varying known | `time_idx`, `month_sin`, `month_cos` | Ya, jika dibentuk untuk 30 hari |
| Time-varying unknown | cuaca historis, `water_deficit`, `SPEI_3`, `SPEI_3_diff` | Hanya sampai waktu observasi |
| Target | `SPEI_3` masa depan | Label, bukan fitur masa depan yang diketahui |

**Catatan implementasi:** pada konfigurasi aktual, `super_node_id` adalah satu-satunya static categorical/group ID. Kovariat kontinu memakai scaler dataset, sedangkan target memakai `EncoderNormalizer`. Tabel ini harus tetap dicocokkan dengan konfigurasi `TimeSeriesDataSet` pada run final. Jangan menyatakan suatu fitur sebagai known future jika pipeline tidak benar-benar mengisinya pada decoder.

---

# 6. Sitasi terverifikasi dan cara pakainya

## [S1] Artikel primer TFT

Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*. **International Journal of Forecasting, 37**(4), 1748–1764. https://doi.org/10.1016/j.ijforecast.2021.03.012

Gunakan untuk: desain TFT, VSN, GRN, LSTM encoder-decoder, gating, interpretable attention, multi-horizon, quantile forecasting, dan interpretability.

## [S2] Dokumentasi PyTorch Forecasting

PyTorch Forecasting. (n.d.). *TemporalFusionTransformer API* dan *Demand forecasting with the Temporal Fusion Transformer*. Dokumentasi resmi: https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html dan https://pytorch-forecasting.readthedocs.io/en/v1.4.0/tutorials/stallion.html

Gunakan untuk: API implementasi, `from_dataset`, `output_size`, `QuantileLoss`, kategori input, encoder length, prediction length, dan pola penggunaan `TimeSeriesDataSet`. Sitasi ini tidak menggantikan sitasi artikel primer [S1].

## [S3] Conformalized Quantile Regression sebagai pembanding

Romano, Y., Patterson, E., & Candès, E. J. (2019). *Conformalized Quantile Regression*. **Advances in Neural Information Processing Systems 32**, 3538–3548. https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html

Gunakan untuk menjelaskan teori CQR dan membedakannya dari kalibrasi faktor lebar interval aktual. Jangan menyatakan implementasi penelitian sebagai CQR formal tanpa penyesuaian algoritma dan bukti coverage.

## [S4] Rumus Haversine

Sinnott, R. W. (1984). *Virtues of the Haversine*. **Sky & Telescope, 68**(2), 159. Metadata bibliografis dapat ditelusuri melalui https://www.semanticscholar.org/paper/Virtues-of-the-Haversine-Sinnott/d1761591716859275573d4d315c973f2dbc26eae

Gunakan untuk: rumus jarak great-circle. Rumus ini standar, bukan kontribusi baru.

## [S5] Artikel primer SPEI

Vicente-Serrano, S. M., Beguería, S., & López-Moreno, J. I. (2010). *A multi-scalar drought index sensitive to global warming: The standardized precipitation evapotranspiration index*. **Journal of Climate, 23**(7), 1696–1718. https://doi.org/10.1175/2009JCLI2909.1

Gunakan untuk: neraca air `P−PET`, gagasan SPEI multiskala, sensitivitas terhadap evapotranspirasi, dan standardisasi berbasis distribusi.

## [S6] Tinjauan metode SPEI

Beguería, S., Vicente-Serrano, S. M., Reig, F., & Latorre, B. (2014). *Standardized precipitation evapotranspiration index (SPEI) revisited: Parameter fitting, evapotranspiration models, tools, datasets and drought monitoring*. **International Journal of Climatology, 34**(10), 3001–3023. https://doi.org/10.1002/joc.3887

Gunakan untuk: parameter fitting, model evapotranspirasi, tooling, dan isu implementasi SPEI.

## Aturan nomor sitasi

- Jika daftar pustaka skripsi memakai sistem angka, nomor `[S1]–[S6]` harus dipetakan ke nomor daftar pustaka final setelah daftar pustaka disusun ulang.
- Jangan langsung mengubah `[S1]` menjadi `[24]` sebelum nomor final dipastikan.
- Entri [S4] tidak memiliki DOI yang terverifikasi pada pemeriksaan ini; gunakan hanya metadata/URL tersebut atau ganti dengan sumber GIS akademik yang benar-benar dimiliki penulis.
- Sitasi `Hasanah & Suharso (2023)`, `Maria et al. (2020)`, `Tang & Zhao (2025)`, serta sumber [34]–[35] pada PDF tidak dipakai di dokumen ini karena metadata lengkap dan relevansinya belum diverifikasi dari sumber primer.

---

# 7. Koreksi wajib sebelum masuk naskah final

1. **Frontend:** implementasi proyek yang dibaca memakai React berbasis Vite dan TypeScript, bukan Vue.js. Ganti semua “Vue.js” dan “antarmuka Vue.js” pada Bab II dan Bab III menjadi “React berbasis Vite” dan “antarmuka React”.
2. **P10/P90:** jangan memberi label basah/kering hanya dari nomor kuantil. SPEI lebih negatif berarti lebih kering.
3. **SPEI harian:** nyatakan secara eksplisit bahwa rolling 90 hari adalah operasionalisasi `scale*30` pada kode.
4. **ET0/PET:** pilih satu istilah utama. Tulis bahwa `et0_fao_evapotranspiration` dipakai sebagai proksi PET pada neraca air.
5. **Kalibrasi:** gunakan istilah “conformal-style” atau “post-hoc per-city interval calibration”, bukan “CQR” formal.
6. **Novelty:** jangan menyebut TFT, VSN, GRN, LSTM, attention, atau QuantileLoss sebagai novelty.
7. **Hybrid score:** bobot 0,7/0,3 adalah konfigurasi penelitian. Klaim “optimal” hanya boleh dipakai jika ada ablation/optimasi.
8. **Daftar wilayah:** konsistenkan Bojonegoro, Lamongan, Nganjuk, Ngawi, dan Tuban di semua bab.
9. **Nomor gambar/tabel:** PDF memiliki beberapa nomor tabel/gambar yang berulang/bergeser. Renumber setelah struktur final.
10. **Data leakage:** jelaskan bahwa node selection, scaler, dan parameter distribusi memakai data train-only; kalibrasi memakai validasi; evaluasi akhir memakai test.
11. **Interpretabilitas:** jelaskan bagaimana bobot VSN dan attention diekstrak, diagregasi, lalu divisualisasikan. Bobot attention menunjukkan pola model, bukan bukti kausal.
12. **Kode vs naskah:** jangan menulis Vue.js, horizon 12 bulan, atau fitur lain jika konfigurasi aktual yang dipakai adalah React/Vite dan horizon 30 hari.

---

# 8. Checklist final sebelum diserahkan

- [ ] Flow raw data → TFT → calibration → evaluation → API/dashboard sudah satu arah dan lengkap.
- [ ] Setiap rumus pipeline penelitian memiliki definisi simbol.
- [ ] Rumus default TFT tidak diduplikasi di Bab II dan Bab III.
- [ ] Semua formula SPEI cocok dengan implementasi `src/data/spei.py`.
- [ ] Kalibrasi cocok dengan `src/evaluation/calibration.py`.
- [ ] Quantile ordering diverifikasi dari `model.loss.quantiles`.
- [ ] P10/P50/P90 tidak diberi interpretasi basah/kering secara terbalik.
- [ ] Sitasi [S1]–[S6] dapat ditelusuri ke DOI, penerbit, atau dokumentasi resmi.
- [ ] Sitasi yang belum terverifikasi tidak dimasukkan.
- [ ] Frontend pada naskah cocok dengan implementasi aktual.
- [ ] Tidak ada klaim novelty tanpa eksperimen pembanding.
- [ ] Nomor sitasi sementara `[S1]–[S6]` dipetakan ke daftar pustaka final.

---

# 9. Hasil audit sub-agent

Bagian ini diisi setelah lima pemeriksaan independen:

1. Audit kelengkapan flow TFT: **lulus dengan koreksi** — shift Fisk harus bulanan; static categorical hanya `super_node_id`; target memakai `EncoderNormalizer`.
2. Audit rumus terhadap kode: **lulus dengan koreksi** — deep-tail dipicu ambang kuantil `eps`, bukan “di luar support”; shift dan fitting Fisk per bulan; konfigurasi encoder/prediction adalah 90/30 pada run utama.
3. Audit verifikasi sitasi: **lulus terbatas** — [S1], [S2], [S3], [S5], [S6] terverifikasi; [S4] hanya metadata historis, URL tidak diperlakukan sebagai sumber stabil.
4. Audit klaim novelty dan kalibrasi: **lulus dengan koreksi** — tidak ada novelty algoritmik; kalibrasi pooled per kota, kuantil empiris, tanpa jaminan conformal; tambahkan pemeriksaan quantile crossing.
5. Audit bahasa dan struktur: **lulus dengan koreksi** — konsistenkan ET0 sebagai proksi PET, dokumentasikan fallback `behavior_score=-1.0`, ganti Vue.js menjadi React berbasis Vite, dan hindari klaim “optimal” tanpa ablation.

Lima audit independen selesai. Temuan koreksi kritis sudah dimasukkan ke dokumen ini; pemeriksaan akhir tetap diperlukan sebelum menyalin ke naskah skripsi.
