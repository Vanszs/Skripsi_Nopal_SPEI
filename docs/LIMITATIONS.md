# Keterbatasan Penelitian

Dokumen ini menyajikan keterbatasan nyata dari pipeline peramalan SPEI-3 menggunakan
Temporal Fusion Transformer (TFT) dalam skripsi *"Peramalan Multi-Horizon Indeks
Kekeringan Lahan Pertanian (SPEI) di Sentra Padi Jawa Timur Menggunakan Temporal
Fusion Transformer"*. Setiap poin didasarkan pada verifikasi langsung terhadap kode
sumber dan konfigurasi aktual.

---

## 1. Jumlah Entitas Spasial yang Terbatas (5 Kabupaten)

Model TFT dalam penelitian ini dilatih pada **hanya 5 entitas spasial**
(`super_node_id`), masing-masing merepresentasikan satu kabupaten sentra padi:
Lamongan, Ngawi, Bojonegoro, Tuban, dan Nganjuk. Kelima kabupaten diagregasi menjadi
satu super-node per kota (`src/models/dataset.py`, `group_ids=[super_node_id]`).

Jumlah entitas yang sangat kecil ini menempatkan model dalam **rezim data terbatas**
(*data-limited regime*). Model deep learning seperti TFT, yang memiliki ratusan ribu
parameter, sangat rentan terhadap *overfitting* ketika jumlah grup pelatihan sedikit.
Temuan empiris dalam penelitian ini mengkonfirmasi hal tersebut:

- Peningkatan kapasitas model (misalnya `hidden_size=64`) menyebabkan **overfitting
  sejak epoch 0** — *validation loss* langsung lebih tinggi dari *training loss* tanpa
  pernah membaik (didokumentasikan dalam `src/models/tft.py`, komentar pada parameter
  `hidden_size`).
- Konfigurasi akhir yang stabil memerlukan regularisasi kuat: `dropout=0.35`,
  `weight_decay=1e-4`, `hidden_size=48`, `attention_head_size=1`,
  `hidden_continuous_size=8` (terverifikasi di `src/models/tft.py::build_tft_model()`
  dan `logs/run_config.json`).

**Implikasi:** Hasil penelitian ini berlaku untuk 5 kabupaten tersebut. Generalisasi
ke wilayah lain memerlukan validasi ulang. Penambahan entitas spasial di masa depan
berpotensi meningkatkan kapasitas generalisasi model.

---

## 2. Ekstrapolasi Ekor Distribusi SPEI (Kekeringan Ekstrem)

Perhitungan SPEI menggunakan distribusi log-logistik (*fisk*) yang di-*fit* pada data
pelatihan saja (leakage-safe, `src/data/spei.py`). Untuk nilai akumulasi defisit air
yang **berada di bawah support distribusi yang ter-fit** (yaitu lebih kering dari
seluruh rekam historis pelatihan), pipeline melakukan **ekstrapolasi linier dalam
ruang-z** alih-alih melakukan *hard clamp* (baris 85–95 `src/data/spei.py`):

```python
# Deep-tail extrapolation: z_floor + (sm[below] - x_eps) / scale
z_score[below] = z_floor + (sm[below] - x_eps) / scale
```

Pendekatan ini mempertahankan **urutan monoton** (nilai lebih kering tetap menghasilkan
SPEI lebih rendah) dan menghindari kolapsnya seluruh ekor kering ke satu nilai tunggal.
Namun, nilai SPEI di bawah sekitar −3 merupakan **ekstrapolasi di luar support
distribusi yang ter-fit** dan harus diinterpretasikan dengan hati-hati:

- Nilai tersebut mengindikasikan kondisi **"ekstrem/belum pernah terjadi"** (*unprecedented*),
  bukan magnitudo yang terkalibrasi secara presisi terhadap distribusi probabilistik.
- Semakin jauh dari batas support, semakin besar ketidakpastian kuantitatif.
- Untuk keperluan klasifikasi kekeringan (misalnya SPEI ≤ −2.0 = "Kekeringan Ekstrem"),
  ambang batas standar WMO tetap berlaku, tetapi perbedaan antara SPEI −3.5 dan −4.0
  tidak memiliki interpretasi probabilistik yang sama ketatnya dengan perbedaan antara
  SPEI −1.0 dan −1.5.

---

## 3. Pembagian Temporal dan Variabilitas Iklim Antar-Tahun

Pembagian data mengikuti skema kronologis ketat (terverifikasi di
`src/training/train.py`):

| Split      | Periode         | Keterangan                          |
|------------|-----------------|-------------------------------------|
| Training   | < 2023          | ~32.420 baris (2005–2022)           |
| Validation | 2023            | ~2.275 baris                        |
| Test       | ≥ 2024          | Data yang belum pernah dilihat model|

Pembagian ini realistis untuk skenario peramalan operasional (model tidak pernah
melihat masa depan). Namun, setiap tahun memiliki **rezim iklim yang berbeda** — tahun
2023 cenderung lebih kering (El Niño), sedangkan 2024+ cenderung lebih basah. Akibatnya:

- Metrik validasi mencerminkan performa pada tahun kering tertentu.
- Metrik test mencerminkan performa pada tahun basah tertentu.
- Evaluasi multi-tahun (misalnya *rolling-origin cross-validation*) akan memberikan
  estimasi performa yang lebih robust terhadap variabilitas iklim, namun tidak
  dilakukan dalam penelitian ini karena keterbatasan data dan waktu komputasi.

---

## 4. Variansi Antar-Seed (Stochastic Variance)

Pelatihan neural network bersifat stokastik — inisialisasi bobot, urutan mini-batch,
dan dropout menghasilkan model yang sedikit berbeda pada setiap *seed*. Untuk
transparansi:

- Hasil dilaporkan sebagai **mean ± std** dari beberapa *seed* (bukan satu *seed*
  tunggal yang dipilih karena hasilnya terbaik).
- Variasi antar-*seed* pada dataset sekecil ini bisa signifikan (beberapa persen pada
  RMSE/MAE).
- Pelaporan multi-*seed* digunakan untuk **kejujuran ilmiah** — menunjukkan rentang
  performa yang realistis, bukan titik optimistik tunggal.

---

## 5. Horizon Peramalan: Jangka Pendek–Menengah (30 Hari)

Model memprediksi **30 langkah waktu ke depan** secara langsung (*direct multi-horizon*,
`max_prediction_length=30` di `src/models/dataset.py`). Ini termasuk peramalan
**jangka pendek hingga menengah**, bukan jangka panjang (musiman/tahunan).

Karakteristik yang perlu dicatat:

- **Akurasi menurun seiring bertambahnya horizon** — ini adalah sifat inheren peramalan
  deret waktu. Kurva degradasi per-horizon (h=1 sampai h=30) dilaporkan dalam evaluasi.
- Horizon 30 hari relevan untuk perencanaan irigasi dan peringatan dini kekeringan
  jangka pendek, tetapi **tidak cukup** untuk perencanaan musim tanam jangka panjang
  yang memerlukan horizon 3–6 bulan.
- Encoder length 90 hari (3 bulan, selaras dengan skala temporal SPEI-3) memberikan
  konteks historis yang memadai untuk horizon 30 hari.

---

## 6. Protokol Kejujuran Evaluasi

Untuk menjamin integritas hasil, pipeline menerapkan protokol evaluasi berikut
(terverifikasi di kode sumber):

1. **Test set digunakan sekali saja di akhir** — tidak ada tuning hyperparameter,
   pemilihan model, atau iterasi berdasarkan metrik test (`full_evaluation.py`:
   `test_data = data[data.year >= 2024]`, dipanggil setelah model final terpilih).

2. **Kalibrasi interval prediksi di-fit hanya pada data validasi** — faktor pelebaran
   interval P10–P90 dihitung dari residual validasi per kota, kemudian diterapkan ke
   test tanpa melihat label test (`src/evaluation/calibration.py`:
   `fit_per_city_interval_calibration()` menerima `df_val` saja).

3. **Tidak ada metrik yang di-hardcode** — semua angka (RMSE, MAE, R², PICP, POD, FAR,
   CSI) dihitung secara dinamis dari prediksi aktual model terhadap ground truth.

4. **Pelaporan multi-seed** — menghindari *cherry-picking* satu hasil terbaik.

5. **Warmup encoder** — evaluasi pada test dan validasi menyertakan data warmup
   sepanjang `encoder_length` hari sebelum awal split, sehingga prediksi pertama
   memiliki konteks penuh (bukan prediksi tanpa sejarah).

---

## Ringkasan

| # | Keterbatasan | Dampak |
|---|---|---|
| 1 | Hanya 5 entitas spasial | Regularisasi kuat wajib; generalisasi terbatas |
| 2 | Ekstrapolasi ekor SPEI | Nilai < −3 bersifat indikatif, bukan presisi probabilistik |
| 3 | Split temporal satu tahun per split | Metrik terikat rezim iklim tahun tertentu |
| 4 | Variansi stokastik | Dilaporkan mean±std untuk transparansi |
| 5 | Horizon 30 hari | Jangka pendek–menengah; bukan peramalan musiman |
| 6 | Protokol evaluasi | Test sekali, kalibrasi pada val, tanpa hardcode |

---

*Dokumen ini dibuat sebagai bagian dari dokumentasi metodologi penelitian untuk
memenuhi prinsip transparansi dan reprodusibilitas ilmiah.*
