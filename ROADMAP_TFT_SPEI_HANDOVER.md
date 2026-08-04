# Handover ML: TFT untuk Pseudo-SPEI Harian

> **Scope:** hanya pipeline machine-learning proyek ini.  
> **Sumber kebenaran:** implementasi `src/`, script evaluasi, test, checkpoint, dan artifact lokal.  
> **Label:** **Fakta kode** = terbukti di source; **Batas implementasi** = belum dijamin source; **Tidak dibuktikan** = jangan diklaim tanpa experiment/artifact yang cocok.

Dokumen ini mandiri untuk pemula atau AI lain. Tujuannya memahami data, target, TFT, training, evaluasi, artifact, serta kapan perubahan mewajibkan retraining.

---

# 1. Jawaban singkat

Sistem adalah **forecasting indeks kekeringan harian** untuk lima kota Jawa Timur. Ia menerima riwayat cuaca harian, mengubahnya menjadi satu **super-node kota**, membangun target `SPEI_3` internal, lalu memakai **Temporal Fusion Transformer (TFT)** untuk memprediksi distribusi target sampai 30 hari.

```text
9 titik cuaca/kota
  -> pilih 5 titik berdasar data training
  -> rata-ratakan menjadi 1 kota/super-node
  -> water deficit
  -> rolling 90 hari + transform distribusi = pseudo-SPEI
  -> TFT: 90 hari encoder, 30 hari decoder
  -> P10 / P50 / P90 per hari masa depan
```

**Bukan:** graph neural network, RDM, STGNN, GAT, retrieval model, atau model spasial dengan message passing. Unsur spasial selesai sebelum TFT: top-5 node dipilih secara deterministik lalu dirata-ratakan.

## 1.1 Output

Untuk setiap `super_node_id` dan setiap horizon `h=1..30`, TFT memberi tiga quantile:

```text
P10  = batas bawah prediksi
P50  = median prediksi
P90  = batas atas prediksi
```

Output bukan satu angka pasti. P10/P50/P90 adalah ringkasan distribusi kondisional yang dipelajari melalui quantile loss.

## 1.2 Fakta performa terpenting

Artifact checkpoint aktif seed 44 menunjukkan TFT **kalah dari persistence pada t+1**:

```text
TFT t+1 RMSE       0.27685
naive t+1 RMSE     0.17306
TFT t+1 skill     -59.97%
```

Namun artifact yang sama melaporkan all-horizon skill `+33.61%`. Sumber: `results/seed44_eval/metrics_summary.json:19-37,60-66`.

Jangan menyebut model unggul untuk forecast hari pertama. Keunggulan yang tercatat hanya perlu dibahas bersama horizon, checkpoint, data, evaluator, dan baseline yang tepat.

---

# 2. Kamus zero-to-hero

| Istilah | Arti sederhana | Arti tepat di proyek |
|---|---|---|
| Drought/kekeringan | Kekurangan air relatif terhadap kebutuhan. | Direpresentasikan melalui target `SPEI_3` internal. |
| Precipitation | Curah hujan. | Komponen water deficit. |
| ET0 / FAO ET0 | Estimasi penguapan-kebutuhan air referensi. | Dikurangkan dari precipitation. |
| Water deficit | Neraca air sederhana. | `precipitation_sum - et0_fao_evapotranspiration`. |
| SPEI | Standardized Precipitation-Evapotranspiration Index. | Nama target internal proyek. |
| Pseudo-SPEI | Aproksimasi indeks, bukan implementasi standard otoritatif. | Rolling daily 90 hari + fit Fisk per bulan + z-score. |
| Node | Satu titik cuaca geografis. | 9 kandidat tetap per kota. |
| Super-node | Satu representasi kota hasil agregasi node terpilih. | Mean top-5 node kota. |
| Train-only | Keputusan preprocessing memakai data sebelum cutoff training. | Seleksi node memakai data `<= 2022-12-31`. |
| Leakage | Informasi masa depan masuk ke proses training. | Dicegah sebagian melalui cutoff selection dan chronological split. |
| Feature/covariate | Variabel input model. | Cuaca, temporal feature, dan target lagged dalam encoder. |
| Static feature | Tidak berubah sepanjang time series entity. | `super_node_id`, elevation, lat, lon. |
| Known future feature | Nilainya diasumsikan tersedia untuk decoder. | `time_idx`, `month_sin`, `month_cos`. |
| Unknown real | Tidak dianggap tersedia sebagai future external input. | Target, weather, deficit, soil moisture, dsb. |
| Encoder | History yang diberikan ke model. | 90 hari. |
| Decoder/horizon | Masa depan yang diprediksi. | 30 hari. |
| TFT | Temporal Fusion Transformer. | Library PyTorch Forecasting model. |
| Quantile | Titik distribusi; contoh P50 adalah median. | Output `[0.1,0.5,0.9]`. |
| Quantile loss | Loss untuk melatih prediksi quantile. | `QuantileLoss`. |
| Persistence/naive | Baseline: masa depan diperkirakan sama dengan masa lalu. | Untuk horizon `h`, baseline memakai `SPEI` pada `t-(h+1)`. |
| RMSE/MAE | Ukuran error point forecast. | Dihitung terhadap P50. |
| PICP | Proporsi observasi di interval P10–P90. | Metrik coverage interval. |
| Calibration | Menyesuaikan interval agar coverage validation lebih sesuai. | Scaling P10/P90 per kota dari validation. |
| Checkpoint | File parameter + normalizer model terlatih. | Lightning `.ckpt`. |
| Warmup | History tambahan sebelum periode target evaluasi. | 90 hari untuk membentuk encoder pertama. |

---

# 3. Peta kode ML

| Lokasi | Tanggung jawab |
|---|---|
| `data/config/city_centers.json` | Lima kota target dan koordinat pusat. |
| `src/data/ingest.py` | Bentuk candidate nodes, fetch data harian, raw parquet/schema. |
| `src/data/preprocess.py` | Fill missing, seleksi top-k, super-node, pseudo-SPEI, feature engineering. |
| `src/data/spei.py` | Water deficit, rolling 90 hari, Fisk CDF, z-score/extrapolation. |
| `src/models/dataset.py` | Validasi schema dan `TimeSeriesDataSet`. |
| `src/models/tft.py` | Konstruksi/loading TFT. |
| `src/training/train.py` | Split chronological, training Lightning, checkpoint. |
| `src/evaluation/calibration.py` | Kalibrasi interval P10/P90 per kota. |
| `full_evaluation.py` | Evaluasi paling lengkap. |
| `evaluate.py` | Evaluasi lama/basic; jangan samakan output dengan evaluator lengkap. |
| `src/models/mlp_baseline.py` | Tidak ada file ini pada pipeline; baseline utama adalah persistence. |
| `scripts/smoke_e2e_v2.py` | Smoke structural memakai fixture synthetic. |
| `test_pipeline.py` | Script assertion/integration, bukan suite pytest murni. |
| `results/seed44_eval/` | Evaluasi checkpoint yang dirujuk `logs/run_config.json`. |
| `results/full_eval_20260602_063310/` | Evaluasi checkpoint lain, seed 43. |
| `logs/run_config.json` | Konfigurasi run/checkpoint yang dipilih saat ini. |

---

# 4. Kontrak data geografis dan raw

## 4.1 Lima kota

Kota yang dikonfigurasi: Bojonegoro, Lamongan, Nganjuk, Ngawi, Tuban. Sumber: `data/config/city_centers.json:1-7`.

Setiap kota mempunyai sembilan kandidat titik tetap:

```text
center
north, south, east, west: offset ±0.12 derajat
four diagonal: offset ±0.08 derajat
```

Sumber: `src/data/ingest.py:14-26`.

ID node stabil:

```text
node_id = {city_id}__{node_local_id}__{sha1(lat,lon)[:8]}
```

Sumber: `src/data/ingest.py:71-95`.

Mengubah pusat kota, offset, atau formula ID mengubah kandidat seleksi serta data contract downstream. Retrain dan regenerate semua artifact diperlukan.

## 4.2 Raw weather

**Fakta kode:** ingestion memakai Open-Meteo archive daily dengan timezone `Asia/Jakarta`. Required fields:

```text
precipitation_sum
et0_fao_evapotranspiration
soil_moisture_0_to_7cm
temperature_2m_max
temperature_2m_min
relative_humidity_2m_mean
shortwave_radiation_sum
wind_speed_10m_max
```

Sumber: `src/data/ingest.py:28-41,98-106`.

Raw parquet menyimpan identity/geografi/weather serta `schema_version=2`: `src/data/ingest.py:140-171,267-296`.

Failure behavior:

- request memiliki retry;
- kegagalan satu node tidak langsung menghentikan semua fetch;
- setelah save, coverage kota dan minimum node diverifikasi;
- default mensyaratkan minimal lima node ter-fetch per kota.

Sumber: `src/data/ingest.py:98-178,201-209,298-333`.

**Batas implementasi:** respons sumber eksternal bisa berubah; source tidak menyimpan snapshot mentah respons atau versi layanan. Downstream reproducible hanya sejauh memakai raw parquet lokal yang sama.

---

# 5. Preprocessing: raw menjadi pseudo-SPEI

## 5.1 Missing value

Weather di-forward-fill per raw node maksimum tujuh row. Sisa null memberi warning; row dengan target/feature null akhirnya dibuang. Sumber: `src/data/preprocess.py:63-77,291-308`.

Forward fill memakai masa lalu, jadi tidak melakukan future-to-past interpolation.

**Batas implementasi:** gap di atas tujuh hari dapat mengurangi sample secara tidak merata. Tidak ada missingness feature atau completeness report detail per node.

## 5.2 Seleksi top-5 node per kota

Cutoff selection: `2022-12-31` (`src/data/preprocess.py:11-14,152-160,183-193`). Untuk setiap kandidat node:

1. Hitung mean Pearson correlation terhadap *leave-one-node-out city profile* pada delapan weather fields.
2. Hitung jarak terhadap rata-rata koordinat node kota.
3. Hitung:

```text
behavior_score = mean correlation
inverse_distance_score = 1 / (1 + distance_km)
hybrid_score = 0.7 * behavior_score + 0.3 * inverse_distance_score
```

4. Ambil top-5 dengan tie-break deterministik:

```text
hybrid descending
behavior descending
distance descending
raw_node_id ascending
```

Sumber: `src/data/preprocess.py:80-149`.

Artifact selection mencatat cutoff, seed, method, ordering, raw-data fingerprint, selected-node fingerprint: `src/data/preprocess.py:204-244`.

**Fakta kode:** selection memakai data training-only, tetapi bukan model learned. Ia adalah heuristik deterministic sebelum TFT.

## 5.3 Super-node

Top-5 node terpilih dirata-ratakan menjadi satu entity kota:

```text
super_node_id = SN_{city_id}
location_id = super_node_id
mean: weather, elevation, latitude, longitude
```

Sumber: `src/data/preprocess.py:246-270`.

Konsekuensi: TFT mempelajari **lima entity kota**, bukan 45 node individual dan bukan relasi edge antar-node.

## 5.4 Water deficit dan pseudo-SPEI

Water deficit:

```text
D_t = precipitation_sum_t - et0_fao_evapotranspiration_t
```

Sumber: `src/data/spei.py:8-15`.

Target bernama `SPEI_3` dibangun:

```text
W_t = sum(D_{t-89} ... D_t)              # trailing 90 hari
fit distribusi Fisk/log-logistic per bulan, memakai data <= 2022-12-31
p_t = CDF_month(W_t)
SPEI_3_t = NormalInverseCDF(p_t)
```

Sumber: `src/data/spei.py:17-111`; invocation `src/data/preprocess.py:272-281`.

Nilai ekstrem di bawah support fit memakai extrapolation linear di ruang z-score, bukan probability clipping (`src/data/spei.py:76-100`).

**Batas implementasi kritis:** source menyebut ini non-standard daily-resolution SPEI approximation (`src/data/spei.py:17-27`). Jangan menyebutnya canonical monthly SPEI-3 atau indeks yang tervalidasi otoritas tanpa validasi eksternal.

## 5.5 Engineered features

```text
SPEI_3_diff
time_idx
month
month_sin
month_cos
precipitation_log
```

Sumber: `src/data/preprocess.py:279-289`.

**Fakta mismatch:** first `SPEI_3_diff` menjadi null dan dibuang final cleaning. Tidak ada `.fillna(0)` pada source meskipun dokumen lama mengklaim demikian. Sumber: `src/data/preprocess.py:280-293`.

---

# 6. Dataset contract TFT

Kode: `src/models/dataset.py:7-145`.

## 6.1 Group, target, windows

```text
group:      super_node_id
target:     SPEI_3
encoder:    90 hari
decoder:    30 hari
sequence:   fixed exact-length
```

`create_dataset()` menolak required model columns yang tidak ada (`src/models/dataset.py:40-69`).

## 6.2 Feature roles

| Role | Kolom |
|---|---|
| Static categorical | `super_node_id` |
| Static real | `elevation`, `lat`, `lon` |
| Known future real | `time_idx`, `month_sin`, `month_cos` |
| Unknown real | `SPEI_3`, `SPEI_3_diff`, water deficit, precipitation log, ET0, soil moisture, temperature max/min, humidity, radiation, wind |

Sumber: `src/models/dataset.py:120-145`.

Continuous feature memakai `ArrayStandardScaler`; target memakai `EncoderNormalizer(transformation=None)` (`src/models/dataset.py:12-37,93-145`).

## 6.3 Apa yang model lihat

Untuk satu sample, secara konseptual:

```text
encoder: [90 hari, static + known + unknown history]
decoder: [30 hari, static + known future + dataset-required fields]
target:  [30 hari] SPEI_3
prediction: [batch, 30, 3 quantile]
```

TFT dan `TimeSeriesDataSet` mengatur encoding internal exact tensor; jangan membuat array manual lalu mengasumsikan urutannya sama tanpa memakai dataset contract.

**Batas implementasi:** tidak ada fungsi inference operational yang menyusun 90 hari history dan 30 hari known-future covariate untuk pengamatan baru. Offline evaluator memiliki processed historical rows lengkap. Input sederhana seperti kota saja tidak cukup untuk forecast TFT nyata.

---

# 7. Algoritma: Temporal Fusion Transformer

## 7.1 Intuisi

TFT adalah model sequence-to-sequence untuk time series multivariat. Ia membaca 90 hari riwayat, menggunakan static/known/unknown feature roles, lalu memprediksi 30 target future. TFT internal library mencakup variable selection, recurrent processing, gating, static context, dan temporal attention. Implementasi proyek memakai TFT dari PyTorch Forecasting; proyek ini tidak menulis ulang blok transformer manual.

## 7.2 Konfigurasi aktual

| Parameter | Nilai |
|---|---:|
| model | `TemporalFusionTransformer` |
| hidden size | 48 |
| dropout | 0.40 |
| attention heads | 1 |
| hidden continuous size | 8 |
| learning rate | `3e-4` |
| weight decay | `1e-4` |
| output size | 3 |
| quantiles | `[0.1, 0.5, 0.9]` |
| loss | `QuantileLoss` |

Sumber: `src/models/tft.py:34-85`.

Output:

```text
prediction [B, 30, 3]
index 0: P10
index 1: P50
index 2: P90
```

`full_evaluation.py` mengambil index quantile dari loss checkpoint, bukan hardcode (`full_evaluation.py:130-135`).

## 7.3 Quantile loss

Untuk target `y`, quantile prediction `q_tau`, dan level `tau`:

```text
L_tau(y,q) = max(tau * (y-q), (tau-1) * (y-q))
```

Training tiga quantile mengarahkan P10, P50, P90 menjadi estimasi bagian bawah, median, dan bagian atas distribusi. Ini tidak otomatis menjamin interval terkalibrasi; karena itu evaluator menghitung PICP dan memiliki calibration step.

## 7.4 Checkpoint security

Loader retry `torch.load(weights_only=False)` karena checkpoint Lightning menyimpan normalizer/scaler (`src/models/tft.py:7-32`).

**Batas keamanan:** deserialisasi pickle dapat mengeksekusi kode. Muat hanya checkpoint lokal/terpercaya; jangan menerima checkpoint dari pengguna tidak terpercaya.

---

# 8. Training contract

Kode: `src/training/train.py:95-281`.

## 8.1 Split chronological

```text
train:             year < 2023
validation target: year == 2023
test:              tidak dipakai train
```

Validation prepends 90 hari sebelum 2023 untuk encoder (`src/training/train.py:163-181`). Ini memastikan target awal validation tetap punya history.

## 8.2 Training behavior

```text
seed: Lightning seed_everything(seed, workers=True)
gradient clipping: 0.5
early stopping: monitor val_loss, patience 10
checkpoint: best validation loss
logging: TensorBoard
accelerator: GPU default
CPU: perlu allow_cpu=True
```

Sumber: `src/training/train.py:134-145,209-244`.

**Batas reproducibility:** strict deterministic mode optional dan current run mematikannya. Source juga memakai acceleration settings. Seed mengurangi variasi, bukan jaminan bitwise identical. Bukti: `src/training/train.py:134-145`; `logs/run_config.json:14-15`.

## 8.3 Current run config

`logs/run_config.json` merujuk checkpoint selected seed 44:

```text
encoder=90
max epochs=60
best val loss=0.1867
entities=5
train rows=32,420
validation rows=2,275
```

Sumber: `logs/run_config.json:1-22`.

Jangan menyamakan folder hasil lain dengan checkpoint ini tanpa membuka metadata/foldernya.

---

# 9. Evaluasi

## 9.1 Evaluator yang diprioritaskan

Ada dua evaluator:

```text
evaluate.py         older/basic
full_evaluation.py  comprehensive; gunakan sebagai rujukan utama
```

`full_evaluation.py` menangani warmup, pooled multihorizon metric, calibration, classification, event metric, dan importance.

## 9.2 Protocol

Schema train dibuat dari `<2023`; test rows `>=2024` (`full_evaluation.py:176-187`). Test/validation per entity diberi encoder warmup sebelumnya (`full_evaluation.py:189-210,407-430`).

Forecast windows overlap. Evaluator menyimpan prediksi pertama untuk setiap timestamp/horizon agar observasi tidak dihitung berulang (`full_evaluation.py:220-236,289-327`).

## 9.3 Metrik

| Kelompok | Metrik |
|---|---|
| P50 point prediction | RMSE, MAE, R², bias, Pearson r |
| Interval P10-P90 | PICP |
| Baseline | Persistence/naive |
| Multi-horizon | metric per horizon |
| Classification | 9-class SPEI dan 3-class broad |
| Event | threshold −1.5 dan −1.0 |
| Interpretability | TFT variable selection importance |

Sumber: `full_evaluation.py:56-68,242-327,329-374,452-476,633-667`.

## 9.4 Persistence baseline

Untuk forecast horizon `h`, naive baseline memakai observed SPEI pada:

```text
t - (h + 1)
```

Sumber: `full_evaluation.py:289-305`.

Baseline ini perlu selalu dilaporkan saat membahas TFT. Jangan membandingkan t+1 model dengan baseline horizon lain.

## 9.5 Interval calibration

Scaling P10/P90 per kota di-fit dari validation, lalu diterapkan ke test; P50 tidak berubah (`src/evaluation/calibration.py:9-48`; `full_evaluation.py:407-450`).

**Batas implementasi:** factor calibration hanya bagian evaluator/result; tidak disimpan sebagai model artifact mandiri untuk pipeline inference baru. Output forecast baru tidak dapat otomatis mereproduksi calibrated interval tanpa menyimpan/menerapkan factor yang sama.

## 9.6 Tidak ada ablation GNN/RDM

Tidak ada GNN/RDM component untuk diablate. Perubahan node selection, aggregation, target construction, feature roles, TFT config, split, atau calibration adalah experiment berbeda dan butuh retrain/evaluasi ulang.

---

# 10. Artifact, hasil, provenance

## 10.1 Multi-seed

`results/multiseed_aggregation.txt:1-59` mencatat seed 42–44:

| Metrik | Mean ± std |
|---|---:|
| t+1 RMSE | `0.3049 ± 0.0250` |
| t+1 MAE | `0.1920 ± 0.0237` |
| t+1 PICP | `0.7888 ± 0.0113` |
| all-horizon RMSE | `0.4118 ± 0.0342` |
| all-horizon skill | `+24.8% ± 6.2%` |
| t+1 skill | `-76.2% ± 14.4%` |

**Fakta artifact:** seluruh seed tercatat kalah dari persistence di first step; aggregate mengklaim unggul 28/30 horizon. **Tidak dibuktikan:** source/test tidak otomatis memverifikasi teks agregasi atau proses run historis.

## 10.2 Seed 44 current config

Link checkpoint: `logs/run_config.json:21`. Link evaluasi: `results/seed44_eval/metrics_summary.json:1-38`.

| Metrik | Nilai |
|---|---:|
| t+1 RMSE | `0.27685` |
| t+1 MAE | `0.16338` |
| t+1 R² | `0.92021` |
| t+1 PICP | `0.79858` |
| t+1 skill vs naive | `-59.97%` |
| all-30-horizon RMSE | `0.36350` |
| all-30-horizon skill | `+33.61%` |

## 10.3 Stale/parallel artifact

`results/full_eval_20260602_063310` mengevaluasi checkpoint seed 43, val loss `0.2405`, bukan seed 44 current run. Sumber: `results/full_eval_20260602_063310/metrics_summary.json:1-25`.

Jangan memilih result “terbaru” berdasarkan nama folder. Resolve checkpoint dari run config, lalu jalankan evaluator secara eksplisit.

**Batas provenance:** artifact tidak mengikat semua checkpoint hash, checksum data, git commit, package version, device, dan waktu run secara lengkap. Jangan overclaim reproducibility atau finality.

---

# 11. Test coverage

## 11.1 Script integration

`test_pipeline.py` memeriksa:

- imports;
- label threshold SPEI;
- parquet integrity;
- dataset/dataloader shape;
- checkpoint load dan prediction tensor shape;
- eksekusi `evaluate.py`.

Sumber: `test_pipeline.py:94-114,145-257,277-415,419-461`.

**Batas:** script mengasumsikan data/checkpoint ada, menulis `results/` melalui evaluator, tidak rebuild raw data/retrain, tidak menguji `full_evaluation.py`, dan tidak memakai fixture expected forecast.

## 11.3 Synthetic smoke

`scripts/smoke_e2e_v2.py:1-142` membuat fixture synthetic delapan kota/enam node dan dapat train satu epoch. Ini berguna untuk kontrak structural; bukan bukti akurasi data nyata.

## 11.4 Test yang belum ada

- ingestion/schema fixture;
- top-k selection dan tie-break;
- cutoff train-only/no leakage;
- pseudo-SPEI fit/extrapolation;
- missing-value behavior;
- quantile calibration;
- expected forecast fixture;
- rolling-origin / spatial holdout;
- artifact/checkpoint provenance.

---

# 12. Dokumentasi lama yang tidak boleh dijadikan sumber utama

Source/artifact yang diinspeksi menang atas dokumen lama.

| Mismatch | Dokumen lama | Source/artifact aktual |
|---|---|---|
| TFT hyperparameter | hidden 64, dropout .20, heads 2, continuous 10, patience 30 | hidden 48, dropout .40, heads 1, continuous 8, patience 10 |
| Feature inventory | lima weather variable/fewer unknowns | delapan source weather + 11 unknown reals |
| Processed rows/date | 37,460; start 2005-06-29 | inspected 37,905; start 2005-04-01 |
| Split totals | 31,975/1,975/3,365 | current run 32,420/2,275 train/val |
| Result model | stale enc30/7 quantiles | current enc90/3 quantiles |

Evidence old docs: `docs/DOKUMENTASI_METODE_PENELITIAN_TFT_SPEI_SUPERNODE.md:51-55,88-90,132,144-148,187-202,276-301`. Actual source: `src/models/tft.py:34-85`, `src/training/train.py:209-218`, `src/models/dataset.py:127-139`, `logs/run_config.json:17-20`.

---

# 13. Roadmap belajar

## Tahap 1 — problem, target, contract

Baca bagian 1-5 dan `src/data/spei.py`.

Lulus bila dapat menjawab:

```text
Apa target sebenarnya?
Pseudo-SPEI harian 90 hari internal, bukan canonical monthly SPEI-3.

Apa entity model?
Lima super-node kota, bukan raw node dan bukan graph.
```

## Tahap 2 — data dan selection

Baca `src/data/ingest.py`, `src/data/preprocess.py`.

Lulus bila dapat menulis flow:

```text
9 candidate node/kota -> behavior+distance -> top 5 -> mean kota
```

Jelaskan cutoff 2022-12-31 serta alasannya.

## Tahap 3 — pseudo-SPEI

Baca `src/data/spei.py`.

Lulus bila dapat membedakan:

```text
water deficit harian
rolling accumulation 90 hari
fit distribution per bulan
normal z-score
```

Sebut limitation non-standard daily approximation.

## Tahap 4 — TimeSeriesDataSet

Baca `src/models/dataset.py`.

Lulus bila mampu membedakan static, known future, unknown real; serta menjelaskan encoder 90 vs decoder 30.

## Tahap 5 — TFT quantile forecast

Baca `src/models/tft.py`.

Lulus bila dapat menjelaskan P10/P50/P90 dan mengapa P50 saja tidak cukup untuk uncertainty.

## Tahap 6 — training

Baca `src/training/train.py`.

Lulus bila dapat menunjukkan temporal split, validation warmup, early stop, checkpoint, GPU/seed limitation.

## Tahap 7 — evaluation

Baca `full_evaluation.py`, `src/evaluation/calibration.py`.

Lulus bila dapat menjelaskan baseline `t-(h+1)`, dedup overlap, PICP, calibration, dan fakta t+1 model kalah persistence.

## Latihan aman

```bash
cd /media/DiskE/SKRIPSI/Skripsi_Nopal
python scripts/smoke_e2e_v2.py --help
python full_evaluation.py --help
```

Jangan menjalankan training atau `test_pipeline.py` hanya untuk belajar: keduanya dapat memakai banyak resource atau menulis artifact.

---

# 14. Handover AI ML-only

## 14.1 Prompt bootstrap

```text
Kerjakan hanya ML pipeline repository Skripsi_Nopal. Baca ROADMAP_TFT_SPEI_HANDOVER.md sebelum mengubah kode.

Fakta utama: model adalah PyTorch Forecasting TFT quantile forecast untuk pseudo-SPEI harian. Pipeline spatial bukan GNN: 9 candidate weather node per kota dipilih top-5 memakai train-only hybrid score, lalu dirata-rata menjadi 5 city super-node. Target SPEI_3 adalah rolling daily 90-day water-deficit index dengan monthly Fisk fit; bukan canonical monthly SPEI-3.

Kontrak utama: cutoff node selection, stable IDs, super_node_id, feature roles, encoder 90, horizon 30, quantile order P10/P50/P90, chronological split, checkpoint normalizers, and evaluator baseline. Sebelum edit, sebutkan data/target contract, leakage risk, retrain impact, evaluation impact, and test coverage. Bedakan fakta source, limitation, and experiment evidence.

Jangan mengklaim model unggul pada t+1: artifact current menunjukkan kalah persistence. Jangan menyebut GNN/RDM/STGNN. Cantumkan path:line untuk klaim teknis.
```

## 14.2 File wajib dibaca

```text
ROADMAP_TFT_SPEI_HANDOVER.md
data/config/city_centers.json
src/data/ingest.py
src/data/preprocess.py
src/data/spei.py
src/models/dataset.py
src/models/tft.py
src/training/train.py
src/evaluation/calibration.py
full_evaluation.py
logs/run_config.json
results/seed44_eval/metrics_summary.json
```

## 14.3 Checklist sebelum perubahan

- [ ] Candidate-node geography/ID stability dipertahankan atau artifact diregenerate.
- [ ] Node selection hanya memakai cutoff training.
- [ ] Top-k ordering/tie-break deterministic tetap eksplisit.
- [ ] Aggregation super-node dan feature units terverifikasi.
- [ ] Pseudo-SPEI formula/fitting scope tidak berubah diam-diam.
- [ ] Required dataset columns dan feature roles sinkron.
- [ ] Encoder/horizon/quantile order/checkpoint normalizer kompatibel.
- [ ] Validation/test temporal dan warmup tetap benar.
- [ ] Baseline memakai horizon yang sama.
- [ ] Calibration artifact disimpan bila dipakai output inference.
- [ ] Contract/model/data berubah: retrain, full evaluation, provenance baru.
- [ ] Result menyebut checkpoint, raw/processed data, seed, split, config, evaluator.

---

# 15. Command dan constraint

Jalankan dari root proyek:

```bash
cd /media/DiskE/SKRIPSI/Skripsi_Nopal
python main.py --skip-train
python -c "from src.training.train import train_pipeline; train_pipeline(max_epochs=60, seed=44)"
python full_evaluation.py --checkpoint logs/checkpoints/enc90-run20260602_030625-epoch=20-val_loss=0.1867.ckpt
python test_pipeline.py
```

Kontrak command: `main.py:78-174`, `src/training/train.py:95-281`, `full_evaluation.py:812-839`, `test_pipeline.py:1-4`.

**Constraint:** training default membutuhkan GPU; gunakan parameter CPU explicit bila benar-benar diperlukan. Ingestion tergantung source eksternal. `test_pipeline.py` dapat mengubah results. Checkpoint hanya dari sumber tepercaya.

---

# 16. Referensi cepat

| Area | Path:line |
|---|---|
| Candidate node/raw ingestion | `src/data/ingest.py:14-333` |
| Missing, selection, aggregate, features | `src/data/preprocess.py:11-308` |
| Pseudo-SPEI | `src/data/spei.py:8-111` |
| Dataset contract | `src/models/dataset.py:7-145` |
| TFT config/load | `src/models/tft.py:7-85` |
| Training | `src/training/train.py:95-281` |
| Calibration | `src/evaluation/calibration.py:9-48` |
| Full evaluation | `full_evaluation.py:56-839` |
| Current run provenance | `logs/run_config.json:1-22` |
| Current checkpoint result | `results/seed44_eval/metrics_summary.json:1-66` |

---

# 17. Definisi “paham”

Manusia atau AI dianggap memahami bila dapat:

1. Menggambar full data flow tanpa menyebut graph model yang tidak ada.
2. Menjelaskan mengapa target adalah pseudo-SPEI, bukan canonical SPEI.
3. Menjelaskan top-5 train-only selection dan super-node mean aggregation.
4. Menyebut encoder 90, horizon 30, feature roles, quantile order.
5. Menjelaskan checkpoint normalizer, warmup validation/test, dan persistence baseline.
6. Menyatakan t+1 deficit terhadap naive dari artifact current.
7. Menentukan perubahan mana yang memaksa artifact regenerate/retrain/full evaluation.
8. Menolak klaim performa/provenance yang tidak memiliki bukti lokal memadai.

Tidak ada dokumen dapat membuktikan behavior yang belum diuji atau provenance yang tidak disimpan. Dokumen ini menyebut batas tersebut agar handover tidak berubah menjadi asumsi.