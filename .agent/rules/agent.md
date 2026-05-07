---
trigger: always_on
---

ANDA ADALAH AI ENGINEER UNTUK PIPELINE SPEI-TFT SCHEMA V2.
Fokus: implementasi teknis yang robust, leakage-safe, reproducible.

=====================================================================
ATURAN UTAMA (WAJIB)
=====================================================================
1. Jangan hardcode nama kota atau jumlah kota di logic pipeline.
2. Sumber daftar kota/node harus config-driven (`data/config/city_centers.json`).
3. Raw schema wajib v2 dan harus memuat: `schema_version`, `city_id`, `node_local_id`, `node_id`, `raw_node_id`, `lat`, `lon`, `time`.
4. Preprocessing wajib urut:
   - cleaning/interpolation per node
   - node selection train-only (<= batas training)
   - top_k per city deterministik
   - agregasi menjadi super-node
   - feature engineering final
5. Group key model wajib `super_node_id`.
6. Evaluasi wajib mendukung:
   - per-entity (`super_node_id`)
   - per-city (`city_id`)
7. Semua file schema lama harus ditolak lewat version guard (tidak boleh fallback implisit).

=====================================================================
KONTRAK ANTI-LEAKAGE
=====================================================================
- Tidak boleh mencampur urutan sequence beberapa raw node di group id yang sama sebelum agregasi.
- Similarity/select node tidak boleh pakai val/test period.
- Merge key evaluasi harus konsisten dengan schema v2.

=====================================================================
KONTRAK REPRODUCIBILITY
=====================================================================
- Seed eksplisit untuk preprocessing dan training.
- Sort tie-breaker eksplisit saat top-k selection.
- Persist artifact:
  - `data/processed/node_selection_v2.parquet`
  - `data/processed/node_selection_v2.meta.json`
  - `logs/run_config.json`

=====================================================================
CHECKLIST KUALITAS EKSEKUSI
=====================================================================
- Unik raw: `(node_id, time)` dan `(raw_node_id, time)`.
- Unik processed: `(super_node_id, time)` dan `(city_id, time)`.
- Tidak ada asumsi fixed layout jumlah lokasi pada visualisasi.
- Pipeline jalan end-to-end tanpa crash pada cardinality > 5.
