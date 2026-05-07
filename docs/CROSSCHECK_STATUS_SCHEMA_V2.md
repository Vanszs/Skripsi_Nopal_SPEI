# Crosscheck Status Schema v2 (Verified)

Tanggal verifikasi: `2026-04-21`

Scope: audit migrasi arsitektur multi-node dengan entitas model `super_node_id` (schema_version = 2).

## 1) Ringkasan Status

- Total item audit: **16**
- Sudah di-crosscheck (Closed): **16**
- Belum di-crosscheck (Open/Unchecked): **0**

## 2) Checked Items (Closed)

### A. Identitas Data Mentah & Integritas Kunci

- [x] (1) Every raw node has a unique `node_id`  
  Status: Closed  
  Risiko: High  
  Bukti singkat: `node_id` dibentuk deterministik (`city_id + node_local_id + coord_hash`), validasi uniqueness `(node_id,time)` dan `(raw_node_id,time)` lulus.

- [x] (6) Tests validate uniqueness of group keys  
  Status: Closed  
  Risiko: High  
  Bukti singkat: test menegaskan duplicate `(super_node_id,time_idx)` dan `(city_id,time_idx)` = 0.

### B. Leakage Safety & Determinisme

- [x] (2) No data leakage between nodes within the same city  
  Status: Closed  
  Risiko: High  
  Bukti singkat: seleksi node memakai train window saja (`time <= selection_end_date`), operasi node-level tetap per `raw_node_id`.

- [x] (3) Selection logic is deterministic/reproducible  
  Status: Closed  
  Risiko: High  
  Bukti singkat: sorting stabil dengan tie-breaker eksplisit + fingerprint SHA256 node selection konsisten antar rerun.

- [x] (16) System is reproducible  
  Status: Closed  
  Risiko: High  
  Bukti singkat: seed dikontrol, metadata run/selection dipersist, hash seleksi identik pada input+seed yang sama.

### C. Cardinality Dinamis & Anti Hardcode

- [x] (4) No hardcoded “5 locations” logic exists  
  Status: Closed (execution path)  
  Risiko: Medium  
  Bukti singkat: evaluasi/plotting memakai loop, grid, dan palette dinamis.

- [x] (5) Tests do not assume exactly 5 cities  
  Status: Closed  
  Risiko: Medium  
  Bukti singkat: assertion cardinality sudah dinamis (`>=1`, konsistensi entity-city).

- [x] (14) Cardinality > 5 works correctly  
  Status: Closed  
  Risiko: High  
  Bukti singkat: smoke 8 entitas lulus (`all_pass: true`).

- [x] (15) No hardcoded city list anywhere  
  Status: Closed (active code + required docs/instructions)  
  Risiko: Medium  
  Bukti singkat: registry city berbasis config, asumsi daftar kota statis dibersihkan dari scope aktif.

### D. E2E Runtime, Test, dan Evaluasi

- [x] (7) End-to-end pipeline runs with new schema  
  Status: Closed  
  Risiko: High  
  Bukti singkat: smoke schema-v2 dan jalur train/eval berhasil.

- [x] (11) Evaluation methodology is updated  
  Status: Closed  
  Risiko: Medium  
  Bukti singkat: eval kompatibel schema-v2, output tetap per-entity dan per-city.

- [x] (13) End-to-end pipeline runs without crash  
  Status: Closed  
  Risiko: High  
  Bukti singkat: test pipeline dan smoke run selesai tanpa runtime exception di jalur kritikal.

### E. Dokumentasi & Instruksi Internal

- [x] (8) Documentation reflects multi-node architecture  
  Status: Closed  
  Risiko: Medium  
  Bukti singkat: dokumen utama memuat override/addendum schema-v2.

- [x] (9) Super-node concept is clearly documented  
  Status: Closed  
  Risiko: Medium  
  Bukti singkat: `super_node_id` dijelaskan sebagai group key training sequence.

- [x] (10) Schema changes are documented  
  Status: Closed  
  Risiko: Medium  
  Bukti singkat: kontrak schema-v2 + guard `schema_version == 2` terdokumentasi.

- [x] (12) Internal agent/dev instructions updated  
  Status: Closed  
  Risiko: Medium  
  Bukti singkat: aturan internal diselaraskan ke config-driven schema-v2.

## 3) Unchecked Items (Belum Selesai)

- [ ] Tidak ada item open pada scope audit 16 poin ini.

Catatan: meskipun seluruh item berstatus Closed, beberapa area tetap berisiko regresi tinggi jika refactor mengubah urutan preprocess, tie-breaker, atau schema guard.

## 4) Watchlist Re-check (Bukan Open Item, tapi wajib dipantau)

- [x] Re-check leakage guard setelah perubahan besar di `preprocess.py`.
- [x] Re-check deterministic hash setelah perubahan scoring/sorting pemilihan node.
- [x] Re-check dynamic cardinality jika ada perubahan plotting/layout evaluasi.
- [x] Re-check schema guard (`schema_version == 2`) jika ada perubahan format data processed.

## 5) Closure Checklist (Final)

- [x] `node_id` unik, deterministik, dan tervalidasi.
- [x] Tidak ada mixing node-level sebelum agregasi super-node.
- [x] Seleksi top-k deterministik pada input+seed yang sama.
- [x] Tidak ada fixed logic 5 lokasi pada execution path aktif.
- [x] Test bersifat cardinality-agnostic.
- [x] Uniqueness key teruji di raw dan processed path.
- [x] Jalur E2E schema-v2 berjalan (ingest/preprocess/dataset/train/evaluate).
- [x] Dokumentasi & instruksi internal konsisten dengan arsitektur multi-node.
- [x] Reproducibility dibuktikan via artifact metadata dan fingerprint stabil.
