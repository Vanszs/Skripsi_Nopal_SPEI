# Migration Checklist Status (Verified)

Verification date: `2026-04-22`

Scope: migration `5 nodes per city -> 1 super-node` pada jalur aktif pipeline (`ingest -> preprocess -> dataset -> train -> evaluate -> visualize`) dan artefak dataset default.

## A. IDENTITY & SCHEMA

- [x] every raw node has unique `node_id` (`src/data/ingest.py`, uniqueness checks)
- [x] `city_id` explicitly defined and consistent (`src/data/ingest.py`, `src/data/preprocess.py`)
- [x] `super_node_id` exists and used as model entity (`src/data/preprocess.py`, `src/models/dataset.py`)
- [x] model grouping key unique per training entity (`src/models/dataset.py`)
- [x] no multi-node sharing same training `group_id` before aggregation (`src/data/preprocess.py`)
- [x] schema includes `lat/lon` per node (`src/data/ingest.py`)
- [x] schema version guard exists (`main.py`, `src/data/preprocess.py`, `src/training/train.py`)

## B. LEAKAGE & DATA SAFETY

- [x] no node-level leakage in selection flow (train-only cutoff enforced)
- [x] sequence boundaries isolated by entity (`super_node_id`)
- [x] similarity computed from training window only (`<= selection_end_date`)
- [x] no val/test data used in node selection
- [x] aggregation performed after node selection
- [x] train/val/test split preserved after transformation

## C. INGEST PIPELINE

- [x] ingestion runs at node level (not single city point)
- [x] city registry supports multi-node per city (config-driven)
- [x] metadata persisted: `city_id`, `node_local_id`, `node_id`, `raw_node_id`, `lat`, `lon`
- [x] uniqueness assertions active for `(node_id,time)` and `(raw_node_id,time)`
- [x] ingest fail-fast completeness guard exists for all config cities + minimum nodes/city (`main.py` + `src/data/ingest.py`)

## D. PREPROCESS PIPELINE

- [x] preprocessing runs per node before aggregation
- [x] top-5 node selection per city implemented
- [x] selection deterministic (stable tie-break sort + seed)
- [x] exactly 1 super-node per `(city_id,time)` produced
- [x] feature engineering applied after aggregation
- [x] reproducibility metadata persisted (`selected_nodes_fingerprint_sha256`)
- [x] default raw dataset currently satisfies top-5 requirement for every configured city

## E. DATASET & TRAINING

- [x] leakage-safe `group_ids` used (`super_node_id`)
- [x] static categoricals updated (`super_node_id`, `city_id`)
- [x] static reals updated (`elevation`, `lat`, `lon`)
- [x] sequence construction respects entity boundaries
- [x] training logs include entity cardinality
- [x] sanity checks for group size/duplicate keys exist
- [x] run config artifact persisted (`logs/run_config.json`)

## F. EVALUATION & VISUALIZATION

- [x] evaluation uses schema-v2-consistent merge keys
- [x] per-entity evaluation supported
- [x] per-city aggregated evaluation supported
- [x] no fixed-cardinality plotting assumption in active eval scripts
- [x] subplot layout dynamic (no fixed `2x3`)
- [x] color palette dynamic
- [x] visualization script data-driven (`scripts/visualize_predictions.py`, dynamic csv resolution)

## G. TESTING

- [x] tests do not require exactly 5 cities (cardinality agnostic)
- [x] tests validate schema-v2 integrity
- [x] tests validate uniqueness of group keys (`(super_node_id,time_idx)`, `(city_id,time_idx)`)
- [x] tests validate selected node count per city (`selected_node_count == 5`)
- [x] tests fail/warn on stale schema and use schema-v2 fallback path

## H. PIPELINE ROBUSTNESS

- [x] stale dataset schema detection active
- [x] version guard prevents silent reuse (schema version/columns)
- [x] version guard validates completeness coverage (all config cities and node minimum)
- [x] end-to-end smoke with new schema passes (`scripts/smoke_e2e_v2.py --seed 42`)
- [x] cardinality `> 5` validated (8 entities scenario)
- [x] reproducibility verified by identical selection hash across reruns
- [x] full default E2E is complete for all configured cities in current primary artifacts

## I. DOCUMENTATION & INSTRUCTIONS

- [x] required docs updated with schema-v2 multi-node architecture
- [x] super-node concept explicitly documented
- [x] schema/versioning behavior documented
- [x] evaluation granularity/method updated (per-entity + per-city)
- [x] internal instructions updated (`.github/copilot-instructions.md`, `.agent/rules/agent.md`)

## J. Current Open Items

- [x] none

## Runtime Evidence Artifacts

- Config city registry: `data/config/city_centers.json` (5 cities)
- Raw primary dataset: `data/raw/weather_history_east_java.parquet` (node/city saat cek: `Bojonegoro=9, Lamongan=9, Nganjuk=9, Ngawi=9, Tuban=9`)
- Processed primary dataset: `data/processed/spei_dataset.parquet` (schema v2, cities saat cek: `Bojonegoro, Lamongan, Nganjuk, Ngawi, Tuban`)
- Smoke report: `results/smoke_v2/smoke_report.json`
- Selection artifact: `results/smoke_v2/node_selection_v2.parquet`
- Selection metadata: `results/smoke_v2/node_selection_v2.meta.json`
- Evaluation outputs: `results/predictions_eval.csv`, `results/evaluation_metrics_detailed.json`
- Latest TFT-30 checkpoint: `logs/checkpoints/enc30-run20260422_034701-epoch=5-val_loss=0.1845.ckpt`
- Latest TFT-30 train monitor logs: `logs/train_gpu_monitor_20260422_034701.log`, `logs/train_gpu_monitor_20260422_034701.err.log`

## K. TFT 30-Hari (Latest Run)

- [x] encoder tetap `30 hari` (`logs/run_config.json` -> `max_encoder_length=30`)
- [x] training berjalan di GPU tanpa crash (`accelerator=gpu`, log monitor per-menit stabil)
- [x] tidak ada anomali numerik (`nan/runtime_error`) pada run terbaru
- [x] checkpoint terbaik tersimpan (`enc30-run20260422_034701-epoch=5-val_loss=0.1845.ckpt`)
- [x] evaluasi terbaru memakai checkpoint TFT-30 yang sama
- [x] metrik terbaru menunjukkan model mengungguli naive di step-1 dan 30/30 horizon (`results/evaluation_metrics_detailed.json`)
