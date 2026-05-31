# Copilot Instructions - SPEI TFT (Schema v2)

## Scope
This repository implements SPEI forecasting with Temporal Fusion Transformer using schema v2:
- raw ingestion: multi-node per city
- preprocessing: train-only top-k node selection per city
- modeling entity: `super_node_id` (1 super-node per city per timestamp)
- evaluation: both per-entity and per-city outputs

## Non-negotiable rules
1. Do not hardcode city names or city count in pipeline logic, tests, evaluation, or plotting.
2. Treat city/node coverage as config-driven (`data/config/city_centers.json`).
3. Enforce schema version guard (`schema_version == 2`) and reject stale schema.
4. Keep leakage safety:
   - node-level ops grouped by `raw_node_id`/`node_id`
   - node similarity computed on training window only
   - aggregation only after node selection
5. Grouping key for training and sequence boundaries must be `super_node_id`.
6. Keep deterministic behavior:
   - fixed seed
   - deterministic sorting with explicit tie-breakers
   - persist node-selection artifact and metadata fingerprint

## Pipeline contract
1. `src/data/ingest.py`
   - emits raw schema v2 with unique `node_id`, `raw_node_id`, `city_id`, `node_local_id`, `lat`, `lon`.
2. `src/data/preprocess.py`
   - selects top-k nodes per city (train-only), aggregates to super-node, computes final features.
3. `src/models/dataset.py`
   - builds dataset with `group_ids=[super_node_id]`.
4. `src/training/train.py`
   - validates schema/cardinality and logs run config.
5. `evaluate.py` and `full_evaluation.py`
   - use schema-v2 merge keys consistently and support dynamic cardinality.

## Testing expectations
- tests must be cardinality-agnostic for city count.
- uniqueness checks required for:
  - raw `(node_id, time)` and `(raw_node_id, time)`
  - processed `(super_node_id, time_idx)` and `(city_id, time_idx)`
- plotting/layout must scale for cardinality > 5.

## Reproducibility
- persist:
  - `data/processed/node_selection_v2.parquet`
  - `data/processed/node_selection_v2.meta.json`
  - `logs/run_config.json`
- rerun with same input+seed must produce same selected node set.
