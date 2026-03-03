# Copilot Instructions — Skripsi SPEI Forecasting (TFT)

## Project Overview
Thesis project forecasting SPEI (Standardized Precipitation-Evapotranspiration Index) for 5 rice-producing regencies in East Java (Bojonegoro, Lamongan, Nganjuk, Ngawi, Tuban) using Temporal Fusion Transformer (TFT). Daily data, 30-day forecast horizon, SPEI-3 and SPEI-6 indices as targets/features.

---

## Error Debugging Priority
**Always search errors with Perplexity + Context7 before guessing a fix:**
1. Use `mcp_perplexity_perplexity_ask` for quick error lookups (pytorch-forecasting, lightning, scipy errors)
2. Use `mcp_upstash_conte_query-docs` (Context7) for library API details — resolve library ID first with `mcp_upstash_conte_resolve-library-id`  
3. Only fall back to codebase grep after exhausting web sources

```
# Quick error search pattern:
perplexity_ask: "pytorch-forecasting TimeSeriesDataSet <error message>"
context7 resolve: "pytorch-forecasting" → then query-docs with specific question
```

---

## Architecture & Data Flow

```
Open-Meteo API → data/raw/weather_history_east_java.parquet
                → data/processed/weather_processed.parquet  (with SPEI indices)
                → TFT model (logs/checkpoints/*.ckpt)
                → results/ (predictions, metrics, plots)
```

**5-stage pipeline:**
1. `src/data/ingest.py` — Open-Meteo Archive API, 5 locations, 2005–2026-01-01
2. `src/data/preprocess.py` → `src/data/spei.py` — compute water deficit, SPEI-3/6 via Log-Logistic (fisk)
3. `src/models/dataset.py` — build `TimeSeriesDataSet`
4. `src/training/train.py` — PyTorch Lightning + TFT
5. `evaluate.py` — load checkpoint, run inference, save to `results/`

Entrypoint for full pipeline: `main.py`. Evaluation only: `evaluate.py`.

---

## Critical Patterns

### Checkpoint ↔ Dataset Compatibility
Checkpoints store `max_encoder_length` in hparams. **Always read from checkpoint before creating datasets:**
```python
model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
ckpt_encoder_len = model.hparams.max_encoder_length  # e.g., 30 for existing ckpt
train_ds = create_dataset(train_data, max_encoder_length=ckpt_encoder_len)
```
Current checkpoint `logs/checkpoints/epoch=8-val_loss=0.37.ckpt` was trained with `max_encoder_length=30`. The default in `dataset.py` is `90` (for new training runs).

### SPEI Computation
- Distribution: **Log-Logistic (`scipy.stats.fisk`)** per Vicente-Serrano et al. (2010) — NOT `pearson3`
- Calibration: per-calendar-month fitting on rolling daily sums (scale × 30 days)
- Data must be shifted to positive domain: `shift = abs(min) + 1.0` before `fisk.fit(..., floc=0)`
- 9 WMO classes in `classify_spei()`: Kekeringan Ekstrem/Parah/Sedang/Ringan, Normal, Basah Ringan/Sedang/Parah/Ekstrem

### Feature Layout in TimeSeriesDataSet
| Category | Features |
|---|---|
| `static_categoricals` | `location_id` |
| `static_reals` | `elevation` ← must NOT be in `time_varying_known_reals` |
| `time_varying_known_reals` | `time_idx`, `month_sin`, `month_cos` |
| `time_varying_unknown_reals` | `SPEI_3`, `SPEI_6`, `water_deficit`, `precipitation_log`, `et0_fao_evapotranspiration`, `soil_moisture`, `temperature_2m_max`, `temperature_2m_min` |

### Train/Val/Test Split
- Train: `< 2023`
- Val: `2023`
- Test: `>= 2024`
- Data verified: 35,635 rows, 0 NaN, 5 locations

---

## Developer Workflows

### Python Environment (Windows)
```powershell
# Activate venv
& d:\SKRIPSI\Skripsi_Nopal\venv\Scripts\Activate.ps1
# Run any script
& "d:\SKRIPSI\Skripsi_Nopal\venv\Scripts\python.exe" <script.py>
```

### Run Full Pipeline Test
```powershell
python test_pipeline.py
# Tests: imports, classify_spei (9 classes), data integrity, dataset creation,
#        model inference + metrics, evaluate.py end-to-end
# All 6 tests should PASS; [WARN] on Bias is expected (existing ckpt trained with old features)
```

### Run Evaluation Only
```powershell
python evaluate.py --checkpoint logs/checkpoints/epoch=8-val_loss=0.37.ckpt
```

### Git Push
```powershell
git add -A; git commit -m "message"; git push origin main
# Note: /data/ is excluded by .gitignore (anchored), but src/data/ is tracked
```

---

## Framework Notes
- **Use `lightning` (not `pytorch_lightning`)** — import as `import lightning as L`
- Variable importance from `model.interpret_output(raw_preds, reduction="sum")` returns a dict with keys `encoder_variables`, `decoder_variables`
- SPEI is already Z-score normalized → use `GroupNormalizer(transformation=None)` in dataset
- GPU: NVIDIA RTX 3050 (CUDA) — tensor `.numpy()` requires `.cpu()` first
- Add `torch.set_float32_matmul_precision('medium')` to suppress Tensor Core warnings

---

## Key Files
| File | Purpose |
|---|---|
| [src/data/spei.py](../src/data/spei.py) | SPEI computation (fisk distribution, 9-class classifier) |
| [src/models/dataset.py](../src/models/dataset.py) | `create_dataset()` with overridable encoder/pred length |
| [evaluate.py](../evaluate.py) | Full evaluation pipeline, reads ckpt hparams first |
| [test_pipeline.py](../test_pipeline.py) | 6-stage CLI test suite |
| [logs/checkpoints/epoch=8-val_loss=0.37.ckpt](../logs/checkpoints/epoch=8-val_loss=0.37.ckpt) | Best checkpoint (encoder=30, pred=30) |
