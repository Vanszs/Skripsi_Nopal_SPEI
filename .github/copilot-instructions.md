# Copilot Instructions — Skripsi SPEI Forecasting (TFT)

## Project Overview
Thesis project forecasting SPEI (Standardized Precipitation-Evapotranspiration Index) for 5 rice-producing regencies in East Java (Bojonegoro, Lamongan, Nganjuk, Ngawi, Tuban) using Temporal Fusion Transformer (TFT).

- Daily data
- 30-day forecast horizon
- SPEI-3 and SPEI-6 as targets/features
- Step-0-only decoding for fair temporal evaluation
- Naive persistence baseline + PICP included in evaluation

---

## Error Debugging Priority
Always search errors with Perplexity + Context7 before guessing a fix:

1. mcp_perplexity_perplexity_ask
2. mcp_upstash_conte_resolve-library-id
3. mcp_upstash_conte_query-docs
4. Only then grep local codebase

Example:
perplexity_ask: "pytorch-forecasting TimeSeriesDataSet <error>"
context7 resolve: "pytorch-forecasting"

---

## Architecture & Data Flow

Open-Meteo API  
→ data/raw/weather_history_east_java.parquet  
→ data/processed/weather_processed.parquet (with SPEI indices)  
→ TFT model (logs/checkpoints/*.ckpt)  
→ results/ (predictions, metrics, plots)

5-stage pipeline:
1. src/data/ingest.py — Open-Meteo Archive API (2005–2026-01-01)
2. src/data/preprocess.py → src/data/spei.py — water deficit + SPEI (Log-Logistic / fisk)
3. src/models/dataset.py — TimeSeriesDataSet builder
4. src/training/train.py — Lightning + TFT
5. evaluate.py — checkpoint inference + full metrics

Entrypoint full pipeline: main.py  
Evaluation only: evaluate.py

---

## Critical Patterns

### 1️⃣ Checkpoint ↔ Dataset Compatibility

Checkpoint stores:
- max_encoder_length
- max_prediction_length

Always load checkpoint FIRST:

model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
enc_len = model.hparams.max_encoder_length
pred_len = model.hparams.max_prediction_length

train_ds = create_dataset(train_data,
                          max_encoder_length=enc_len,
                          max_prediction_length=pred_len)

Current checkpoint:
logs/checkpoints/epoch=8-val_loss=0.37.ckpt  
→ trained with encoder=30, prediction=30

dataset.py default = 90 (for new training runs)

---

### 2️⃣ No Partial Windows (CRITICAL)

To avoid window ambiguity:

- min_encoder_length = max_encoder_length
- min_prediction_length = max_prediction_length

This ensures:
- No partial context
- No partial 1–29 day targets
- Full 30-day horizon consistency

---

### 3️⃣ SPEI Computation

- Distribution: Log-Logistic (scipy.stats.fisk)
- NOT pearson3
- Calibration: per-calendar-month
- Rolling window = scale × 30 days
- Shift to positive domain before fit:
  shift = abs(min_value) + 1.0
  fisk.fit(data_shifted, floc=0)

SPEI already Z-score normalized.

9 WMO classes:
Ekstrem/Parah/Sedang/Ringan Kering,
Normal,
Ringan/Sedang/Parah/Ekstrem Basah

---

### 4️⃣ Feature Layout (TimeSeriesDataSet)

static_categoricals:
- location_id

static_reals:
- elevation

time_varying_known_reals:
- time_idx
- month_sin
- month_cos

time_varying_unknown_reals:
- SPEI_3
- SPEI_6
- water_deficit
- precipitation_log
- et0_fao_evapotranspiration
- soil_moisture
- temperature_2m_max
- temperature_2m_min

SPEI already standardized → use:
GroupNormalizer(transformation=None)

---

### 5️⃣ Train / Val / Test Split (STRICT)

Train: year < 2023  
Val: year == 2023  
Test: year >= 2024  

Scaler MUST be fit on train only.

No leakage from 2023/2024 into scaler.

---

## Evaluation Standards

### Step-0-Only Decoding (IMPORTANT)

For each time_idx:
- Use only prediction at step=0
- DO NOT average step 0–29

Reason:
Avoid temporal smoothing bias and stale-context averaging.

---

### Metrics

Primary:
- RMSE
- MAE
- R²
- Pearson r
- Bias

Probabilistic:
- PICP (Prediction Interval Coverage Probability)
  For P10–P90 → nominal coverage ≈ 80%

Baseline:
- Naive persistence:
  SPEI(t) ≈ SPEI(t-1)

Model must outperform naive RMSE.

---

## Developer Workflows

Activate venv:
& d:\SKRIPSI\Skripsi_Nopal\venv\Scripts\Activate.ps1

Run evaluation:
python evaluate.py --checkpoint logs/checkpoints/epoch=8-val_loss=0.37.ckpt

Full pipeline test:
python test_pipeline.py

All 6 tests must PASS.

---

## Framework Notes

- Use lightning (NOT pytorch_lightning)
  import lightning as L

- Variable importance:
  model.interpret_output(raw_preds, reduction="sum")

- GPU RTX 3050
  Use .cpu().numpy() before numpy conversion

- Add:
  torch.set_float32_matmul_precision('medium')

---

## Key Files

src/data/spei.py  
→ Log-Logistic SPEI computation

src/models/dataset.py  
→ Encoder/pred length controlled here  
→ No partial windows

evaluate.py  
→ Step-0-only decoding  
→ PICP  
→ Naive baseline  

test_pipeline.py  
→ 6-stage validation suite  

logs/checkpoints/epoch=8-val_loss=0.37.ckpt  
→ Encoder=30 model

---

## Experimental Note

If retraining:
- encoder=30 → baseline model
- encoder=90 → ablation comparison

Conclusion must be empirical (RMSE comparison),
not theoretical assumption.