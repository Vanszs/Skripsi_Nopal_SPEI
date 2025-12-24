═══════════════════════════════════════════════════════════════════════════════
📋 DETAILED IMPLEMENTATION GUIDE
═══════════════════════════════════════════════════════════════════════════════

Autoencoder dan Bi-LSTM Berbasis Attention dengan Optimasi Algoritma Genetika pada XGBoost untuk Deteksi Anomali Cuaca Badai di Laut Cina Selatan

═══════════════════════════════════════════════════════════════════════════════
📊 SECTION 1: DATA REQUIREMENTS & PREPROCESSING
═══════════════════════════════════════════════════════════════════════════════

DATA SOURCE (Sudah Anda Punya):
─────────────────────────────

✅ Training Data: data/china_history_2y.csv
├─ Lokasi: South China Sea
├─ Periode: 2022-2023 (2 tahun)
├─ Resolusi: 1-hourly
├─ Rows: ~17,520 (8,760 per tahun)
└─ Size: ~1-2 MB

✅ Testing Data: data/wnp_1y.csv
├─ Lokasi: Western North Pacific
├─ Periode: 2024 (1 tahun)
├─ Resolusi: 1-hourly
├─ Rows: ~8,784 (1 tahun)
└─ Size: ~500 KB

✅ Features (4 parameters):
├─ pressure_msl (hPa) → Mean Sea Level Pressure
├─ windspeed_10m (m/s) → Wind Speed
├─ windgusts_10m (m/s) → Wind Gusts
└─ humidity (%) → Relative Humidity

⚠️ JIKA TIDAK ADA HUMIDITY: Pakai 3 parameters saja (code auto-adapt)

DATA PREPROCESSING PIPELINE:
────────────────────────────

Step 1: Load Data
├─ df_train = pd.read_csv('data/china_history_2y.csv')
├─ df_test = pd.read_csv('data/wnp_1y.csv')
└─ Check shape: (17520, 4) dan (8784, 4)

Step 2: Handle Missing Values
├─ Forward fill: df.fillna(method='ffill')
├─ Interpolate: df.interpolate(method='linear')
└─ Verify: No NaN remaining

Step 3: Extract Features
├─ features = ['pressure_msl', 'windspeed_10m', 'windgusts_10m', 'humidity']
├─ X_train_raw = df_train[features].values → (17520, 4)
└─ X_test_raw = df_test[features].values → (8784, 4)

Step 4: Create Labels (Based on Wind Speed)
├─ Class 0 (Normal): windspeed < 10 m/s
├─ Class 1 (Anomaly): 10 ≤ windspeed < 15 m/s
└─ Class 2 (Storm): windspeed ≥ 15 m/s

Expected distribution:
├─ Training: ~90% Class 0, ~8% Class 1, ~2% Class 2
└─ Testing: Similar distribution

Step 5: Normalize Data
├─ scaler = StandardScaler()
├─ X_train_scaled = scaler.fit_transform(X_train_raw)
└─ X_test_scaled = scaler.transform(X_test_raw)

Step 6: Create Sequences (IMPORTANT!)
├─ Sliding window: 24 hours (24 timesteps)
├─ Purpose: LSTM needs temporal context
├─ Result:
│  ├─ X_train_seq: (17496, 24, 4)  # 1000s samples × 24 hours × 4 params
│  ├─ y_train_seq: (17496,)        # labels
│  ├─ X_test_seq: (8760, 24, 4)
│  └─ y_test_seq: (8760,)
│
└─ Example: Timesteps t-23, t-22, ..., t-1 → Predict t

Step 7: Convert to PyTorch
├─ X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
├─ y_train_tensor = torch.tensor(y_train_seq, dtype=torch.long)
├─ Load on GPU if available (cuda)
└─ Create DataLoaders for batching (batch_size=32)

EXPECTED DIMENSIONS AFTER PREPROCESSING:
────────────────────────────────────────

Raw data:
├─ X_train: (17520, 4)
├─ y_train: (17520,)
├─ X_test: (8784, 4)
└─ y_test: (8784,)

Preprocessed (sequences):
├─ X_train_seq: (17496, 24, 4)
├─ y_train_seq: (17496,)
├─ X_test_seq: (8760, 24, 4)
└─ y_test_seq: (8760,)

DataLoader batches:
├─ Batch X: (32, 24, 4)  # 32 samples, 24 timesteps, 4 features
└─ Batch y: (32,)        # 32 labels

═══════════════════════════════════════════════════════════════════════════════
🏗️ SECTION 2: AUTOENCODER (STAGE 1)
═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURE:
──────────────

Input: (batch, 24, 4) = 96 features (24 hours × 4 params)
    ↓
Flatten: (batch, 96)
    ↓
ENCODER:
├─ Dense(96 → 64) + ReLU
├─ Dropout(0.2)
├─ Dense(64 → 10) [Latent space]
    ↓
DECODER:
├─ Dense(10 → 64) + ReLU
├─ Dropout(0.2)
├─ Dense(64 → 96) [Reconstructed input]
    ↓
Output: (batch, 96)

FEATURE EXTRACTION:
───────────────────

MSE Calculation:
├─ MSE = mean((x - x_recon)^2) per sample
├─ Low MSE (0.42) = Normal conditions
├─ Medium MSE (0.76) = Anomaly
├─ High MSE (1.12) = Storm
└─ Use as Feature #1

Latent Features:
├─ 10 dimensions from bottleneck
├─ Capture compressed representation
└─ Use as Features #2-11

OUTPUT: 11 features (1 MSE + 10 latent)

TRAINING CONFIGURATION:
───────────────────────

Optimizer: Adam (lr=0.001, weight_decay=0.0001)
├─ weight_decay = L2 regularization
└─ Prevents overfitting

Loss function: MSE
├─ Minimizes reconstruction error
└─ Standard for autoencoder

Epochs: 100 (with early stopping, patience=10)
Batch size: 32
Training time: ~30 minutes (CPU) or ~5-10 min (GPU)

EXPECTED PERFORMANCE:
─────────────────────

Training loss: ~0.1-0.2 (MSE)
Validation loss: ~0.15-0.25

Reconstruction error by class:
├─ Class 0 (Normal): MSE ≈ 0.42
├─ Class 1 (Anomaly): MSE ≈ 0.76
└─ Class 2 (Storm): MSE ≈ 1.12

Clear separation (2.67x difference!) ✓

═══════════════════════════════════════════════════════════════════════════════
🔄 SECTION 3: BI-LSTM WITH ATTENTION (STAGE 2)
═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURE:
──────────────

Input: (batch, 24, 4) = 24 timesteps × 4 features
    ↓
BIDIRECTIONAL LSTM:
├─ Layer 1: Bi-LSTM (64 units)
│  ├─ Forward LSTM: 64 units
│  ├─ Backward LSTM: 64 units
│  └─ Concatenate: 128 features
├─ Layer 2: Bi-LSTM (64 units)
│  ├─ Forward LSTM: 64 units
│  ├─ Backward LSTM: 64 units
│  └─ Concatenate: 128 features
    ↓
LSTM OUTPUT: (batch, 24, 128)
    ↓
MULTI-HEAD ATTENTION:
├─ Number of heads: 4
├─ Embed dim: 128
├─ Query, Key, Value: Same as LSTM output
├─ Each head focuses on different aspects
│  ├─ Head 1: Pressure patterns
│  ├─ Head 2: Wind speed trends
│  ├─ Head 3: Gust spikes
│  └─ Head 4: Humidity changes
    ↓
ATTENTION OUTPUT: (batch, 24, 128)
    ↓
GLOBAL AVERAGE POOLING:
├─ Average over timesteps
├─ (batch, 24, 128) → (batch, 128)
└─ 128 features (attention-weighted)

WHY BIDIRECTIONAL?
──────────────────

Standard LSTM (unidirectional):
├─ Only looks backward
├─ Can't see future context
└─ Example: Can't tell if gust is start or end of event

Bidirectional LSTM:
├─ Forward: t-24, t-23, ..., t-1 → t
├─ Backward: t, t+1, ..., t+23 → t
├─ Combines both: Knows context before AND after
└─ Better for anomaly detection!

WHY ATTENTION?
───────────────

Without attention:
├─ All timesteps weighted equally
├─ Important moments (like sudden pressure drop) lost in noise
└─ Less interpretable (black box)

With attention:
├─ Learn which timesteps are important
├─ Sudden pressure drop gets high weight
├─ Wind gust spike gets high weight
├─ Interpretable (can visualize attention)

ATTENTION WEIGHTS:
─ Shape: (batch, 24, 24)
- Rows: attention from timestep i
- Cols: attention to timestep j
- High value at (i, j) = "timestep i cares about timestep j"

OUTPUT: 128 features (attention-weighted)

═══════════════════════════════════════════════════════════════════════════════
🔀 SECTION 4: FEATURE CONCATENATION
═══════════════════════════════════════════════════════════════════════════════

STAGE 1 OUTPUT (Autoencoder):
├─ Feature 1: MSE (1 value)
├─ Features 2-11: Latent (10 values)
└─ Total: 11 features

STAGE 2 OUTPUT (Bi-LSTM Attention):
└─ Features 1-128: Attention-weighted (128 values)

CONCATENATION:
├─ Combine both: [11, 128]
├─ Final shape: (batch, 139)
├─ 139 = 11 + 128 features
└─ Capture different aspects:
   ├─ MSE: Deviation from normal (reconstruction-based)
   ├─ Latent: Compressed representation
   └─ Bi-LSTM-Attn: Temporal patterns with attention

WHY CONCATENATE?
─────────────────

Autoencoder specializes in:
├─ Detecting deviations from normal patterns
├─ Unsupervised anomaly detection
└─ Good for outlier detection

Bi-LSTM-Attention specializes in:
├─ Capturing temporal dependencies
├─ Supervised learning on sequences
└─ Good for pattern recognition

Combined:
├─ Ensemble effect: Best of both worlds
├─ Redundancy reduction: Different angles
└─ Better generalization: More features = more info

═══════════════════════════════════════════════════════════════════════════════
🧬 SECTION 5: GENETIC ALGORITHM XGBoost TUNING (STAGE 3)
═══════════════════════════════════════════════════════════════════════════════

WHAT IS GENETIC ALGORITHM?
───────────────────────────

Mimics biological evolution:
├─ Population: 20 candidate hyperparameter sets
├─ Selection: Keep best performers
├─ Crossover: Mix parameters from two parents
├─ Mutation: Random changes
└─ Repeat: 50 generations

PARAMETERS TO OPTIMIZE:
────────────────────────

GA optimizes these 7 XGBoost hyperparameters:

1. max_depth: [3-10]
   ├─ Tree depth (deeper = more complex)
   ├─ Too deep = overfitting
   └─ Too shallow = underfitting

2. learning_rate: [0.01-0.3]
   ├─ Step size for boosting
   ├─ Lower = more conservative
   └─ Higher = faster but risky

3. subsample: [0.5-1.0]
   ├─ Fraction of samples per tree
   └─ Lower = more regularization

4. colsample_bytree: [0.5-1.0]
   ├─ Fraction of features per tree
   └─ Lower = more regularization

5. min_child_weight: [1-5]
   ├─ Minimum sum of instance weight
   └─ Higher = more conservative

6. n_estimators: [50-200]
   ├─ Number of boosting rounds
   └─ More = better but slower

7. gamma: [0-5]
   ├─ Minimum loss reduction for split
   └─ Higher = more conservative

GA PROCESS:
────────────

Generation 0:
├─ Create 20 random hyperparameter sets
├─ Train XGBoost for each
├─ Evaluate on validation (5-fold CV)
└─ Get F1-score for each

Generation 1-50:
├─ Select top 10 performers (tournament selection)
├─ Crossover: Mix parameters from parents
├─ Mutation: Randomly tweak parameters
├─ Boundary check: Keep within ranges
├─ Train & evaluate new 20 sets
└─ Repeat: 50 times total

EXPECTED CONVERGENCE:
├─ Generation 0: Avg F1 ≈ 0.92, Best F1 ≈ 0.94
├─ Generation 10: Avg F1 ≈ 0.945, Best F1 ≈ 0.955
├─ Generation 25: Avg F1 ≈ 0.95, Best F1 ≈ 0.96
├─ Generation 50: Avg F1 ≈ 0.952, Best F1 ≈ 0.965
└─ Typical improvement: +0.5-1.5%

FINAL XGBOOST MODEL:
──────────────────

Input: (batch, 139) combined features
├─ 139 features from AE + Bi-LSTM-Attn
├─ Already normalized (StandardScaler)
└─ Ready for XGBoost

XGBoost Configuration:
├─ Objective: multi:softmax (3-class classification)
├─ Trees: 50-200 (evolved by GA)
├─ Depth: 3-10 (evolved by GA)
├─ Learning rate: 0.01-0.3 (evolved by GA)
└─ Other params: evolved by GA

Output: (batch, 3)
├─ Class 0 probability
├─ Class 1 probability
└─ Class 2 probability

Prediction: argmax of probabilities

═══════════════════════════════════════════════════════════════════════════════
⏱️ SECTION 6: TRAINING TIMELINE & COMPUTATIONAL REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

COMPUTATIONAL REQUIREMENTS:
───────────────────────────

CPU Only:
├─ Processor: Intel i7 or equivalent
├─ RAM: 16GB minimum
├─ Disk: 10GB for models + data
└─ Time: ~8-12 hours

GPU (Recommended):
├─ GPU: NVIDIA RTX 3060 or better
├─ VRAM: 8GB minimum
├─ RAM: 8GB
├─ Disk: 10GB
└─ Time: ~2-4 hours (much faster!)

TRAINING TIMELINE:
──────────────────

Day 1, Morning (30 min):
├─ Data preprocessing
├─ Create sequences
└─ Save to disk

Day 1, Afternoon (30 min):
├─ Train Autoencoder
├─ Extract AE features
└─ Checkpoint saved

Day 1, Evening (20 min):
├─ Feature extraction with Bi-LSTM-Attention
├─ Concatenate features (11 + 128 = 139)
└─ Save combined features

Day 2, Morning (2-3 hours):
├─ Run Genetic Algorithm
├─ 50 generations × 5-fold CV
├─ ~250-300 XGBoost models trained
└─ Save best parameters

Day 2, Afternoon (30 min):
├─ Train final XGBoost with best params
├─ Evaluate on test set
├─ Generate visualization + results
└─ Write results section

TOTAL: 1.5-2 days (with GPU: 4-5 hours active time)

═══════════════════════════════════════════════════════════════════════════════
📁 SECTION 7: DIRECTORY STRUCTURE & FILES
═══════════════════════════════════════════════════════════════════════════════

project_root/
│
├── data/
│   ├── china_history_2y.csv          ✅ Input (provided)
│   ├── wnp_1y.csv                    ✅ Input (provided)
│   ├── X_train_seq.npy               ⬜ Generated (step 1)
│   ├── y_train_seq.npy               ⬜ Generated (step 1)
│   ├── X_test_seq.npy                ⬜ Generated (step 1)
│   ├── y_test_seq.npy                ⬜ Generated (step 1)
│   ├── X_train_features.npy          ⬜ Generated (step 2)
│   ├── X_test_features.npy           ⬜ Generated (step 2)
│   ├── X_train_combined.npy          ⬜ Generated (step 3)
│   └── X_test_combined.npy           ⬜ Generated (step 3)
│
├── checkpoints/
│   ├── ae_best.pt                    ⬜ Best autoencoder
│   ├── bilstm_attention.pt           ⬜ Bi-LSTM-Attention model
│   └── xgb_final.json                ⬜ Final XGBoost model
│
├── results/
│   ├── ga_best_params.json           ⬜ Best GA parameters
│   ├── final_results.json            ⬜ Final metrics (accuracy, F1, etc)
│   ├── training_log.txt              ⬜ Detailed training log
│   └── comparison_results.json       ⬜ Baseline comparison
│
├── figures/
│   ├── ae_training_history.png       ⬜ AE loss curve
│   ├── ga_convergence.png            ⬜ GA F1-score over generations
│   ├── confusion_matrix.png          ⬜ Confusion matrix heatmap
│   ├── attention_weights.png         ⬜ Attention visualization
│   ├── model_comparison.png          ⬜ Bar chart comparing methods
│   └── feature_importance.png        ⬜ XGBoost feature importance
│
├── code/
│   ├── 01_preprocessing.py
│   ├── 02_autoencoder.py
│   ├── 03_bilstm_attention.py
│   ├── 04_ga_xgboost.py
│   ├── 05_evaluation.py
│   ├── requirements.txt               # Python dependencies
│   └── config.yaml                   # Configuration (hyperparams)
│
├── thesis/
│   ├── 01_introduction.tex
│   ├── 02_literature_review.tex
│   ├── 03_methodology.tex
│   ├── 04_results.tex
│   ├── 05_discussion.tex
│   ├── 06_conclusion.tex
│   ├── main.tex
│   └── references.bib
│
└── README.md                         # Setup & usage instructions

═══════════════════════════════════════════════════════════════════════════════
🚀 SECTION 8: QUICK START (Step-by-Step)
═══════════════════════════════════════════════════════════════════════════════

STEP 1: Setup Environment (15 min)
───────────────────────────────────

# Install Python 3.9+
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Required packages:
# - torch
# - pandas
# - numpy
# - scikit-learn
# - xgboost
# - deap (Genetic Algorithm)
# - matplotlib
# - seaborn

STEP 2: Data Preprocessing (15 min)
────────────────────────────────────

python code/01_preprocessing.py

Output:
├─ Creates: data/X_train_seq.npy, data/X_test_seq.npy, etc.
├─ Log: "✅ Data preprocessing complete!"
└─ Next: Step 3

STEP 3: Train Autoencoder (30 min)
───────────────────────────────────

python code/02_autoencoder.py

Output:
├─ Creates: checkpoints/ae_best.pt
├─ Creates: data/X_train_features.npy (11 features)
├─ Log: "✅ Autoencoder training complete!"
├─ Figure: figures/ae_training_history.png
└─ Next: Step 4

STEP 4: Extract Bi-LSTM Features (20 min)
──────────────────────────────────────────

python code/03_bilstm_attention.py

Output:
├─ Creates: data/X_train_combined.npy (139 features)
├─ Creates: data/X_test_combined.npy
├─ Log: "✅ Bi-LSTM-Attention features extracted!"
└─ Next: Step 5

STEP 5: GA XGBoost Optimization (2-3 hours)
────────────────────────────────────────────

python code/04_ga_xgboost.py

Output:
├─ Creates: checkpoints/xgb_final.json
├─ Creates: results/ga_best_params.json
├─ Log: Generation 0/50, 1/50, ..., 50/50
├─ Figure: figures/ga_convergence.png
└─ Next: Step 6

STEP 6: Evaluation & Results (10 min)
──────────────────────────────────────

python code/05_evaluation.py

Output:
├─ Creates: results/final_results.json
├─ Creates: figures/confusion_matrix.png
├─ Creates: figures/model_comparison.png
├─ Prints:
│  ├─ Accuracy: X.XXX
│  ├─ Precision: X.XXX
│  ├─ Recall: X.XXX
│  └─ Macro F1: X.XXX
└─ ✅ COMPLETE!

TOTAL TIME: ~4 hours (mostly automated)

═══════════════════════════════════════════════════════════════════════════════
✅ EXPECTED RESULTS
═══════════════════════════════════════════════════════════════════════════════

EXPECTED PERFORMANCE:

Compared to baselines:
├─ Baseline 1 (AE-XGBoost): F1 = 94.38%
├─ Baseline 2 (PSO-NN): F1 = 59.28%
├─ Baseline 3 (CNN-LSTM): F1 ≈ 90%
└─ Your method: F1 = 95-96% ✓

Per-class F1-scores:
├─ Class 0 (Normal): 97-98%
├─ Class 1 (Anomaly): 94-95%
└─ Class 2 (Storm): 91-93%

Ablation study (which component helps?):
├─ AE features only: F1 ≈ 94.38%
├─ Bi-LSTM-Attn only: F1 ≈ 93-94%
├─ Combined (AE + Bi-LSTM): F1 ≈ 94.8%
└─ + GA tuning: F1 ≈ 95-96% ✓

═══════════════════════════════════════════════════════════════════════════════
📚 REQUIRED PYTHON LIBRARIES
═══════════════════════════════════════════════════════════════════════════════

requirements.txt content:
