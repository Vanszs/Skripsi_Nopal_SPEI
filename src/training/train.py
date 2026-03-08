import os
import time
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd
from src.models.dataset import create_dataset
from src.models.tft import build_tft_model


class EpochSummaryCallback(Callback):
    """Print one clean line per epoch instead of thousands of batch progress bars."""

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch    = trainer.current_epoch
        val_loss = metrics.get("val_loss",  float("nan"))
        trn_loss = metrics.get("train_loss_epoch", float("nan"))
        elapsed  = time.time() - getattr(self, "_t0", time.time())
        # Format as a single readable line
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else float("nan")
        print(
            f"  Epoch {epoch:>3}  "
            f"train={float(trn_loss):.4f}  "
            f"val={float(val_loss):.4f}  "
            f"lr={lr:.2e}  "
            f"({elapsed:.0f}s)"
        )

def train_pipeline(data_path="data/processed/spei_dataset.parquet",
                   max_epochs=80,
                   batch_size=32,
                   max_encoder_length=90,
                   hidden_size=48,
                   dropout=0.35,
                   attention_head_size=1,
                   hidden_continuous_size=8,
                   learning_rate=3e-4,
                   weight_decay=1e-4,
                   gradient_clip_val=0.5):
    
    
    print("Loading data for training...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}")
        
    data = pd.read_parquet(data_path)
    
    # Split Data (Cutoff based on time_idx or Year)
    # Strategy: Last 2 years = Validation + Test
    # Validation = 2023, Test = 2024-2025
    # We use 'year' column if available or time index.
    # Assuming 'time' column exists and is datetime
    data['year'] = data['time'].dt.year
    
    # Train: < 2023
    # Val: == 2023
    # Test: >= 2024 (Held out completely from this script usually, or used for final eval)
    
    training_cutoff = data[data.year < 2023]["time_idx"].max()
    validation_cutoff = data[data.year == 2023]["time_idx"].max()
    
    print(f"Training Cutoff Index: {training_cutoff}")
    print(f"Encoder Length       : {max_encoder_length}")

    # Create Dataset
    # Filter only relevant data to avoid confusing the dataset creator
    # train_cutoff covers all history up to end of training
    train_ds = create_dataset(
        data[data.time_idx <= training_cutoff],
        max_encoder_length=max_encoder_length
    )

    # Validation Dataset (only 2023 windows, with enough encoder history)
    # ─────────────────────────────────────────────────────────────────────
    # BUG FIXED: previous code used data[time_idx <= validation_cutoff]
    # which included ALL training years → validation loss was partially
    # computed on already-seen data (leakage) AND with predict=True on
    # only 5 groups it produced as few as 5–60 samples (1-2 val batches)
    # causing huge run-to-run variance (0.18 ↔ 0.40 at epoch=1).
    #
    # Fix: restrict to 2023 data + encoder_length rows of history so the
    # val windows only cover unseen 2023 timestamps.  predict=False gives
    # full rolling windows (~1 600 samples, ~25 stable batches).
    val_start_idx = data[data.year == 2023]["time_idx"].min() - max_encoder_length
    val_data = data[
        (data.time_idx >= val_start_idx) &
        (data.time_idx <= validation_cutoff)
    ]
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        val_data,
        predict=False,       # full rolling windows → stable val metric
        stop_randomization=True,
    )
    
    # Dataloaders - Optimized for GPU (RTX 3050)
    train_dataloader = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0, pin_memory=True)
    val_dataloader = val_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0, pin_memory=True)
    
    # Build Model
    model = build_tft_model(
        train_ds,
        hidden_size=hidden_size,
        dropout=dropout,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=30, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        dirpath="logs/checkpoints",
        filename=f"enc{max_encoder_length}" + "-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1
    )
    epoch_summary = EpochSummaryCallback()

    # Trainer - Optimized for RTX 3050 GPU
    # enable_progress_bar=False suppresses the per-batch tqdm lines that
    # produce hundreds of MB of log noise; EpochSummaryCallback replaces them.
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        precision=32,
        enable_model_summary=True,
        enable_progress_bar=False,
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop_callback, lr_logger, checkpoint_callback, epoch_summary],
        logger=TensorBoardLogger("logs/lightning_logs")
    )
    
    print("Starting Training...")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # Best Path
    print(f"Best model path: {trainer.checkpoint_callback.best_model_path}")
    
    return trainer.checkpoint_callback.best_model_path

if __name__ == "__main__":
    train_pipeline()
