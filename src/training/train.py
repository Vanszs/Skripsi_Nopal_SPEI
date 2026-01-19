import os
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd
from src.models.dataset import create_dataset
from src.models.tft import build_tft_model

def train_pipeline(data_path="data/processed/spei_dataset.parquet", 
                   max_epochs=60, 
                   batch_size=32):
    
    
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
    
    # Create Dataset
    # Filter only relevant data to avoid confusing the dataset creator
    # train_cutoff covers all history up to end of training
    train_ds = create_dataset(data[data.time_idx <= training_cutoff])
    
    # Validation Dataset (Rolling origin from training)
    # We validate on 2023 data using history from Train
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds, 
        data[data.time_idx <= validation_cutoff], 
        predict=True, 
        stop_randomization=True
    )
    
    # Dataloaders - Optimized for GPU (RTX 3050)
    train_dataloader = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0, pin_memory=True)
    val_dataloader = val_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0, pin_memory=True)
    
    # Build Model
    model = build_tft_model(train_ds)
    
    # Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=25, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        dirpath="logs/checkpoints", 
        filename="{epoch}-{val_loss:.2f}", 
        monitor="val_loss",
        save_top_k=1
    )
    
    # Trainer - Optimized for RTX 3050 GPU
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",  # Force GPU
        devices=1,
        precision=32,  # Full precision - more stable for TFT
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_logger, checkpoint_callback],
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
