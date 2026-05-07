import os
import time
import json
import warnings
from datetime import datetime

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet

from src.models.dataset import MODEL_GROUP_COL, create_dataset
from src.models.tft import build_tft_model

EXPECTED_SCHEMA_VERSION = 2
DEFAULT_SEED = 42


class EpochSummaryCallback(Callback):
    """Print one clean line per epoch instead of per-batch progress bars."""

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._t0 = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip lightning sanity-validation phase to avoid false NaN logs.
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        if "val_loss" not in metrics or "train_loss_epoch" not in metrics:
            return
        val_loss = metrics.get("val_loss")
        trn_loss = metrics.get("train_loss_epoch")
        elapsed = time.time() - getattr(self, "_t0", time.time())
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else float("nan")
        print(
            f"  Epoch {epoch:>3}  train={float(trn_loss):.4f}  val={float(val_loss):.4f}  "
            f"lr={lr:.2e}  ({elapsed:.0f}s)"
        )


def _validate_training_schema(data: pd.DataFrame):
    required_cols = {
        "schema_version",
        "time",
        "time_idx",
        "city_id",
        "super_node_id",
        "selected_node_count",
        "SPEI_3",
    }
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required training columns: {sorted(missing)}")

    versions = sorted(data["schema_version"].dropna().unique().tolist())
    if versions != [EXPECTED_SCHEMA_VERSION]:
        raise ValueError(
            f"Invalid processed schema version {versions}; expected [{EXPECTED_SCHEMA_VERSION}]. "
            "Run preprocess_pipeline() for schema v2."
        )

    if data[MODEL_GROUP_COL].nunique() == 0:
        raise ValueError("No super_node_id found in processed data.")

    bad_count = data.groupby("city_id")["selected_node_count"].max()
    if not (bad_count == 5).all():
        raise ValueError(f"selected_node_count must be 5 for all cities: {bad_count.to_dict()}")

    dup_keys = data.duplicated(subset=[MODEL_GROUP_COL, "time"]).sum()
    if dup_keys:
        raise ValueError(f"Duplicate ({MODEL_GROUP_COL}, time) rows found: {dup_keys}")


def train_pipeline(
    data_path="data/processed/spei_dataset.parquet",
    max_epochs=80,
    batch_size=32,
    max_encoder_length=30,
    hidden_size=48,
    dropout=0.35,
    attention_head_size=1,
    hidden_continuous_size=8,
    learning_rate=3e-4,
    weight_decay=1e-4,
    gradient_clip_val=0.5,
    seed=DEFAULT_SEED,
    run_config_path="logs/run_config.json",
    suppress_library_warnings=True,
    strict_determinism=False,
    run_id=None,
):
    effective_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if suppress_library_warnings:
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but StandardScaler was fitted with feature names",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Deterministic behavior was enabled with either .* but this operation is not deterministic .*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms.*",
            category=UserWarning,
        )

    L.seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if strict_determinism:
        # Required by PyTorch for deterministic CuBLAS operations on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)
    # Leverage tensor cores while keeping full precision training (fp32).
    torch.set_float32_matmul_precision("medium")
    print("Loading data for training...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}")

    data = pd.read_parquet(data_path)
    data["year"] = pd.to_datetime(data["time"]).dt.year
    _validate_training_schema(data)

    n_cities = data["city_id"].nunique()
    n_groups = data[MODEL_GROUP_COL].nunique()
    group_sizes = data.groupby(MODEL_GROUP_COL).size().sort_values()
    print(
        f"Schema v{EXPECTED_SCHEMA_VERSION} OK | cities={n_cities} "
        f"| model_entities={n_groups} | min_rows_per_entity={int(group_sizes.min())}"
    )

    training_cutoff = data[data.year < 2023]["time_idx"].max()
    validation_cutoff = data[data.year == 2023]["time_idx"].max()
    if pd.isna(training_cutoff) or pd.isna(validation_cutoff):
        raise ValueError("Train/validation split is empty. Check processed dataset years.")

    print(f"Training Cutoff Index: {int(training_cutoff)}")
    print(f"Encoder Length       : {max_encoder_length}")

    train_data = data[data.time_idx <= training_cutoff].copy()
    train_ds = create_dataset(train_data, max_encoder_length=max_encoder_length)

    val_start_idx = data[data.year == 2023]["time_idx"].min() - max_encoder_length
    val_data = data[(data.time_idx >= val_start_idx) & (data.time_idx <= validation_cutoff)]
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        val_data,
        predict=False,
        stop_randomization=True,
    )

    train_loader = train_ds.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=batch_size * 2,
        num_workers=0,
        pin_memory=True,
    )

    model = build_tft_model(
        train_ds,
        hidden_size=hidden_size,
        dropout=dropout,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=30, verbose=False, mode="min"),
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath="logs/checkpoints",
            filename=(f"enc{max_encoder_length}-run{effective_run_id}" + "-{epoch}-{val_loss:.4f}"),
            monitor="val_loss",
            save_top_k=1,
        ),
        EpochSummaryCallback(),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=32,
        enable_model_summary=True,
        enable_progress_bar=False,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        logger=TensorBoardLogger("logs/lightning_logs"),
    )

    print("Starting Training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Best model path: {trainer.checkpoint_callback.best_model_path}")

    os.makedirs(os.path.dirname(run_config_path), exist_ok=True)
    run_cfg = {
        "schema_version_expected": EXPECTED_SCHEMA_VERSION,
        "seed": seed,
        "data_path": data_path,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "max_encoder_length": max_encoder_length,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_val": gradient_clip_val,
        "accelerator": accelerator,
        "suppress_library_warnings": bool(suppress_library_warnings),
        "strict_determinism": bool(strict_determinism),
        "run_id": effective_run_id,
        "n_cities": int(n_cities),
        "n_entities": int(n_groups),
        "train_rows": int(len(train_data)),
        "val_rows": int(len(val_data)),
        "best_model_path": trainer.checkpoint_callback.best_model_path,
    }
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)
    print(f"Saved run config to {run_config_path}")
    return trainer.checkpoint_callback.best_model_path


if __name__ == "__main__":
    train_pipeline()
