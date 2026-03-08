import torch
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

def build_tft_model(
    training_dataset,
    hidden_size: int = 48,
    dropout: float = 0.35,
    attention_head_size: int = 1,
    hidden_continuous_size: int = 8,
    learning_rate: float = 3e-4,
    reduce_on_plateau_patience: int = 8,
    weight_decay: float = 1e-4,
    **kwargs
):
    """
    Constructs the TFT model using the dataset properties.

    Parameters are fully configurable to support ablation experiments:

        hidden_size (default 48):
            Baseline uses 48 (enc=30). Experiment C uses 64 to give enc=90
            model more representational capacity without overfit risk.
        dropout (default 0.35):
            Baseline uses 0.35 (enc=30). Experiment C lowers to 0.20 since
            enc=90 naturally regularises via longer context.
        attention_head_size (default 1):
            Baseline uses 1 (adequate for 5 locations). Experiment C uses 2
            to capture richer temporal attention patterns.
        hidden_continuous_size (default 8):
            Proportional to hidden_size; typically hidden_size // 6.
        learning_rate (default 3e-4):
            Stable compromise between 1e-4 (too slow) and 1e-3 (diverges).
        reduce_on_plateau_patience (default 8):
            How many val epochs without improvement before LR is halved.
            Raised from 3→5→8: with a proper validation set (~1 600 samples)
            the plateau detector is now meaningful; 8 gives the optimizer
            enough room to escape local flat spots before committing to a
            smaller step size.
        weight_decay (default 1e-4):
            L2 penalty — prevents large weight growth driving train-val gap.
            Raise to 1e-3 when overfitting is observed (val rising while
            train falls).
    """
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=7,  # 7 quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=reduce_on_plateau_patience,
        weight_decay=weight_decay,
        **kwargs
    )
    return model
