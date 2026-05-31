import torch
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

def build_tft_model(
    training_dataset,
    hidden_size: int = 64,
    dropout: float = 0.20,
    attention_head_size: int = 2,
    hidden_continuous_size: int = 10,
    learning_rate: float = 3e-4,
    reduce_on_plateau_patience: int = 8,
    weight_decay: float = 1e-4,
    **kwargs
):
    """
    Constructs the TFT model using the dataset properties.

    Parameters are fully configurable to support ablation experiments:

        hidden_size (default 64):
            enc=90 baseline uses 64 to give the longer context window more
            representational capacity without overfit risk.
        dropout (default 0.20):
            enc=90 baseline uses 0.20 since the longer encoder naturally
            regularises via richer context.
        attention_head_size (default 2):
            2 heads to capture richer temporal attention patterns across the
            90-day encoder window.
        hidden_continuous_size (default 10):
            Proportional to hidden_size; typically hidden_size // 6.
        learning_rate (default 3e-4):
            Stable compromise between 1e-4 (too slow) and 1e-3 (diverges).
        reduce_on_plateau_patience (default 8):
            How many val epochs without improvement before LR is halved.
            8 gives the optimizer enough room to escape local flat spots.
        weight_decay (default 1e-4):
            L2 penalty — prevents large weight growth driving train-val gap.
            Raise to 1e-3 when overfitting is observed.
    """
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=3,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        log_interval=10,
        reduce_on_plateau_patience=reduce_on_plateau_patience,
        weight_decay=weight_decay,
        **kwargs
    )
    return model
