import torch
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

def build_tft_model(training_dataset, **kwargs):
    """
    Constructs the TFT model using the dataset properties.

    Hyperparameter rationale — "small data optimized" TFT
    (5 locations × ~6500 rows, encoder=30, horizon=30):
        hidden_size=48       : Reduced from 64. enc30 v1 (128) and v2 (64) both
                               overfit. 48 units further limits capacity while
                               retaining enough for SPEI-3 seasonal dynamics.
        dropout=0.35         : Raised from 0.30. Stronger stochastic regularisation.
        attention_head_size=1 : Single head. Multi-head attention is wasteful on
                               5-location small datasets; single head is sufficient.
        weight_decay=1e-4    : L2 penalty on weights. Prevents large weight growth
                               that drives train-val gap.
        learning_rate=3e-4   : Stable compromise: 1e-4 too slow, 1e-3 diverges.
        hidden_continuous_size=8 : Kept from previous. Proportional to hidden_size.
        reduce_on_plateau_patience=3 : Aggressive LR decay if val stalls.
    """
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=3e-4,
        hidden_size=48,
        attention_head_size=1,
        dropout=0.35,
        hidden_continuous_size=8,
        output_size=7,  # 7 quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=3,
        weight_decay=1e-4,
        **kwargs
    )
    return model
