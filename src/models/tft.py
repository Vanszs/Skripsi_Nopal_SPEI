import torch
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

def build_tft_model(training_dataset, **kwargs):
    """
    Constructs the TFT model using the dataset properties.
    """
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.0001,  # Optimized LR for stable convergence
        hidden_size=128,      # Increased capacity
        attention_head_size=4,
        dropout=0.15,        # Optimized dropout for balanced regularization
        hidden_continuous_size=16,
        output_size=7,  # 7 quantiles by default for QuantileLoss
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        **kwargs
    )
    return model
