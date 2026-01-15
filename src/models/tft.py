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
        learning_rate=0.001,  # Lower LR for stability
        hidden_size=128,      # Increased capacity
        attention_head_size=4,
        dropout=0.3,         # Increased dropout to 0.3 to fix overfitting
        hidden_continuous_size=16,
        output_size=7,  # 7 quantiles by default for QuantileLoss
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        **kwargs
    )
    return model
