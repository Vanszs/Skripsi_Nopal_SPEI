import torch
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


def load_tft_checkpoint(checkpoint_path, map_location="cpu"):
    """
    Load a TFT checkpoint robustly across torch versions (R2 fix).

    torch>=2.6 defaults torch.load(weights_only=True), which rejects the
    EncoderNormalizer/scaler objects pickled in our checkpoints. These are
    our own trusted local files, so we allow full unpickling.
    """
    try:
        return TemporalFusionTransformer.load_from_checkpoint(
            checkpoint_path, map_location=map_location
        )
    except Exception:
        _orig_load = torch.load

        def _trusted_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        torch.load = _trusted_load
        try:
            return TemporalFusionTransformer.load_from_checkpoint(
                checkpoint_path, map_location=map_location
            )
        finally:
            torch.load = _orig_load

def build_tft_model(
    training_dataset,
    hidden_size: int = 48,
    dropout: float = 0.40,
    attention_head_size: int = 1,
    hidden_continuous_size: int = 8,
    learning_rate: float = 3e-4,
    reduce_on_plateau_patience: int = 8,
    weight_decay: float = 1e-4,
    log_interval: int = 10,
    **kwargs
):
    """
    Constructs the TFT model using the dataset properties.

    Parameters are fully configurable to support ablation experiments:

        hidden_size (default 48):
            Proven-healthy capacity for the 5-entity dataset; larger sizes
            (64) overfit from epoch 0 on so few groups.
        dropout (default 0.40):
            Strong dropout is the primary regularizer for this small-entity
            setup; 0.40 reduces train-val gap further than 0.35 (verified
            v32 vs v31) while maintaining healthy convergence.
        attention_head_size (default 1):
            Single head is sufficient and avoids extra memorization paths
            over the 90-day encoder window.
        hidden_continuous_size (default 8):
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
        log_interval=log_interval,
        reduce_on_plateau_patience=reduce_on_plateau_patience,
        weight_decay=weight_decay,
        **kwargs
    )
    return model
