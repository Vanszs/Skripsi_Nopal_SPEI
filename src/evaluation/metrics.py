import torch
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer

def load_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model

def calculate_metrics(actuals, predictions):
    """
    Calculate RMSE and MAE.
    predictions: Tensor of shape (batch, horizons, quantiles) -> We typically use P50 (index 3 or 4) for point metrics.
    """
    # Assuming P50 is at index 3 (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9) - Check config!
    # Default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98] -> Index 3 is 0.5
    
    p50_pred = predictions[:, :, 3] 
    
    mse = torch.mean((actuals - p50_pred) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(actuals - p50_pred))
    
    return {
        "RMSE": rmse.item(),
        "MAE": mae.item()
    }

def get_variable_importance(model, dataset):
    """
    Extracts variable importance using the model's interpretation hooks.
    """
    interpretation = model.interpret_output(
        dataset.to_dataloader(train=False, batch_size=128).next(), # Get one batch
        reduction="sum"
    )
    return interpretation["encoder_variables"], interpretation["decoder_variables"]


def plot_variable_importance(model, dataloader, save_path="results/variable_importance.png"):
    """
    Plots variable importance from the TFT model's Variable Selection Network.
    This is a KEY NOVELTY feature for interpretability.
    """
    import matplotlib.pyplot as plt
    import os
    
    # Get interpretation from model
    raw_predictions, x = model.predict(dataloader, mode="raw", return_x=True)
    interpretation = model.interpret_output(raw_predictions, reduction="sum")
    
    # Extract encoder and decoder variable importance
    encoder_importance = interpretation["encoder_variables"]
    decoder_importance = interpretation["decoder_variables"]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot encoder variables
    encoder_vars = list(encoder_importance.keys())
    encoder_vals = [encoder_importance[k].item() for k in encoder_vars]
    
    ax1 = axes[0]
    bars1 = ax1.barh(encoder_vars, encoder_vals, color='steelblue')
    ax1.set_xlabel("Importance Score")
    ax1.set_title("Encoder Variable Importance (Past Inputs)")
    ax1.invert_yaxis()
    
    # Plot decoder variables
    decoder_vars = list(decoder_importance.keys())
    decoder_vals = [decoder_importance[k].item() for k in decoder_vars]
    
    ax2 = axes[1]
    bars2 = ax2.barh(decoder_vars, decoder_vals, color='darkorange')
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Decoder Variable Importance (Future Inputs)")
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Variable importance plot saved to {save_path}")
    return encoder_importance, decoder_importance


def plot_attention_weights(model, dataloader, save_path="results/attention_weights.png", sample_idx=0):
    """
    Plots the temporal attention weights from the TFT model.
    This visualizes which time steps the model focuses on for predictions.
    WAJIB untuk interpretability sesuai agent.md
    """
    import matplotlib.pyplot as plt
    import os
    
    # Get raw predictions with attention
    raw_predictions, x = model.predict(dataloader, mode="raw", return_x=True)
    interpretation = model.interpret_output(raw_predictions, reduction="none")
    
    # Extract attention weights
    attention = interpretation["attention"]  # Shape: (batch, prediction_horizon, encoder_length)
    
    # Select a sample for visualization
    sample_attention = attention[sample_idx].detach().cpu().numpy()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(sample_attention, aspect='auto', cmap='YlOrRd')
    
    ax.set_xlabel("Encoder Time Steps (Past Days)")
    ax.set_ylabel("Decoder Time Steps (Forecast Horizon)")
    ax.set_title("Temporal Attention Weights - TFT Multi-Head Attention")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight")
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Attention weights plot saved to {save_path}")
    return attention

