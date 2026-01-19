
import os
import sys
import torch
from pytorch_forecasting import TemporalFusionTransformer

# Add project root to path
sys.path.insert(0, os.path.abspath('d:/SKRIPSI/Skripsi_Nopal'))

checkpoint_path = "d:/SKRIPSI/Skripsi_Nopal/logs/checkpoints/epoch=8-val_loss=0.37.ckpt"

if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

print(f"Loading checkpoint: {checkpoint_path}")
try:
    # Load with map_location to CPU to avoid GPU OOM if another process is running
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location="cpu")
    print("Model loaded successfully.")
    
    print("\n--- Model Hyperparameters ---")
    print(f"max_encoder_length: {model.hparams.max_encoder_length}")
    print(f"learning_rate: {model.hparams.learning_rate}")
    print(f"dropout: {model.hparams.dropout}")
    print(f"hidden_size: {model.hparams.hidden_size}")
    
    # Check if this matches optimization target
    target_encoder = 30
    target_lr = 0.0001
    target_dropout = 0.15
    
    matches = (
        model.hparams.max_encoder_length == target_encoder and
        abs(model.hparams.learning_rate - target_lr) < 1e-6 and
        abs(model.hparams.dropout - target_dropout) < 1e-6
    )
    
    if matches:
        print("\n✅ Checkpoint matches optimized configuration!")
    else:
        print("\n❌ Checkpoint DOES NOT match optimized configuration.")
        
except Exception as e:
    print(f"Error loading model: {e}")
