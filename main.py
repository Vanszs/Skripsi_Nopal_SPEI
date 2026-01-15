import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

from src.data.ingest import main as ingest_data
from src.data.preprocess import preprocess_pipeline
from src.training.train import train_pipeline

def main():
    print("=== STARTING PIPELINE ===")
    
    print("\n--- STEP 1: INGESTION ---")
    try:
        if not os.path.exists("data/raw/weather_history_east_java.parquet"):
             ingest_data()
        else:
             print("Data already exists. Skipping ingestion (delete data/raw to force).")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        return

    print("\n--- STEP 2: PREPROCESSING ---")
    try:
        preprocess_pipeline()
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return

    print("\n--- STEP 3: TRAINING ---")
    try:
        best_model_path = train_pipeline(max_epochs=20) # Lower epochs for default run
        print(f"Training completed. Model saved at {best_model_path}")
    except Exception as e:
        print(f"Training failed: {e}")
        # Continue execution to allow debugging? No, strict pipeline.
        return

    print("\n=== PIPELINE FINISHED ===")
    print("Now run the visualization notebook: notebooks/Visualization_and_Evaluation.ipynb")

if __name__ == "__main__":
    main()
