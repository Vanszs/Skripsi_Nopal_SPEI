import json
import os

notebook_path = r'd:\SKRIPSI\Skripsi_Nopal\cnn_lstm_ea_multivariate_weather.ipynb'

def create_code_cell(source_code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_code.splitlines(True)
    }

metrics_code = """from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# --- METRIC CALCULATION CELL ---
# IMPORTANT: Ensure you have executed the prediction cells above 
# (where actuals_2d and predictions_2d are created) before running this!

# Ensure target_cols is defined for robustness
if 'target_cols' not in locals():
    # Standard headers for this dataset
    target_cols = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed']

if 'actuals_2d' not in locals() or 'predictions_2d' not in locals():
    print("❌ ERROR: 'actuals_2d' or 'predictions_2d' variables are missing.")
    print("   Please scroll up and RUN the cell that performs the predictions (Step 4 / Inference).")
    print("   Then run this cell again.")
else:
    print("\\n--- Detailed Test Set Metrics ---")
    print(f"{'Variable':<15} {'MAE':<10} {'RMSE':<10} {'R2 Score':<10}")
    print("-" * 55)

    for i, col in enumerate(target_cols):
        act = actuals_2d[:, i]
        pred = predictions_2d[:, i]
        
        mae = mean_absolute_error(act, pred)
        rmse = np.sqrt(mean_squared_error(act, pred))
        r2 = r2_score(act, pred)
        
        print(f"{col:<15} {mae:.4f}     {rmse:.4f}     {r2:.4f}")
"""

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Filter out existing metric cells to avoid duplicates or broken versions
    # We look for unique signatures in the source code
    new_cells = []
    removed_count = 0
    
    for cell in nb['cells']:
        source = "".join(cell.get('source', []))
        # Remove the cell with the Indonesian comment causing NameError
        if "Pastikan cell evaluasi sebelumnya sudah dijalankan" in source:
            removed_count += 1
            continue
        # Remove my previous attempt to avoid duplicates
        if "Ensure target_cols is defined for robustness" in source:
            removed_count += 1
            continue
        new_cells.append(cell)
    
    nb['cells'] = new_cells
    
    # Append the clean, robust cell
    nb['cells'].append(create_code_cell(metrics_code))
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
        
    print(f"Successfully cleaned {removed_count} old cells and added the robust metrics cell.")

except Exception as e:
    print(f"Error editing notebook: {e}")
