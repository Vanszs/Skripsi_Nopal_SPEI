import pandas as pd
import numpy as np
import os
from .spei import calculate_water_deficit, calculate_spei

def preprocess_pipeline(input_path="data/raw/weather_history_east_java.parquet",
                        output_path="data/processed/spei_dataset.parquet"):
    
    print("Loading raw data...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data not found at {input_path}. Run ingest.py first.")
        
    df = pd.read_parquet(input_path)
    # Ensure time is datetime
    df["time"] = pd.to_datetime(df["time"])
    
    # 1. Calculate SPEI for each location
    print("Calculating SPEI...")
    processed_dfs = []
    
    for loc_id, group in df.groupby("location_id"):
        group = group.sort_values("time").copy()
        
        # Missing Value Handling (Linear Interpolation) - WAJIB "STEP 2"
        # Interpolate only numeric columns
        numeric_cols = group.select_dtypes(include=[np.number]).columns
        group[numeric_cols] = group[numeric_cols].interpolate(method='linear', limit_direction='both')
        
        # Calculate Deficit
        group["water_deficit"] = calculate_water_deficit(group)
        
        # Calculate SPEI-3 and SPEI-6
        # Note: Set index to date for rolling, but here we passed Series
        # Re-indexing handled inside our custom spei function? 
        # No, let's pass Series with DatetimeIndex if needed, but our function uses month from index
        group_indexed = group.set_index("time")
        
        group["SPEI_3"] = calculate_spei(group_indexed["water_deficit"], scale=3).values
        group["SPEI_6"] = calculate_spei(group_indexed["water_deficit"], scale=6).values
        
        # SPEI-3 first-difference: direction of change (trend).
        # High autocorrelation (ρ≈0.97) means the model benefits from knowing
        # whether SPEI is rising or falling, not just its level.
        group["SPEI_3_diff"] = group["SPEI_3"].diff().fillna(0)
        
        processed_dfs.append(group)
        
    df_processed = pd.concat(processed_dfs, ignore_index=True)
    
    # 2. Temporal Features
    print("Engineering features...")
    df_processed["time_idx"] = (df_processed["time"] - df_processed["time"].min()).dt.days
    df_processed["month"] = df_processed["time"].dt.month
    
    # Cyclical Encoding for Month
    df_processed["month_sin"] = np.sin(2 * np.pi * df_processed["month"] / 12)
    df_processed["month_cos"] = np.cos(2 * np.pi * df_processed["month"] / 12)
    
    # Log transform precipitation (handling zeros)
    df_processed["precipitation_log"] = np.log1p(df_processed["precipitation_sum"])
    
    # 3. Clean NaN (caused by Rolling window or Failed Fit)
    # Replace infs first
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows where Targets are NaN
    # We drop the first few months to ensure SPEI-6 is valid
    df_clean = df_processed.dropna(subset=["SPEI_3", "SPEI_6", "water_deficit"]).reset_index(drop=True)
    
    # Final Sanity Check
    nan_counts = df_clean.isna().sum()
    if nan_counts.sum() > 0:
        print("WARNING: NaNs found after cleaning!")
        print(nan_counts[nan_counts > 0])
        # Force drop
        df_clean = df_clean.dropna().reset_index(drop=True)
        
    print(f"Data Cleaned. Rows: {len(df_processed)} -> {len(df_clean)}")

    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    print(f"Processed data saved to {output_path}. Shape: {df_clean.shape}")
    
    return df_clean

if __name__ == "__main__":
    preprocess_pipeline()
