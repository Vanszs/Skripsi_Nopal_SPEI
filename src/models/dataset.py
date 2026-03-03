from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import pandas as pd

MAX_ENCODER_LENGTH = 90   # 90 days history (matches SPEI-3 = 3×30 day window)
MAX_PREDICTION_LENGTH = 30  # 30 days forecast

def create_dataset(data: pd.DataFrame):
    """
    Creates a TimeSeriesDataSet from the processed dataframe.
    """
    # Ensure no NaN in critical columns
    data = data.replace([float('inf'), float('-inf')], float('nan')).dropna()
    print(f"Dataset Shape after dropna: {data.shape}")
    
    # Check for NaNs one last time
    if data.isna().any().any():
        print("CRITICAL WARNING: NaNs still present in data passed to TimeSeriesDataSet!")
        print(data.isna().sum())
    
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx < x.time_idx.max() - MAX_PREDICTION_LENGTH],
        time_idx="time_idx",
        target="SPEI_3",
        group_ids=["location_id"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        
        static_categoricals=["location_id"],
        static_reals=["elevation"],
        
        time_varying_known_reals=[
            "time_idx", 
            "month_sin", 
            "month_cos",
        ],
        
        time_varying_unknown_reals=[
            "SPEI_3",
            "SPEI_6",
            "water_deficit",
            "precipitation_log",
            "et0_fao_evapotranspiration",
            "soil_moisture",
            "temperature_2m_max",
            "temperature_2m_min"
        ],
        
        target_normalizer=GroupNormalizer(
            groups=["location_id"], transformation=None  # SPEI is already Z-score
        ),
            
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    
    return training
