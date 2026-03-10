from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data.encoders import EncoderNormalizer, NaNLabelEncoder
import pandas as pd

MAX_ENCODER_LENGTH = 90   # 90 days history (matches SPEI-3 = 3×30 day window)
MAX_PREDICTION_LENGTH = 30  # 30 days forecast

def create_dataset(data: pd.DataFrame,
                   max_encoder_length: int = MAX_ENCODER_LENGTH,
                   max_prediction_length: int = MAX_PREDICTION_LENGTH):
    """
    Creates a TimeSeriesDataSet from the processed dataframe.

    Args:
        max_encoder_length: encoder context window in days.
            MUST be >= 90 so SPEI-3 (90-day rolling) is fully observable.
            Passed from checkpoint hparams so train/eval are always consistent.
        max_prediction_length: forecast horizon in days.

    Design notes:
        min_encoder_length = max_encoder_length   → no short-context samples;
            every sample sees exactly 'max_encoder_length' past days.
        min_prediction_length = max_prediction_length → full 30-day horizon
            on every sample; prevents the model learning on partial horizons.
        allow_missing_timesteps=True handles minor calendar gaps.
        EncoderNormalizer: normalises each sample using its own encoder
            window statistics (mean / std).  This makes the model predict
            *deviations from recent SPEI* instead of absolute levels,
            automatically adapting to distribution shifts between train
            (pre-2023) and test (2024+) periods.  Critical because SPEI-3
            means can shift by >1 σ across periods (e.g. Nganjuk drought).
    """
    # Ensure no NaN in critical columns
    data = data.replace([float('inf'), float('-inf')], float('nan')).dropna()
    print(f"Dataset Shape after dropna: {data.shape}")
    
    # Check for NaNs one last time
    if data.isna().any().any():
        print("CRITICAL WARNING: NaNs still present in data passed to TimeSeriesDataSet!")
        print(data.isna().sum())
    
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx < x.time_idx.max() - max_prediction_length],
        time_idx="time_idx",
        target="SPEI_3",
        group_ids=["location_id"],
        # Enforce full context: every sample must have exactly max_encoder_length
        # past days so the model always sees a complete SPEI-3 rolling window.
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        # Enforce full forecast window: every sample predicts exactly
        # max_prediction_length steps (no partial-horizon samples).
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        
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
            "SPEI_3_diff",
            "water_deficit",
            "precipitation_log",
            "et0_fao_evapotranspiration",
            "soil_moisture",
            "temperature_2m_max",
            "temperature_2m_min"
        ],
        
        target_normalizer=EncoderNormalizer(
            transformation=None,   # SPEI already Z-score, no Box-Cox needed
        ),
            
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    
    return training
