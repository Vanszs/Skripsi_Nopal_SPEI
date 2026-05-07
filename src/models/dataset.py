import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer
from sklearn.preprocessing import StandardScaler

MAX_ENCODER_LENGTH = 30
MAX_PREDICTION_LENGTH = 30
MODEL_GROUP_COL = "super_node_id"


class ArrayStandardScaler(StandardScaler):
    """
    StandardScaler variant that always fits/transforms on ndarray.

    This avoids sklearn's "valid feature names" warnings caused by fitting on
    DataFrame columns and transforming with ndarray batches.
    """

    @staticmethod
    def _to_2d_array(X):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def fit(self, X, y=None, sample_weight=None):
        return super().fit(self._to_2d_array(X), y=y, sample_weight=sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None):
        return super().partial_fit(self._to_2d_array(X), y=y, sample_weight=sample_weight)

    def transform(self, X, copy=None):
        return super().transform(self._to_2d_array(X), copy=copy)

    def inverse_transform(self, X, copy=None):
        return super().inverse_transform(self._to_2d_array(X), copy=copy)


def _validate_schema(data: pd.DataFrame):
    required = {
        "schema_version",
        "time_idx",
        "SPEI_3",
        "SPEI_6",
        "SPEI_3_diff",
        "water_deficit",
        "precipitation_log",
        "et0_fao_evapotranspiration",
        "soil_moisture",
        "temperature_2m_max",
        "temperature_2m_min",
        "month_sin",
        "month_cos",
        "city_id",
        "location_id",
        MODEL_GROUP_COL,
        "elevation",
        "lat",
        "lon",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Processed schema missing columns: {sorted(missing)}. "
            "Run preprocess_pipeline() with schema v2."
        )


def create_dataset(
    data: pd.DataFrame,
    max_encoder_length: int = MAX_ENCODER_LENGTH,
    max_prediction_length: int = MAX_PREDICTION_LENGTH,
):
    """
    Creates a TimeSeriesDataSet from processed schema-v2 data.

    Model grouping is based on super_node_id to guarantee leakage-safe sequence
    boundaries when raw nodes are expanded per city before aggregation.
    """
    data = data.replace([float("inf"), float("-inf")], float("nan")).dropna().copy()
    _validate_schema(data)
    print(f"Dataset Shape after dropna: {data.shape}")

    if data[MODEL_GROUP_COL].nunique() == 0:
        raise ValueError("No super_node_id found in data.")

    for col in ["city_id", "location_id", MODEL_GROUP_COL]:
        data[col] = data[col].astype(str)

    real_scalers = {
        "elevation": ArrayStandardScaler(),
        "lat": ArrayStandardScaler(),
        "lon": ArrayStandardScaler(),
        "month_sin": ArrayStandardScaler(),
        "month_cos": ArrayStandardScaler(),
        "SPEI_6": ArrayStandardScaler(),
        "SPEI_3_diff": ArrayStandardScaler(),
        "water_deficit": ArrayStandardScaler(),
        "precipitation_log": ArrayStandardScaler(),
        "et0_fao_evapotranspiration": ArrayStandardScaler(),
        "soil_moisture": ArrayStandardScaler(),
        "temperature_2m_max": ArrayStandardScaler(),
        "temperature_2m_min": ArrayStandardScaler(),
    }

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx < x.time_idx.max() - max_prediction_length],
        time_idx="time_idx",
        target="SPEI_3",
        group_ids=[MODEL_GROUP_COL],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[MODEL_GROUP_COL, "city_id"],
        static_reals=["elevation", "lat", "lon"],
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
            "temperature_2m_min",
        ],
        target_normalizer=EncoderNormalizer(transformation=None),
        scalers=real_scalers,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    return training
