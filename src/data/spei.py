import numpy as np
import pandas as pd
from scipy.stats import fisk, norm

def calculate_water_deficit(df):
    """
    D = P - PET (Water Deficit)
    Sesuai Vicente-Serrano et al. (2010)
    """
    return df["precipitation_sum"] - df["et0_fao_evapotranspiration"]


def calculate_spei(series, scale=3):
    """
    Calculates SPEI for a given time scale (e.g., 3 months).
    Uses fisk (Log-Logistic) distribution per Vicente-Serrano et al. (2010).
    
    NOTE: Input is daily data. Scale (months) is converted to ~30 days/month.
    This is a non-standard daily-resolution SPEI. Documented in methodology.
    
    Expected input: Series of Water Deficit (D) values with DatetimeIndex.
    """
    # 1. Rolling Sum of Deficit
    window_days = scale * 30
    d_accumulated = series.rolling(window=window_days, min_periods=window_days).sum()
    
    # Prepare output container
    spei_values = pd.Series(index=series.index, dtype=float)
    spei_values[:] = np.nan
    
    # 2. Fit Distribution (Per Calendar Month)
    months = series.index.month
    
    for month in range(1, 13):
        mask = (months == month)
        month_data = d_accumulated[mask]
        
        # Filter valid data for fitting
        valid_mask = month_data.notna() & (~np.isinf(month_data))
        valid_data = month_data[valid_mask]
        
        if len(valid_data) < 10:
            continue  # Not enough data for reliable fitting
            
        try:
            # Shift data to positive domain for Log-Logistic (fisk) fitting
            # fisk requires positive values; shift by (min - small_offset)
            shift = 0.0
            if valid_data.min() <= 0:
                shift = abs(valid_data.min()) + 1.0
            
            shifted_data = valid_data + shift
            shifted_month = month_data + shift
            
            # Fit Log-Logistic (fisk) distribution — SPEI standard
            params = fisk.fit(shifted_data, floc=0)
            
            # Calculate CDF on shifted data
            cdf = fisk.cdf(shifted_month, *params)
            
            # Clip CDF to avoid infinity in z-score transform
            cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
            z_score = norm.ppf(cdf)
            
            spei_values.loc[mask] = z_score
            
        except Exception:
            # Fallback: leave as NaN
            spei_values.loc[mask] = np.nan
            
    # Final cleanup
    spei_values.replace([np.inf, -np.inf], np.nan, inplace=True)
            
    return spei_values

def classify_spei(value):
    """
    Standardized SPEI classification (McKee et al., 1993 / WMO).
    Consistent thresholds used across all scripts.
    """
    if value <= -2.0: return "Kekeringan Ekstrem"
    elif value <= -1.5: return "Kekeringan Parah"
    elif value <= -1.0: return "Kekeringan Sedang"
    elif value < -0.5: return "Kekeringan Ringan"
    elif value <= 0.5: return "Normal"
    elif value < 1.0: return "Basah Ringan"
    elif value < 1.5: return "Basah Sedang"
    elif value < 2.0: return "Basah Parah"
    else: return "Basah Ekstrem"
