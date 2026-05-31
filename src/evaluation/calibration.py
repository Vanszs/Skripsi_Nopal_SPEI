"""
Per-city interval calibration (conformal-style).
Fits multiplicative widening factors on VALIDATION data only (year==2023).
"""
import numpy as np
import pandas as pd


def fit_per_city_interval_calibration(
    df_val: pd.DataFrame,
    city_col: str = "city_id",
    nominal: float = 0.80,
) -> dict:
    """
    Given validation predictions with columns [city_col, actual, pred_p10, pred_p50, pred_p90],
    compute per-city multiplicative factor so that scaled interval achieves ~nominal PICP.

    Returns dict {city_id: factor}.
    """
    factors = {}
    for city, grp in df_val.groupby(city_col):
        half_width = (grp["pred_p90"] - grp["pred_p10"]) / 2.0
        center = grp["pred_p50"]
        # Normalized residual: |actual - center| / half_width
        mask = half_width > 1e-9
        if mask.sum() < 5:
            factors[str(city)] = 1.0
            continue
        abs_norm_resid = np.abs(grp["actual"].values[mask] - center.values[mask]) / half_width.values[mask]
        # Factor = quantile of abs_norm_resid at nominal level
        factor = float(np.quantile(abs_norm_resid, nominal))
        factors[str(city)] = max(factor, 0.5)  # floor 0.5: avoid extreme interval shrinkage (C3)
    return factors


def apply_calibration(df: pd.DataFrame, factors: dict, city_col: str = "city_id") -> pd.DataFrame:
    """
    Apply calibration factors: scale half-width by factor per city.
    Returns df with calibrated pred_p10 / pred_p90 (pred_p50 unchanged).
    """
    df = df.copy()
    for city, factor in factors.items():
        mask = df[city_col].astype(str) == str(city)
        half_width = (df.loc[mask, "pred_p90"] - df.loc[mask, "pred_p10"]) / 2.0
        center = df.loc[mask, "pred_p50"]
        df.loc[mask, "pred_p10"] = center - half_width * factor
        df.loc[mask, "pred_p90"] = center + half_width * factor
    return df


def compute_picp(df: pd.DataFrame, city_col: str = "city_id") -> tuple[float, dict]:
    """Compute overall and per-city PICP."""
    in_interval = (df["actual"] >= df["pred_p10"]) & (df["actual"] <= df["pred_p90"])
    overall = float(in_interval.mean())
    per_city = {}
    for city in sorted(df[city_col].astype(str).unique()):
        mask = df[city_col].astype(str) == city
        per_city[city] = float(in_interval[mask].mean())
    return overall, per_city
