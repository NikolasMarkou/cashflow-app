"""Outlier treatment strategies - SDD Section 11.4."""

from __future__ import annotations
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from loguru import logger


class TreatmentMethod(str, Enum):
    """Supported outlier treatment methods."""

    MEDIAN = "median"
    ROLLING_MEDIAN = "rolling_median"
    CAPPED = "capped"
    WINSORIZE = "winsorize"


def treat_outliers(
    df: pd.DataFrame,
    value_col: str,
    is_outlier_col: str = "is_outlier",
    method: str = "median",
    **kwargs,
) -> pd.DataFrame:
    """Apply treatment to detected outliers.

    Per SDD Section 11.4:
    - Flag outliers
    - Replace with rolling median or capped value
    - Preserve original values for audit traceability (dual-value model)

    Args:
        df: DataFrame containing the series and outlier flags
        value_col: Name of column containing values to treat
        is_outlier_col: Name of boolean column indicating outliers
        method: Treatment method (median, rolling_median, capped, winsorize)
        **kwargs: Method-specific parameters

    Returns:
        DataFrame with treated values and audit columns added
    """
    df = df.copy()

    if is_outlier_col not in df.columns:
        raise ValueError(f"Outlier column '{is_outlier_col}' not found")

    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found")

    # Preserve original values for audit
    original_col = f"{value_col}_original"
    clean_col = f"{value_col}_clean"

    df[original_col] = df[value_col]

    # Get treatment values based on method
    method = method.lower()

    if method == TreatmentMethod.MEDIAN.value:
        replacement = _median_treatment(df, value_col, is_outlier_col)
    elif method == TreatmentMethod.ROLLING_MEDIAN.value:
        replacement = _rolling_median_treatment(df, value_col, is_outlier_col, **kwargs)
    elif method == TreatmentMethod.CAPPED.value:
        replacement = _capped_treatment(df, value_col, is_outlier_col, **kwargs)
    elif method == TreatmentMethod.WINSORIZE.value:
        replacement = _winsorize_treatment(df, value_col, is_outlier_col, **kwargs)
    else:
        raise ValueError(f"Unknown treatment method: {method}")

    # Apply treatment only to outliers
    df[clean_col] = df[value_col].copy()
    outlier_mask = df[is_outlier_col]
    df.loc[outlier_mask, clean_col] = replacement[outlier_mask]

    # Add treatment tag
    df["treatment_tag"] = np.where(
        df[is_outlier_col],
        "ABNORMAL_EXTERNAL_FLOW",
        "NORMAL",
    )

    treated_count = outlier_mask.sum()
    if treated_count > 0:
        logger.info(
            f"Outlier treatment ({method}): treated {treated_count} values, "
            f"original preserved in '{original_col}'"
        )

    return df


def _median_treatment(
    df: pd.DataFrame,
    value_col: str,
    is_outlier_col: str,
) -> pd.Series:
    """Replace outliers with the overall median of non-outliers.

    This is the default treatment per SDD compliance.md.
    """
    non_outlier_values = df.loc[~df[is_outlier_col], value_col]
    median_value = non_outlier_values.median()

    # Return a series of median values with same index as df
    return pd.Series(median_value, index=df.index)


def _rolling_median_treatment(
    df: pd.DataFrame,
    value_col: str,
    is_outlier_col: str,
    window: int = 5,
) -> pd.Series:
    """Replace outliers with rolling median of surrounding non-outliers.

    More sophisticated than simple median as it preserves local trends.
    """
    values = df[value_col].copy()

    # Mask outliers temporarily
    masked = values.copy()
    masked[df[is_outlier_col]] = np.nan

    # Compute rolling median, interpolating over masked values
    # Use center=False to prevent data leakage (no future data in calculation)
    rolling = masked.rolling(window=window, center=False, min_periods=1).median()

    # Fill any remaining NaN with overall median
    overall_median = masked.median()
    rolling = rolling.fillna(overall_median)

    return rolling


def _capped_treatment(
    df: pd.DataFrame,
    value_col: str,
    is_outlier_col: str,
    lower_percentile: float = 5,
    upper_percentile: float = 95,
) -> pd.Series:
    """Cap outliers at percentile bounds.

    Values below lower_percentile are set to that percentile value.
    Values above upper_percentile are set to that percentile value.
    """
    values = df[value_col]
    non_outlier_values = values[~df[is_outlier_col]]

    lower_bound = non_outlier_values.quantile(lower_percentile / 100)
    upper_bound = non_outlier_values.quantile(upper_percentile / 100)

    capped = values.clip(lower=lower_bound, upper=upper_bound)

    return capped


def _winsorize_treatment(
    df: pd.DataFrame,
    value_col: str,
    is_outlier_col: str,
    limits: tuple[float, float] = (0.05, 0.05),
) -> pd.Series:
    """Winsorize outliers - replace extreme values with percentile values.

    Similar to capping but applied symmetrically from both tails.
    """
    from scipy.stats import mstats

    values = df[value_col].values
    winsorized = mstats.winsorize(values, limits=limits)

    return pd.Series(winsorized, index=df.index)


def apply_residual_treatment(
    decomposed_df: pd.DataFrame,
    detection_method: str = "modified_zscore",
    detection_threshold: float = 3.5,
    treatment_method: str = "median",
) -> pd.DataFrame:
    """Full outlier detection and treatment pipeline for residual series.

    This is the main entry point for outlier handling in the forecast pipeline.
    Applies detection and treatment to the 'residual' column.

    Args:
        decomposed_df: DataFrame with 'residual' column from decomposition
        detection_method: Method for detecting outliers
        detection_threshold: Threshold for detection
        treatment_method: Method for treating outliers

    Returns:
        DataFrame with outlier detection and treatment applied
    """
    from cashflow.outliers.detector import detect_outliers

    df = decomposed_df.copy()

    if "residual" not in df.columns:
        logger.warning("No 'residual' column found, skipping outlier treatment")
        return df

    # Detect outliers
    is_outlier, scores = detect_outliers(
        df["residual"],
        method=detection_method,
        threshold=detection_threshold,
    )

    df["is_outlier"] = is_outlier
    df["outlier_score"] = scores

    # Apply treatment
    df = treat_outliers(
        df,
        value_col="residual",
        is_outlier_col="is_outlier",
        method=treatment_method,
    )

    return df
