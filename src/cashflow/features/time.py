"""Time-based feature engineering - SDD Section 12.2."""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


def add_time_features(df: pd.DataFrame, month_col: str = "month_key") -> pd.DataFrame:
    """Add time-based features to monthly data.

    Per SDD Section 12.2:
    - Month-of-year encoding
    - Trend indicators

    Args:
        df: Monthly DataFrame
        month_col: Name of month key column

    Returns:
        DataFrame with time features added
    """
    df = df.copy()

    # Parse month_key to extract components
    month_dt = pd.to_datetime(df[month_col])

    # Month of year (1-12)
    df["month_of_year"] = month_dt.dt.month

    # Quarter
    df["quarter"] = month_dt.dt.quarter

    # Year
    df["year"] = month_dt.dt.year

    # Month index (continuous for trend)
    df["month_index"] = range(len(df))

    # Seasonal encoding (sin/cos for cyclical)
    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12)

    # Is start/end of quarter
    df["is_quarter_start"] = df["month_of_year"].isin([1, 4, 7, 10])
    df["is_quarter_end"] = df["month_of_year"].isin([3, 6, 9, 12])

    # Is year end (December/January)
    df["is_year_boundary"] = df["month_of_year"].isin([1, 12])

    return df


def add_lag_features(
    df: pd.DataFrame,
    value_col: str,
    lags: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Add lagged features for time series modeling.

    Per SDD Section 12.2:
    - Lagged residuals (t-1, t-2, t-12)

    Args:
        df: DataFrame with time series data
        value_col: Column to create lags from
        lags: List of lag periods (default: [1, 2, 12])

    Returns:
        DataFrame with lag features added
    """
    if lags is None:
        lags = [1, 2, 12]

    df = df.copy()

    for lag in lags:
        df[f"{value_col}_lag{lag}"] = df[value_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Add rolling window features.

    Args:
        df: DataFrame with time series data
        value_col: Column to compute rolling stats from
        windows: List of window sizes (default: [3, 6, 12])

    Returns:
        DataFrame with rolling features added
    """
    if windows is None:
        windows = [3, 6, 12]

    df = df.copy()

    for window in windows:
        df[f"{value_col}_roll{window}_mean"] = (
            df[value_col].rolling(window=window, min_periods=1).mean()
        )
        df[f"{value_col}_roll{window}_std"] = (
            df[value_col].rolling(window=window, min_periods=1).std()
        )

    return df
