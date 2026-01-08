"""Shared utilities - metrics, dates, validation."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from datetime import date
from dateutil.relativedelta import relativedelta


def calculate_wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Weighted Mean Absolute Percentage Error.

    Per SDD Section 13.4:
    WMAPE = SUM(|y - y_hat|) / SUM(|y|)

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        WMAPE as percentage (0-100 scale)
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    denominator = np.abs(actual).sum()

    if denominator == 0:
        return 100.0  # Maximum error if no actual values

    numerator = np.abs(actual - predicted).sum()

    return (numerator / denominator) * 100


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAPE as percentage
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Avoid division by zero
    mask = actual != 0
    if not mask.any():
        return 100.0

    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        RMSE value
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return np.sqrt(np.mean((actual - predicted) ** 2))


def generate_month_range(
    start: str,
    end: str,
) -> list[str]:
    """Generate list of month keys between start and end.

    Args:
        start: Start month (YYYY-MM)
        end: End month (YYYY-MM)

    Returns:
        List of month keys in YYYY-MM format
    """
    start_date = pd.to_datetime(start + "-01")
    end_date = pd.to_datetime(end + "-01")

    dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    return [d.strftime("%Y-%m") for d in dates]


def get_forecast_period(
    last_historical_month: str,
    horizon: int = 12,
) -> tuple[str, str]:
    """Get the forecast period start and end.

    Args:
        last_historical_month: Last month with historical data (YYYY-MM)
        horizon: Number of months to forecast

    Returns:
        Tuple of (forecast_start, forecast_end) in YYYY-MM format
    """
    last_date = pd.to_datetime(last_historical_month + "-01")
    start_date = last_date + relativedelta(months=1)
    end_date = last_date + relativedelta(months=horizon)

    return start_date.strftime("%Y-%m"), end_date.strftime("%Y-%m")


def determine_confidence_level(
    data_quality_score: float,
    month_count: int,
    wmape: float,
) -> str:
    """Determine confidence level for forecast.

    Per SDD, confidence is affected by:
    - Data quality
    - Amount of historical data
    - Model accuracy

    Args:
        data_quality_score: Quality score 0-100
        month_count: Number of historical months
        wmape: Model WMAPE

    Returns:
        'High', 'Medium', or 'Low'
    """
    score = 0

    # Data quality contribution
    if data_quality_score >= 95:
        score += 2
    elif data_quality_score >= 80:
        score += 1

    # History length contribution
    if month_count >= 36:
        score += 2
    elif month_count >= 24:
        score += 1

    # Model accuracy contribution
    if wmape < 10:
        score += 2
    elif wmape < 20:
        score += 1

    # Map score to level
    if score >= 5:
        return "High"
    elif score >= 3:
        return "Medium"
    else:
        return "Low"


def split_train_test(
    series: pd.Series,
    test_size: int = 4,
) -> tuple[pd.Series, pd.Series]:
    """Split time series into train and test sets.

    Args:
        series: Time series to split
        test_size: Number of periods for test set

    Returns:
        Tuple of (train_series, test_series)
    """
    if len(series) <= test_size:
        raise ValueError(
            f"Series length ({len(series)}) must be greater than test_size ({test_size})"
        )

    train = series[:-test_size]
    test = series[-test_size:]

    return train, test
