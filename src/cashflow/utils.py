"""Shared utilities - metrics, dates, validation."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from datetime import date
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass


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


# Phase 3.4: Enhanced confidence scoring system
@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence score components.

    Each component contributes 0-25 points to the total score (0-100).
    """
    data_quality_score: float  # 0-25
    history_length_score: float  # 0-25
    model_accuracy_score: float  # 0-25
    forecast_stability_score: float  # 0-25
    total_score: float  # 0-100
    level: str  # "High", "Medium", "Low"

    # Optional per-component confidence
    deterministic_confidence: Optional[float] = None  # 0-100
    residual_confidence: Optional[float] = None  # 0-100

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_score": round(self.total_score, 1),
            "level": self.level,
            "components": {
                "data_quality": round(self.data_quality_score, 1),
                "history_length": round(self.history_length_score, 1),
                "model_accuracy": round(self.model_accuracy_score, 1),
                "forecast_stability": round(self.forecast_stability_score, 1),
            },
            "per_component": {
                "deterministic": round(self.deterministic_confidence, 1) if self.deterministic_confidence else None,
                "residual": round(self.residual_confidence, 1) if self.residual_confidence else None,
            }
        }


def calculate_enhanced_confidence(
    data_quality_score: float,
    month_count: int,
    wmape: float,
    ci_width_ratio: Optional[float] = None,
    recurrence_coverage: Optional[float] = None,
    historical_volatility: Optional[float] = None,
) -> ConfidenceBreakdown:
    """Calculate enhanced numeric confidence score with breakdown.

    Phase 3.4: Production-grade confidence signaling.

    Args:
        data_quality_score: Quality score 0-100 (from data validation)
        month_count: Number of historical months
        wmape: Model WMAPE (%)
        ci_width_ratio: Optional ratio of CI width to forecast magnitude
        recurrence_coverage: Optional ratio of deterministic coverage (0-1)
        historical_volatility: Optional coefficient of variation of historical data

    Returns:
        ConfidenceBreakdown with detailed scoring
    """
    # Component 1: Data Quality (0-25 points)
    # Scale: 0% = 0 points, 100% = 25 points
    data_quality_pts = min(25.0, max(0.0, data_quality_score * 0.25))

    # Component 2: History Length (0-25 points)
    # Scale: 0 months = 0, 12 months = 10, 24 months = 18, 36+ months = 25
    if month_count >= 36:
        history_pts = 25.0
    elif month_count >= 24:
        history_pts = 18.0 + (month_count - 24) * (7.0 / 12.0)
    elif month_count >= 12:
        history_pts = 10.0 + (month_count - 12) * (8.0 / 12.0)
    else:
        history_pts = month_count * (10.0 / 12.0)

    # Component 3: Model Accuracy (0-25 points)
    # Scale: 0% WMAPE = 25, 20% WMAPE = 12.5, 40%+ WMAPE = 0
    if wmape <= 0:
        accuracy_pts = 25.0
    elif wmape >= 40:
        accuracy_pts = 0.0
    else:
        accuracy_pts = max(0.0, 25.0 - (wmape * 25.0 / 40.0))

    # Component 4: Forecast Stability (0-25 points)
    # Based on CI width ratio and historical volatility
    if ci_width_ratio is not None:
        # Lower CI width ratio = more stable = higher score
        # Ratio of 0 = 25 points, ratio of 1 = 0 points
        stability_from_ci = max(0.0, 25.0 - (ci_width_ratio * 25.0))
    else:
        stability_from_ci = 12.5  # Neutral if not provided

    if historical_volatility is not None:
        # Lower CV = more stable = higher score
        # CV of 0 = 25, CV of 1 = 0
        cv = min(1.0, historical_volatility)
        stability_from_vol = max(0.0, 25.0 - (cv * 25.0))
        stability_pts = (stability_from_ci + stability_from_vol) / 2
    else:
        stability_pts = stability_from_ci

    # Calculate total score
    total_score = data_quality_pts + history_pts + accuracy_pts + stability_pts

    # Determine level from total score
    if total_score >= 70:
        level = "High"
    elif total_score >= 40:
        level = "Medium"
    else:
        level = "Low"

    # Calculate per-component confidence if recurrence coverage provided
    deterministic_conf = None
    residual_conf = None

    if recurrence_coverage is not None:
        # Deterministic confidence: based on recurrence detection coverage
        # Higher coverage = higher confidence in deterministic component
        deterministic_conf = min(100.0, recurrence_coverage * 100.0)

        # Residual confidence: inverse - if more is deterministic, residual is smaller
        # Also factor in model accuracy
        residual_conf = max(0.0, accuracy_pts * 4.0)  # Scale to 0-100

    return ConfidenceBreakdown(
        data_quality_score=data_quality_pts,
        history_length_score=history_pts,
        model_accuracy_score=accuracy_pts,
        forecast_stability_score=stability_pts,
        total_score=total_score,
        level=level,
        deterministic_confidence=deterministic_conf,
        residual_confidence=residual_conf,
    )


def calculate_data_quality_score(df: pd.DataFrame, required_fields: list = None) -> float:
    """Calculate actual data quality score from DataFrame.

    Phase 3.4: Replace hardcoded 95% with actual calculation.

    Args:
        df: DataFrame to evaluate
        required_fields: List of required fields (default: standard UTF fields)

    Returns:
        Quality score 0-100
    """
    if required_fields is None:
        required_fields = ["tx_id", "customer_id", "account_id", "tx_date", "amount", "currency", "direction"]

    if len(df) == 0:
        return 0.0

    # Component 1: Required field completeness (50% of score)
    required_completeness = []
    for field in required_fields:
        if field in df.columns:
            completeness = 1.0 - df[field].isna().mean()
            required_completeness.append(completeness)
        else:
            required_completeness.append(0.0)

    required_score = (sum(required_completeness) / len(required_fields)) * 50

    # Component 2: Overall cell completeness (30% of score)
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    cell_completeness = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0
    completeness_score = cell_completeness * 30

    # Component 3: Data consistency (20% of score)
    consistency_score = 20.0  # Start with full points

    # Check for duplicate tx_ids
    if "tx_id" in df.columns:
        dup_rate = df["tx_id"].duplicated().mean()
        consistency_score -= dup_rate * 10

    # Check for valid amounts (not NaN, not zero for all)
    if "amount" in df.columns:
        invalid_amounts = df["amount"].isna().mean()
        zero_amounts = (df["amount"] == 0).mean()
        consistency_score -= invalid_amounts * 5
        if zero_amounts > 0.5:  # More than 50% zero is suspicious
            consistency_score -= 5

    consistency_score = max(0, consistency_score)

    return required_score + completeness_score + consistency_score


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
