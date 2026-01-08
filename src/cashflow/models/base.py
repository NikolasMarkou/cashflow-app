"""Abstract base class for forecast models - SDD Section 13."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ForecastOutput:
    """Output from a forecast model."""

    model_name: str
    forecast_mean: np.ndarray
    forecast_lower: np.ndarray
    forecast_upper: np.ndarray
    month_keys: list[str]
    wmape: Optional[float] = None
    params: dict = field(default_factory=dict)
    order: Optional[tuple] = None
    seasonal_order: Optional[tuple] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast to DataFrame."""
        return pd.DataFrame(
            {
                "month_key": self.month_keys,
                "forecast_mean": self.forecast_mean,
                "forecast_lower": self.forecast_lower,
                "forecast_upper": self.forecast_upper,
            }
        )


class ForecastModel(ABC):
    """Abstract base class for all forecast models.

    Per SDD Section 13.1: Prefer simple, auditable models unless
    complexity delivers material accuracy gains.
    """

    name: str = "BaseModel"
    complexity_score: int = 0  # Lower = simpler (used for tie-breaking)

    @abstractmethod
    def fit(self, series: pd.Series) -> "ForecastModel":
        """Fit the model to historical data.

        Args:
            series: Time series to fit (should be residual series)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(
        self,
        steps: int,
        confidence_level: float = 0.95,
    ) -> ForecastOutput:
        """Generate forecast for future periods.

        Args:
            steps: Number of periods to forecast
            confidence_level: Confidence level for intervals (default 95%)

        Returns:
            ForecastOutput with predictions and confidence intervals
        """
        pass

    def fit_predict(
        self,
        series: pd.Series,
        steps: int,
        confidence_level: float = 0.95,
    ) -> ForecastOutput:
        """Convenience method to fit and predict in one call."""
        self.fit(series)
        return self.predict(steps, confidence_level)

    @abstractmethod
    def get_params(self) -> dict:
        """Get fitted model parameters."""
        pass

    def evaluate(
        self,
        train_series: pd.Series,
        test_series: pd.Series,
    ) -> float:
        """Evaluate model using train/test split.

        Args:
            train_series: Training data
            test_series: Test data for evaluation

        Returns:
            WMAPE score
        """
        from cashflow.utils import calculate_wmape

        self.fit(train_series)
        forecast = self.predict(len(test_series))

        return calculate_wmape(test_series.values, forecast.forecast_mean)


def generate_future_month_keys(
    last_date: pd.Timestamp,
    steps: int,
) -> list[str]:
    """Generate month keys for forecast horizon.

    Args:
        last_date: Last date in training data
        steps: Number of months to forecast

    Returns:
        List of month keys in YYYY-MM format
    """
    from dateutil.relativedelta import relativedelta

    # Ensure we start from the first of the next month
    if hasattr(last_date, "to_timestamp"):
        last_date = last_date.to_timestamp()

    start_date = last_date + relativedelta(months=1)
    start_date = start_date.replace(day=1)

    month_keys = []
    for i in range(steps):
        forecast_date = start_date + relativedelta(months=i)
        month_keys.append(forecast_date.strftime("%Y-%m"))

    return month_keys
