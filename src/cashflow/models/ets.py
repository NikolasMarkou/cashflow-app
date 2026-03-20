"""Exponential Smoothing (ETS) model - SDD Section 13.2."""

from __future__ import annotations
from typing import Optional
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from cashflow.models.base import ForecastModel, ForecastOutput, generate_future_month_keys

logger = logging.getLogger(__name__)


class ETSModel(ForecastModel):
    """Exponential Smoothing (ETS) model.

    Per SDD Section 13.2, ETS is a robust baseline model for
    Layer 1 statistical forecasting.

    ETS models decompose time series into Error, Trend, and Seasonal
    components. Suitable for series with trend and/or seasonality.
    """

    name = "ETS"
    complexity_score = 1  # Simplest model

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: int = 12,
        damped_trend: bool = False,
    ):
        """Initialize ETS model.

        Args:
            trend: Trend component type ('add', 'mul', or None)
            seasonal: Seasonal component type ('add', 'mul', or None)
            seasonal_periods: Number of periods in a season (12 for monthly)
            damped_trend: Whether to dampen the trend
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self._fitted_model = None
        self._last_date = None

    def fit(self, series: pd.Series) -> "ETSModel":
        """Fit the ETS model.

        Args:
            series: Time series with PeriodIndex or DatetimeIndex

        Returns:
            Self for method chaining
        """
        # Prepare series
        series = series.dropna()

        if len(series) < self.seasonal_periods * 2:
            logger.warning(
                f"Series length ({len(series)}) may be too short for "
                f"seasonal ETS with period {self.seasonal_periods}"
            )
            # Fall back to non-seasonal
            self.seasonal = None

        try:
            model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods if self.seasonal else None,
                damped_trend=self.damped_trend if self.trend else False,
            )
            self._fitted_model = model.fit(optimized=True)
            self._last_date = series.index[-1]

            logger.info(f"ETS model fitted successfully")

        except Exception as e:
            logger.error(f"ETS fitting failed: {e}")
            raise

        return self

    def predict(
        self,
        steps: int,
        confidence_level: float = 0.95,
    ) -> ForecastOutput:
        """Generate forecast with confidence intervals.

        Args:
            steps: Number of periods to forecast
            confidence_level: Confidence level for intervals

        Returns:
            ForecastOutput with predictions
        """
        if self._fitted_model is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Generate forecast
        forecast = self._fitted_model.forecast(steps)

        # Estimate prediction intervals from residual standard error
        # Scale by sqrt(step) so intervals widen with forecast horizon
        residuals = self._fitted_model.resid
        std_error = residuals.std()

        from scipy import stats

        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha / 2)

        # Horizon-dependent intervals: uncertainty grows with sqrt(step)
        horizon_scale = np.array([np.sqrt(i + 1) for i in range(steps)])
        margin = z * std_error * horizon_scale
        lower = forecast - margin
        upper = forecast + margin

        # Generate month keys
        month_keys = generate_future_month_keys(self._last_date, steps)

        return ForecastOutput(
            model_name=self.name,
            forecast_mean=forecast.values,
            forecast_lower=lower.values,
            forecast_upper=upper.values,
            month_keys=month_keys,
            params=self.get_params(),
        )

    def get_params(self) -> dict:
        """Get fitted model parameters."""
        if self._fitted_model is None:
            return {}

        return {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods,
            "damped_trend": self.damped_trend,
            "smoothing_level": self._fitted_model.params.get("smoothing_level"),
            "smoothing_trend": self._fitted_model.params.get("smoothing_trend"),
            "smoothing_seasonal": self._fitted_model.params.get("smoothing_seasonal"),
            "aic": self._fitted_model.aic if hasattr(self._fitted_model, "aic") else None,
        }
