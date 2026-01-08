"""SARIMA and SARIMAX models - SDD Section 13.2."""

from __future__ import annotations
from typing import Optional
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from cashflow.models.base import ForecastModel, ForecastOutput, generate_future_month_keys

logger = logging.getLogger(__name__)


class SARIMAModel(ForecastModel):
    """Seasonal ARIMA (SARIMA) model.

    Per SDD Section 13.2, SARIMA captures seasonal behavioral patterns
    in the residual series.

    SARIMA(p,d,q)(P,D,Q,s) where:
    - (p,d,q): Non-seasonal ARIMA order
    - (P,D,Q,s): Seasonal order with period s
    """

    name = "SARIMA"
    complexity_score = 2

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 0, 12),
    ):
        """Initialize SARIMA model.

        Args:
            order: (p, d, q) non-seasonal order
            seasonal_order: (P, D, Q, s) seasonal order
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted_model = None
        self._last_date = None

    def fit(self, series: pd.Series) -> "SARIMAModel":
        """Fit the SARIMA model.

        Args:
            series: Time series with PeriodIndex

        Returns:
            Self for method chaining
        """
        series = series.dropna()

        # Convert to PeriodIndex if needed for statsmodels
        if not isinstance(series.index, pd.PeriodIndex):
            if isinstance(series.index, pd.DatetimeIndex):
                series.index = series.index.to_period("M")
            else:
                # Assume index is already in correct format or create one
                pass

        try:
            model = ARIMA(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
            )
            self._fitted_model = model.fit()
            self._last_date = series.index[-1]

            logger.info(
                f"SARIMA{self.order}x{self.seasonal_order} fitted successfully, "
                f"AIC={self._fitted_model.aic:.2f}"
            )

        except Exception as e:
            logger.error(f"SARIMA fitting failed: {e}")
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

        alpha = 1 - confidence_level

        # Get forecast with confidence intervals
        forecast_result = self._fitted_model.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)

        # Generate month keys
        month_keys = generate_future_month_keys(self._last_date, steps)

        return ForecastOutput(
            model_name=self.name,
            forecast_mean=forecast_mean.values,
            forecast_lower=conf_int.iloc[:, 0].values,
            forecast_upper=conf_int.iloc[:, 1].values,
            month_keys=month_keys,
            params=self.get_params(),
            order=self.order,
            seasonal_order=self.seasonal_order,
        )

    def get_params(self) -> dict:
        """Get fitted model parameters."""
        if self._fitted_model is None:
            return {}

        params = self._fitted_model.params.to_dict()
        params["aic"] = self._fitted_model.aic
        params["bic"] = self._fitted_model.bic

        return params


class SARIMAXModel(ForecastModel):
    """Seasonal ARIMA with eXogenous regressors (SARIMAX).

    Per SDD Section 13.2, SARIMAX incorporates known future events
    via exogenous variables (KnownFutureFlow_Delta).

    Per SDD Section 12.3.2, the exogenous vector is used both as:
    1. Model input during training
    2. Explicitly added during forecast recomposition
    """

    name = "SARIMAX"
    complexity_score = 3

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 0, 12),
    ):
        """Initialize SARIMAX model.

        Args:
            order: (p, d, q) non-seasonal order
            seasonal_order: (P, D, Q, s) seasonal order
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted_model = None
        self._last_date = None
        self._exog_cols = None

    def fit(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None,
    ) -> "SARIMAXModel":
        """Fit the SARIMAX model.

        Args:
            series: Time series with PeriodIndex
            exog: Optional exogenous variables DataFrame

        Returns:
            Self for method chaining
        """
        series = series.dropna()

        # Convert to PeriodIndex if needed
        if not isinstance(series.index, pd.PeriodIndex):
            if isinstance(series.index, pd.DatetimeIndex):
                series.index = series.index.to_period("M")

        if exog is not None:
            self._exog_cols = exog.columns.tolist()
            # Ensure exog index matches series
            exog = exog.loc[series.index]

        try:
            model = ARIMA(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                exog=exog,
            )
            self._fitted_model = model.fit()
            self._last_date = series.index[-1]

            logger.info(
                f"SARIMAX{self.order}x{self.seasonal_order} fitted with "
                f"{len(self._exog_cols or [])} exogenous variables, "
                f"AIC={self._fitted_model.aic:.2f}"
            )

        except Exception as e:
            logger.error(f"SARIMAX fitting failed: {e}")
            raise

        return self

    def predict(
        self,
        steps: int,
        confidence_level: float = 0.95,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        """Generate forecast with confidence intervals.

        Args:
            steps: Number of periods to forecast
            confidence_level: Confidence level for intervals
            exog_future: Exogenous variables for forecast period

        Returns:
            ForecastOutput with predictions
        """
        if self._fitted_model is None:
            raise RuntimeError("Model must be fitted before prediction")

        alpha = 1 - confidence_level

        # Get forecast with confidence intervals
        forecast_result = self._fitted_model.get_forecast(steps=steps, exog=exog_future)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)

        # Generate month keys
        month_keys = generate_future_month_keys(self._last_date, steps)

        return ForecastOutput(
            model_name=self.name,
            forecast_mean=forecast_mean.values,
            forecast_lower=conf_int.iloc[:, 0].values,
            forecast_upper=conf_int.iloc[:, 1].values,
            month_keys=month_keys,
            params=self.get_params(),
            order=self.order,
            seasonal_order=self.seasonal_order,
        )

    def get_params(self) -> dict:
        """Get fitted model parameters."""
        if self._fitted_model is None:
            return {}

        params = self._fitted_model.params.to_dict()
        params["aic"] = self._fitted_model.aic
        params["bic"] = self._fitted_model.bic
        params["exog_cols"] = self._exog_cols

        return params
