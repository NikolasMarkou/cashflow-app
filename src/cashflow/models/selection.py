"""Model selection and comparison - SDD Section 13.5."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import logging

import numpy as np
import pandas as pd

from cashflow.models.base import ForecastModel, ForecastOutput
from cashflow.models.ets import ETSModel
from cashflow.models.sarima import SARIMAModel, SARIMAXModel
from cashflow.utils import calculate_wmape

logger = logging.getLogger(__name__)


@dataclass
class FallbackConfig:
    """Configuration for model fallback behavior.

    Phase 3.3: Graceful degradation when models fail or data is insufficient.
    """
    enable_fallback: bool = True
    fallback_chain: List[str] = field(default_factory=lambda: [
        "sarimax", "sarima", "ets", "naive"
    ])
    min_data_for_seasonal: int = 24  # Minimum months for seasonal models
    min_data_for_arima: int = 12  # Minimum months for ARIMA
    naive_window: int = 3  # Window for naive fallback (last N months average)


@dataclass
class ModelResult:
    """Result from evaluating a single model."""

    model: ForecastModel
    wmape: float
    forecast: Optional[ForecastOutput] = None
    error: Optional[str] = None
    is_fallback: bool = False  # True if this model was a fallback choice


class NaiveModel(ForecastModel):
    """Naive forecaster using rolling average.

    Phase 3.3: Last-resort fallback when all other models fail.
    Uses the average of the last N months as the forecast.
    """

    def __init__(self, window: int = 3):
        """Initialize naive model.

        Args:
            window: Number of recent periods to average (default: 3)
        """
        self.window = window
        self._last_values: Optional[np.ndarray] = None
        self._mean_value: Optional[float] = None
        self._std_value: Optional[float] = None
        self._fitted = False
        self._last_index: Optional[pd.Timestamp] = None

    @property
    def name(self) -> str:
        return f"Naive({self.window})"

    @property
    def complexity_score(self) -> int:
        return 0  # Simplest possible

    def fit(self, series: pd.Series, **kwargs) -> "NaiveModel":
        """Fit by storing the last N values."""
        values = series.values
        self._last_values = values[-self.window:] if len(values) >= self.window else values
        self._mean_value = float(np.mean(self._last_values))
        self._std_value = float(np.std(self._last_values)) if len(self._last_values) > 1 else abs(self._mean_value) * 0.1
        self._fitted = True

        # Store last index for month key generation
        if hasattr(series, 'index') and len(series.index) > 0:
            self._last_index = series.index[-1]
        return self

    def predict(self, steps: int, confidence_level: float = 0.95, **kwargs) -> ForecastOutput:
        """Predict by repeating the average of last N values."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        forecast_mean = np.full(steps, self._mean_value)

        # Confidence interval based on historical std and confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        lower = forecast_mean - z_score * self._std_value
        upper = forecast_mean + z_score * self._std_value

        # Generate month keys
        from cashflow.models.base import generate_future_month_keys
        try:
            if self._last_index is not None and hasattr(self._last_index, 'to_timestamp'):
                month_keys = generate_future_month_keys(self._last_index, steps)
            elif self._last_index is not None and isinstance(self._last_index, pd.Timestamp):
                month_keys = generate_future_month_keys(self._last_index, steps)
            else:
                month_keys = [f"T+{i+1}" for i in range(steps)]
        except (TypeError, AttributeError):
            month_keys = [f"T+{i+1}" for i in range(steps)]

        return ForecastOutput(
            forecast_mean=forecast_mean,
            forecast_lower=lower,
            forecast_upper=upper,
            model_name=self.name,
            month_keys=month_keys,
            params=self.get_params(),
        )

    def get_params(self) -> dict:
        """Get fitted model parameters."""
        return {
            "window": self.window,
            "mean_value": self._mean_value,
            "std_value": self._std_value,
            "fitted": self._fitted,
        }


class ModelSelector:
    """Model selection using WMAPE comparison.

    Per SDD Section 13.5:
    1. Lowest WMAPE wins
    2. Tie-breaker: simpler model (lower complexity_score)
    3. Override allowed for explainability if within tolerance (≤0.5pp)

    Phase 3.3 addition: Fallback behavior when models fail.
    """

    def __init__(
        self,
        wmape_threshold: float = 20.0,
        tie_tolerance: float = 0.5,
        fallback_config: Optional[FallbackConfig] = None,
    ):
        """Initialize model selector.

        Args:
            wmape_threshold: Maximum acceptable WMAPE (default 20% per SDD)
            tie_tolerance: WMAPE difference threshold for tie-breaking (0.5pp)
            fallback_config: Configuration for fallback behavior (default: enabled)
        """
        self.wmape_threshold = wmape_threshold
        self.tie_tolerance = tie_tolerance
        self.fallback_config = fallback_config or FallbackConfig()
        self.results: list[ModelResult] = []
        self._winner: Optional[ModelResult] = None
        self._fallback_used: bool = False
        self._fallback_reason: Optional[str] = None
        self._last_train_series: Optional[pd.Series] = None
        self._last_test_series: Optional[pd.Series] = None
        self._last_forecast_steps: int = 12

    def evaluate_model(
        self,
        model: ForecastModel,
        train_series: pd.Series,
        test_series: pd.Series,
        forecast_steps: int = 12,
        train_exog: Optional[pd.DataFrame] = None,
        test_exog: Optional[pd.DataFrame] = None,
        future_exog: Optional[pd.DataFrame] = None,
    ) -> ModelResult:
        """Evaluate a single model.

        Args:
            model: Model instance to evaluate
            train_series: Training data
            test_series: Test data for WMAPE calculation
            forecast_steps: Number of steps for final forecast
            train_exog: Optional exogenous variables for training (SARIMAX)
            test_exog: Optional exogenous variables for test period (SARIMAX)
            future_exog: Optional exogenous variables for forecast period (SARIMAX)

        Returns:
            ModelResult with evaluation metrics
        """
        # Store series for potential fallback use
        self._last_train_series = train_series
        self._last_test_series = test_series
        self._last_forecast_steps = forecast_steps

        try:
            # Check if model supports exogenous variables
            supports_exog = isinstance(model, SARIMAXModel)

            # Fit on training data
            if supports_exog and train_exog is not None:
                model.fit(train_series, exog=train_exog)
            else:
                model.fit(train_series)

            # Predict on test period
            if supports_exog and test_exog is not None:
                test_forecast = model.predict(len(test_series), exog_future=test_exog)
            else:
                test_forecast = model.predict(len(test_series))

            # Calculate WMAPE
            wmape = calculate_wmape(test_series.values, test_forecast.forecast_mean)

            # Generate full forecast
            # Refit on full data for production forecast
            full_series = pd.concat([train_series, test_series])

            if supports_exog and train_exog is not None and test_exog is not None:
                full_exog = pd.concat([train_exog, test_exog])
                model.fit(full_series, exog=full_exog)
                final_forecast = model.predict(forecast_steps, exog_future=future_exog)
            else:
                model.fit(full_series)
                final_forecast = model.predict(forecast_steps)

            final_forecast.wmape = wmape

            result = ModelResult(
                model=model,
                wmape=wmape,
                forecast=final_forecast,
            )

            logger.info(f"{model.name}: WMAPE = {wmape:.3f}%")

        except Exception as e:
            logger.error(f"{model.name} failed: {e}")
            result = ModelResult(
                model=model,
                wmape=float("inf"),
                error=str(e),
            )

        self.results.append(result)
        return result

    def select_winner(self) -> ModelResult:
        """Select the best model based on WMAPE and tie-breaking rules.

        If all models fail and fallback is enabled, returns a naive forecast.

        Returns:
            The winning ModelResult
        """
        if not self.results:
            raise ValueError("No models have been evaluated")

        # Filter out failed models
        valid_results = [r for r in self.results if r.error is None]

        if not valid_results:
            # All models failed - try fallback if enabled
            if self.fallback_config.enable_fallback:
                logger.warning("All models failed, attempting fallback to naive forecast")
                self._fallback_used = True
                self._fallback_reason = "all_models_failed"
                return self._create_fallback_result()
            else:
                raise ValueError("All models failed during evaluation")

        # Sort by WMAPE (ascending)
        sorted_results = sorted(valid_results, key=lambda r: r.wmape)

        # Check for ties within tolerance
        best = sorted_results[0]

        for result in sorted_results[1:]:
            wmape_diff = result.wmape - best.wmape

            if wmape_diff <= self.tie_tolerance:
                # Tie-breaker: prefer simpler model
                if result.model.complexity_score < best.model.complexity_score:
                    logger.info(
                        f"Tie-break: {result.model.name} (simpler) selected over "
                        f"{best.model.name} (WMAPE diff: {wmape_diff:.3f}pp)"
                    )
                    best = result

        self._winner = best

        # Log selection
        logger.info(
            f"Selected model: {best.model.name} with WMAPE = {best.wmape:.3f}%"
        )

        if best.wmape > self.wmape_threshold:
            logger.warning(
                f"Warning: Winner WMAPE ({best.wmape:.3f}%) exceeds "
                f"threshold ({self.wmape_threshold}%)"
            )

        return best

    def _create_fallback_result(self) -> ModelResult:
        """Create a fallback result using naive forecasting.

        This is called when all other models have failed.
        Uses stored train/test series from the last evaluate_model call.

        Returns:
            ModelResult with naive forecast
        """
        if self._last_train_series is None or self._last_test_series is None:
            raise ValueError("Cannot create fallback without any evaluation attempts")

        return self._evaluate_naive_fallback(
            self._last_train_series,
            self._last_test_series,
            self._last_forecast_steps,
        )

    def evaluate_with_fallback(
        self,
        train_series: pd.Series,
        test_series: pd.Series,
        models: List[ForecastModel],
        forecast_steps: int = 12,
        train_exog: Optional[pd.DataFrame] = None,
        test_exog: Optional[pd.DataFrame] = None,
        future_exog: Optional[pd.DataFrame] = None,
    ) -> ModelResult:
        """Evaluate models with fallback support.

        This method evaluates all models and automatically falls back
        to simpler models if complex ones fail due to insufficient data.

        Args:
            train_series: Training data
            test_series: Test data
            models: List of models to try
            forecast_steps: Forecast horizon
            train_exog: Optional exogenous training data
            test_exog: Optional exogenous test data
            future_exog: Optional exogenous future data

        Returns:
            Best ModelResult (possibly a fallback)
        """
        data_length = len(train_series)
        cfg = self.fallback_config

        # Filter models based on data requirements
        eligible_models = []
        for model in models:
            model_type = model.name.lower()

            if "sarimax" in model_type or "sarima" in model_type:
                if data_length < cfg.min_data_for_seasonal:
                    logger.info(
                        f"Skipping {model.name}: requires {cfg.min_data_for_seasonal} "
                        f"months, have {data_length}"
                    )
                    continue
            elif "ets" in model_type:
                if data_length < cfg.min_data_for_arima:
                    logger.info(
                        f"Skipping {model.name}: requires {cfg.min_data_for_arima} "
                        f"months, have {data_length}"
                    )
                    continue

            eligible_models.append(model)

        # If no eligible models, use naive
        if not eligible_models:
            logger.warning(
                f"No models eligible for {data_length} months of data, "
                f"using naive fallback"
            )
            self._fallback_used = True
            self._fallback_reason = "insufficient_data"
            return self._evaluate_naive_fallback(
                train_series, test_series, forecast_steps
            )

        # Evaluate eligible models
        for model in eligible_models:
            self.evaluate_model(
                model=model,
                train_series=train_series,
                test_series=test_series,
                forecast_steps=forecast_steps,
                train_exog=train_exog,
                test_exog=test_exog,
                future_exog=future_exog,
            )

        # Select winner (includes fallback logic)
        return self.select_winner()

    def _evaluate_naive_fallback(
        self,
        train_series: pd.Series,
        test_series: pd.Series,
        forecast_steps: int,
    ) -> ModelResult:
        """Evaluate the naive fallback model.

        Args:
            train_series: Training data
            test_series: Test data
            forecast_steps: Forecast horizon

        Returns:
            ModelResult with naive forecast
        """
        naive_model = NaiveModel(window=self.fallback_config.naive_window)

        try:
            # Fit on training data
            naive_model.fit(train_series)

            # Predict for test period
            test_forecast = naive_model.predict(len(test_series))

            # Calculate WMAPE
            wmape = calculate_wmape(test_series.values, test_forecast.forecast_mean)

            # Refit on full data
            full_series = pd.concat([train_series, test_series])
            naive_model.fit(full_series)
            final_forecast = naive_model.predict(forecast_steps)
            final_forecast.wmape = wmape

            result = ModelResult(
                model=naive_model,
                wmape=wmape,
                forecast=final_forecast,
                is_fallback=True,
            )

            logger.info(f"{naive_model.name} (fallback): WMAPE = {wmape:.3f}%")

        except Exception as e:
            logger.error(f"Naive fallback failed: {e}")
            result = ModelResult(
                model=naive_model,
                wmape=float("inf"),
                error=str(e),
                is_fallback=True,
            )

        self.results.append(result)
        self._winner = result
        return result

    @property
    def winner(self) -> Optional[ModelResult]:
        """Get the winning model result."""
        return self._winner

    @property
    def meets_threshold(self) -> bool:
        """Check if winner meets WMAPE threshold."""
        if self._winner is None:
            return False
        return self._winner.wmape <= self.wmape_threshold

    def get_summary(self) -> dict:
        """Get summary of model comparison."""
        return {
            "models_evaluated": len(self.results),
            "models_failed": sum(1 for r in self.results if r.error is not None),
            "winner": self._winner.model.name if self._winner else None,
            "winner_wmape": self._winner.wmape if self._winner else None,
            "meets_threshold": self.meets_threshold,
            "threshold": self.wmape_threshold,
            "fallback_used": self._fallback_used,
            "fallback_reason": self._fallback_reason,
            "is_fallback": self._winner.is_fallback if self._winner else False,
            "all_results": [
                {
                    "model": r.model.name,
                    "wmape": r.wmape,
                    "error": r.error,
                    "is_fallback": r.is_fallback,
                }
                for r in self.results
            ],
        }


def select_best_model(
    train_series: pd.Series,
    test_series: pd.Series,
    models: Optional[list[ForecastModel]] = None,
    forecast_steps: int = 12,
    wmape_threshold: float = 20.0,
) -> tuple[ForecastOutput, dict]:
    """Convenience function to select the best model.

    Args:
        train_series: Training data
        test_series: Test data
        models: List of models to evaluate (default: ETS, SARIMA, SARIMAX)
        forecast_steps: Number of forecast steps
        wmape_threshold: WMAPE threshold

    Returns:
        Tuple of (winning forecast, summary dict)
    """
    if models is None:
        models = [
            ETSModel(trend="add", seasonal="add"),
            SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)),
        ]

    selector = ModelSelector(wmape_threshold=wmape_threshold)

    for model in models:
        selector.evaluate_model(
            model=model,
            train_series=train_series,
            test_series=test_series,
            forecast_steps=forecast_steps,
        )

    winner = selector.select_winner()
    summary = selector.get_summary()

    return winner.forecast, summary
