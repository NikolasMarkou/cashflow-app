"""Model selection and comparison - SDD Section 13.5."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import logging

import pandas as pd

from cashflow.models.base import ForecastModel, ForecastOutput
from cashflow.models.ets import ETSModel
from cashflow.models.sarima import SARIMAModel, SARIMAXModel
from cashflow.utils import calculate_wmape

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result from evaluating a single model."""

    model: ForecastModel
    wmape: float
    forecast: Optional[ForecastOutput] = None
    error: Optional[str] = None


class ModelSelector:
    """Model selection using WMAPE comparison.

    Per SDD Section 13.5:
    1. Lowest WMAPE wins
    2. Tie-breaker: simpler model (lower complexity_score)
    3. Override allowed for explainability if within tolerance (≤0.5pp)
    """

    def __init__(
        self,
        wmape_threshold: float = 20.0,
        tie_tolerance: float = 0.5,
    ):
        """Initialize model selector.

        Args:
            wmape_threshold: Maximum acceptable WMAPE (default 20% per SDD)
            tie_tolerance: WMAPE difference threshold for tie-breaking (0.5pp)
        """
        self.wmape_threshold = wmape_threshold
        self.tie_tolerance = tie_tolerance
        self.results: list[ModelResult] = []
        self._winner: Optional[ModelResult] = None

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
        try:
            # Check if model supports exogenous variables
            supports_exog = hasattr(model, 'fit') and 'exog' in model.fit.__code__.co_varnames

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

        Returns:
            The winning ModelResult
        """
        if not self.results:
            raise ValueError("No models have been evaluated")

        # Filter out failed models
        valid_results = [r for r in self.results if r.error is None]

        if not valid_results:
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
            "all_results": [
                {
                    "model": r.model.name,
                    "wmape": r.wmape,
                    "error": r.error,
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
