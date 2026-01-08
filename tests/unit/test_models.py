"""Unit tests for forecasting models."""

import numpy as np
import pandas as pd
import pytest

from cashflow.models import ETSModel, SARIMAModel
from cashflow.models.selection import ModelSelector, select_best_model
from cashflow.utils import calculate_wmape


class TestWMAPE:
    """Tests for WMAPE calculation."""

    def test_perfect_prediction(self):
        """WMAPE should be 0 for perfect predictions."""
        actual = np.array([100, 200, 300])
        predicted = np.array([100, 200, 300])

        wmape = calculate_wmape(actual, predicted)

        assert wmape == 0.0

    def test_wmape_calculation(self):
        """WMAPE should be calculated correctly."""
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 310])

        # |100-110| + |200-190| + |300-310| = 10 + 10 + 10 = 30
        # |100| + |200| + |300| = 600
        # WMAPE = 30/600 * 100 = 5%
        wmape = calculate_wmape(actual, predicted)

        assert pytest.approx(wmape, rel=0.01) == 5.0

    def test_handles_zero_actual(self):
        """WMAPE should handle zero actual values."""
        actual = np.array([0, 0, 0])
        predicted = np.array([10, 20, 30])

        wmape = calculate_wmape(actual, predicted)

        assert wmape == 100.0  # Maximum error


class TestETSModel:
    """Tests for ETS model."""

    def test_fit_predict_basic(self, sample_time_series):
        """ETS model should fit and predict."""
        model = ETSModel(trend="add", seasonal=None)

        model.fit(sample_time_series)
        output = model.predict(steps=12)

        assert output.model_name == "ETS"
        assert len(output.forecast_mean) == 12
        assert len(output.forecast_lower) == 12
        assert len(output.forecast_upper) == 12
        assert len(output.month_keys) == 12

    def test_confidence_intervals_order(self, sample_time_series):
        """Lower CI should be less than upper CI."""
        model = ETSModel(trend="add", seasonal=None)

        model.fit(sample_time_series)
        output = model.predict(steps=12)

        for lower, upper in zip(output.forecast_lower, output.forecast_upper):
            assert lower < upper


class TestSARIMAModel:
    """Tests for SARIMA model."""

    def test_fit_predict_basic(self, sample_time_series):
        """SARIMA model should fit and predict."""
        model = SARIMAModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 12))

        model.fit(sample_time_series)
        output = model.predict(steps=12)

        assert output.model_name == "SARIMA"
        assert len(output.forecast_mean) == 12

    def test_params_include_aic(self, sample_time_series):
        """Fitted model should report AIC."""
        model = SARIMAModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 12))

        model.fit(sample_time_series)
        params = model.get_params()

        assert "aic" in params


class TestModelSelector:
    """Tests for model selection."""

    def test_selects_lower_wmape(self, sample_time_series):
        """Should select model with lower WMAPE."""
        train = sample_time_series[:-4]
        test = sample_time_series[-4:]

        selector = ModelSelector(wmape_threshold=20.0)

        # Add two mock results
        selector.results = [
            type("MockResult", (), {
                "model": type("MockModel", (), {"name": "Model_A", "complexity_score": 1})(),
                "wmape": 15.0,
                "forecast": None,
                "error": None,
            })(),
            type("MockResult", (), {
                "model": type("MockModel", (), {"name": "Model_B", "complexity_score": 2})(),
                "wmape": 10.0,
                "forecast": None,
                "error": None,
            })(),
        ]

        winner = selector.select_winner()

        assert winner.model.name == "Model_B"  # Lower WMAPE

    def test_tie_breaker_prefers_simpler(self):
        """Should prefer simpler model when WMAPE is similar."""
        selector = ModelSelector(wmape_threshold=20.0, tie_tolerance=1.0)

        # Models with similar WMAPE but different complexity
        selector.results = [
            type("MockResult", (), {
                "model": type("MockModel", (), {"name": "Complex", "complexity_score": 3})(),
                "wmape": 10.0,
                "forecast": None,
                "error": None,
            })(),
            type("MockResult", (), {
                "model": type("MockModel", (), {"name": "Simple", "complexity_score": 1})(),
                "wmape": 10.5,  # Slightly worse but within tolerance
                "forecast": None,
                "error": None,
            })(),
        ]

        winner = selector.select_winner()

        assert winner.model.name == "Simple"  # Simpler model wins tie

    def test_meets_threshold_property(self):
        """Should correctly report threshold compliance."""
        selector = ModelSelector(wmape_threshold=20.0)

        selector.results = [
            type("MockResult", (), {
                "model": type("MockModel", (), {"name": "Good", "complexity_score": 1})(),
                "wmape": 15.0,
                "forecast": None,
                "error": None,
            })(),
        ]

        selector.select_winner()

        assert selector.meets_threshold == True

        # Now with a bad model
        selector2 = ModelSelector(wmape_threshold=10.0)
        selector2.results = selector.results
        selector2.select_winner()

        assert selector2.meets_threshold == False
