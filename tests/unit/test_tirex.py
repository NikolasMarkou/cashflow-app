"""Unit tests for TiRex ONNX model."""

import numpy as np
import pandas as pd
import pytest

from cashflow.models.tirex import TiRexModel, DEFAULT_MODEL_PATH


class TestTiRexModel:
    """Tests for TiRex ONNX model."""

    @pytest.fixture
    def sample_time_series_24m(self) -> pd.Series:
        """Generate 24-month time series for TiRex testing."""
        np.random.seed(42)
        dates = pd.date_range("2024-01", "2025-12", freq="MS")

        # Realistic cashflow pattern: trend + seasonality + noise
        trend = np.linspace(400, 500, len(dates))
        seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 20, len(dates))

        values = trend + seasonal + noise

        series = pd.Series(values, index=dates.to_period("M"))
        series.name = "residual"
        return series

    @pytest.fixture
    def sample_time_series_36m(self) -> pd.Series:
        """Generate 36-month time series for longer history testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01", "2025-12", freq="MS")

        trend = np.linspace(350, 500, len(dates))
        seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 20, len(dates))

        values = trend + seasonal + noise

        series = pd.Series(values, index=dates.to_period("M"))
        series.name = "residual"
        return series

    def test_model_loads(self):
        """TiRex model should load ONNX file successfully."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        assert model._session is not None
        assert model.name == "TiRex"
        assert model.complexity_score == 4

    def test_model_not_found_raises_error(self, tmp_path):
        """Should raise FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError):
            TiRexModel(model_path=str(tmp_path / "nonexistent.onnx"))

    def test_fit_stores_data(self, sample_time_series_24m):
        """Fit should store last 24 months of data."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        model.fit(sample_time_series_24m)

        assert model._fitted is True
        assert model._input_data is not None
        assert model._input_data.shape == (1, 24, 1)

    def test_fit_with_longer_series(self, sample_time_series_36m):
        """Fit should use only last 24 months from longer series."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        model.fit(sample_time_series_36m)

        # Should still have shape [1, 24, 1] even with 36 months input
        assert model._input_data.shape == (1, 24, 1)

    def test_fit_short_series_pads_data(self):
        """Fit should pad short series with mean value to reach 24 months."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()

        # Create short series (12 months)
        dates = pd.date_range("2024-01", "2024-12", freq="MS")
        short_series = pd.Series(np.random.randn(12) + 100, index=dates.to_period("M"))

        # Should not raise - now pads with mean
        model.fit(short_series)
        assert model._fitted is True

        # Should be able to predict
        output = model.predict(steps=12)
        assert len(output.forecast_mean) == 12

    def test_predict_returns_12_months(self, sample_time_series_24m):
        """Predict should return 12-month forecast."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        model.fit(sample_time_series_24m)
        output = model.predict(steps=12)

        assert output.model_name == "TiRex"
        assert len(output.forecast_mean) == 12
        assert len(output.forecast_lower) == 12
        assert len(output.forecast_upper) == 12
        assert len(output.month_keys) == 12

    def test_predict_fewer_steps(self, sample_time_series_24m):
        """Predict should work with fewer than 12 steps."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        model.fit(sample_time_series_24m)
        output = model.predict(steps=6)

        assert len(output.forecast_mean) == 6
        assert len(output.month_keys) == 6

    def test_predict_too_many_steps_raises_error(self, sample_time_series_24m):
        """Predict should raise ValueError for more than 12 steps."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        model.fit(sample_time_series_24m)

        with pytest.raises(ValueError, match="max 12 months"):
            model.predict(steps=15)

    def test_predict_without_fit_raises_error(self):
        """Predict should raise RuntimeError if not fitted."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(steps=12)

    def test_confidence_intervals_bracket_mean(self, sample_time_series_24m):
        """Lower CI should be less than mean, mean less than upper CI."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        model.fit(sample_time_series_24m)
        output = model.predict(steps=12, confidence_level=0.95)

        for i in range(12):
            assert output.forecast_lower[i] < output.forecast_mean[i], \
                f"Lower CI >= mean at step {i}"
            assert output.forecast_mean[i] < output.forecast_upper[i], \
                f"Mean >= upper CI at step {i}"

    def test_wider_confidence_level(self, sample_time_series_24m):
        """Higher confidence level should produce wider intervals."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        model.fit(sample_time_series_24m)

        output_90 = model.predict(steps=12, confidence_level=0.90)
        output_99 = model.predict(steps=12, confidence_level=0.99)

        # 99% CI should be wider than 90% CI
        width_90 = output_90.forecast_upper - output_90.forecast_lower
        width_99 = output_99.forecast_upper - output_99.forecast_lower

        assert all(width_99 > width_90), "99% CI should be wider than 90% CI"

    def test_month_keys_format(self, sample_time_series_24m):
        """Month keys should be in YYYY-MM format starting after last input."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        model.fit(sample_time_series_24m)
        output = model.predict(steps=12)

        # Last month of input is 2025-12, first forecast should be 2026-01
        assert output.month_keys[0] == "2026-01"
        assert output.month_keys[11] == "2026-12"

        # All should match YYYY-MM format
        import re
        for key in output.month_keys:
            assert re.match(r"^\d{4}-\d{2}$", key), f"Invalid month key format: {key}"

    def test_get_params(self, sample_time_series_24m):
        """get_params should return model metadata."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        params = model.get_params()

        assert params["architecture"] == "TiRex (LSTM + Transformer + RevIN)"
        assert params["input_months"] == 24
        assert params["output_months"] == 12
        assert params["fitted"] is False

        model.fit(sample_time_series_24m)
        params_fitted = model.get_params()

        assert params_fitted["fitted"] is True

    def test_fit_predict_convenience(self, sample_time_series_24m):
        """fit_predict should work as convenience method."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        output = model.fit_predict(sample_time_series_24m, steps=12)

        assert output.model_name == "TiRex"
        assert len(output.forecast_mean) == 12

    def test_evaluate_method(self, sample_time_series_24m):
        """evaluate should return WMAPE score."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        # Need at least 28 months for train (24) + test (4)
        dates = pd.date_range("2023-09", "2025-12", freq="MS")  # 28 months
        np.random.seed(42)
        values = 500 + np.random.normal(0, 50, len(dates))
        full_series = pd.Series(values, index=dates.to_period("M"))

        train = full_series[:-4]  # 24 months
        test = full_series[-4:]   # 4 months

        model = TiRexModel()
        wmape = model.evaluate(train, test)

        assert isinstance(wmape, float)
        assert wmape >= 0  # WMAPE is always non-negative

    def test_output_scale_matches_input(self, sample_time_series_24m):
        """Output scale should be similar to input scale (internal normalization)."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        model = TiRexModel()
        model.fit(sample_time_series_24m)
        output = model.predict(steps=12)

        input_mean = sample_time_series_24m.mean()
        input_std = sample_time_series_24m.std()

        # Forecast should be within reasonable range of input
        # (within 5 standard deviations of input mean)
        output_mean = output.forecast_mean.mean()
        assert abs(output_mean - input_mean) < 5 * input_std, \
            f"Output mean {output_mean} too far from input mean {input_mean}"


class TestTiRexIntegration:
    """Integration tests for TiRex with model selector."""

    def test_tirex_in_model_selection(self, sample_time_series):
        """TiRex should work with ModelSelector."""
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("TiRex model file not available")

        from cashflow.models.selection import ModelSelector

        # Need 28+ months for model selection
        dates = pd.date_range("2023-09", "2025-12", freq="MS")
        np.random.seed(42)
        values = 500 + np.random.normal(0, 50, len(dates))
        series = pd.Series(values, index=dates.to_period("M"))

        train = series[:-4]
        test = series[-4:]

        selector = ModelSelector(wmape_threshold=50.0)

        model = TiRexModel()
        selector.evaluate_model(
            model=model,
            train_series=train,
            test_series=test,
            forecast_steps=12,
        )

        # Should have one result
        assert len(selector.results) == 1
        assert selector.results[0].model.name == "TiRex"
