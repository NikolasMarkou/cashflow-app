"""Regression tests for bugs found by epistemic deconstructor analysis.

BUG-1: Fallback produces None forecast → crash
BUG-2: ETS seasonal state mutation persists across refitting
BUG-3: Web route double pipeline execution (structural — tested by code inspection)
BUG-4: SARIMAX silently ignored when requested
BUG-5: Data quality score uses wrong DataFrame type
BUG-6: Inconsistent customer_id between run() and run_from_dataframe()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestBUG1_FallbackProducesValidForecast:
    """BUG-1: When all models fail, fallback must produce a valid ForecastOutput."""

    def test_fallback_returns_valid_forecast(self, sample_time_series):
        """If all models fail, NaiveModel fallback should return a real forecast."""
        from cashflow.models.selection import ModelSelector, NaiveModel, ModelResult
        from cashflow.models.base import ForecastModel, ForecastOutput
        from cashflow.utils import split_train_test

        train, test = split_train_test(sample_time_series, test_size=4)

        selector = ModelSelector(wmape_threshold=20.0)

        # Simulate a model that always fails
        class AlwaysFailModel(ForecastModel):
            name = "AlwaysFail"
            complexity_score = 0

            def fit(self, series, **kwargs):
                raise ValueError("Intentional failure")

            def predict(self, steps, confidence_level=0.95, **kwargs):
                raise ValueError("Intentional failure")

            def get_params(self):
                return {}

        selector.evaluate_model(
            model=AlwaysFailModel(),
            train_series=train,
            test_series=test,
            forecast_steps=12,
        )

        winner = selector.select_winner()

        # The winner should have a valid forecast (not None)
        assert winner.forecast is not None
        assert isinstance(winner.forecast, ForecastOutput)
        assert len(winner.forecast.forecast_mean) == 12
        assert len(winner.forecast.month_keys) == 12
        assert winner.is_fallback is True

    def test_fallback_forecast_has_reasonable_values(self, sample_time_series):
        """Fallback forecast values should be based on recent data."""
        from cashflow.models.selection import ModelSelector
        from cashflow.models.base import ForecastModel
        from cashflow.utils import split_train_test

        train, test = split_train_test(sample_time_series, test_size=4)

        class AlwaysFailModel(ForecastModel):
            name = "AlwaysFail"
            complexity_score = 0
            def fit(self, series, **kwargs): raise ValueError("fail")
            def predict(self, steps, confidence_level=0.95, **kwargs): raise ValueError("fail")
            def get_params(self): return {}

        selector = ModelSelector()
        selector.evaluate_model(model=AlwaysFailModel(), train_series=train, test_series=test)
        winner = selector.select_winner()

        # Forecast mean should be close to the mean of recent values
        recent_mean = train.values[-3:].mean()
        assert abs(winner.forecast.forecast_mean[0] - recent_mean) < recent_mean * 0.5


class TestBUG2_ETSSeasonalNotMutated:
    """BUG-2: ETS seasonal config must be restored when refitting on longer data."""

    def test_seasonal_restored_after_short_series(self):
        """After fitting on short data (seasonal=None), refitting on long data restores seasonal."""
        from cashflow.models.ets import ETSModel

        model = ETSModel(trend="add", seasonal="add", seasonal_periods=12)

        # First fit: short series → seasonal disabled
        short_series = pd.Series(
            np.random.normal(100, 10, 10),
            index=pd.date_range("2024-01", periods=10, freq="MS").to_period("M"),
        )
        model.fit(short_series)
        assert model.seasonal is None  # Should be disabled for short data

        # Second fit: long series → seasonal should be restored
        long_series = pd.Series(
            np.random.normal(100, 10, 36),
            index=pd.date_range("2022-01", periods=36, freq="MS").to_period("M"),
        )
        model.fit(long_series)
        assert model.seasonal == "add"  # Should be restored from _init_seasonal

    def test_init_seasonal_preserved(self):
        """_init_seasonal should always hold the original config value."""
        from cashflow.models.ets import ETSModel

        model = ETSModel(trend="add", seasonal="mul")
        assert model._init_seasonal == "mul"

        # Even after fitting with short data
        short = pd.Series(np.ones(5), index=pd.date_range("2024-01", periods=5, freq="MS").to_period("M"))
        model.fit(short)
        assert model._init_seasonal == "mul"


class TestBUG3_WebNoDuplicatePipeline:
    """BUG-3: Web route should not run pipeline twice."""

    def test_run_forecast_pipeline_uses_engine_once(self):
        """run_forecast_pipeline should call engine.run_from_dataframe, not manual pipeline."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("fastapi not installed")

        import inspect
        from cashflow.web.routes.forecast import run_forecast_pipeline

        source = inspect.getsource(run_forecast_pipeline)

        # Should use engine.run_from_dataframe
        assert "run_from_dataframe" in source

        # Should NOT have the old manual pipeline pattern
        assert "enrich_with_crf" not in source


class TestBUG4_SARIMAXEvaluated:
    """BUG-4: SARIMAX should be evaluated when included in models_to_evaluate."""

    def test_sarimax_branch_exists(self):
        """_train_and_select_model should handle 'sarimax' in models_to_evaluate."""
        import inspect
        from cashflow.engine.forecast import ForecastEngine

        source = inspect.getsource(ForecastEngine._train_and_select_model)
        assert '"sarimax" in self.config.models_to_evaluate' in source

    def test_sarimax_model_evaluated(self, sample_time_series):
        """When sarimax is requested, it should be evaluated by the selector."""
        from cashflow.engine import ForecastConfig
        from cashflow.engine.forecast import ForecastEngine
        from cashflow.models.selection import ModelSelector
        from cashflow.utils import split_train_test

        config = ForecastConfig(models_to_evaluate=["sarimax"])
        engine = ForecastEngine(config)

        # We need decomposed data to run _train_and_select_model
        train, test = split_train_test(sample_time_series, test_size=4)

        df = pd.DataFrame({
            "month_key": sample_time_series.index.strftime("%Y-%m"),
            "residual_clean": sample_time_series.values,
        })

        forecast_output, summary = engine._train_and_select_model(df)
        assert summary["models_evaluated"] >= 1
        # Check that SARIMAX was one of the evaluated models
        model_names = [r["model"] for r in summary["all_results"]]
        assert any("SARIMAX" in name for name in model_names)


class TestBUG5_DataQualityScoreCorrect:
    """BUG-5: Data quality score should use transaction-level data, not monthly aggregated."""

    def test_quality_score_with_transaction_data(self):
        """calculate_data_quality_score on transaction data should give high score."""
        from cashflow.utils import calculate_data_quality_score
        from datetime import date

        # Full transaction-level data with all required fields
        df = pd.DataFrame({
            "tx_id": [f"T{i}" for i in range(10)],
            "customer_id": ["C1"] * 10,
            "account_id": ["A1"] * 10,
            "tx_date": [date(2024, 1, i + 1) for i in range(10)],
            "amount": [100.0 * (i + 1) for i in range(10)],
            "currency": ["EUR"] * 10,
            "direction": ["CREDIT"] * 5 + ["DEBIT"] * 5,
        })

        score = calculate_data_quality_score(df)
        assert score > 80  # Should be high quality

    def test_quality_score_with_monthly_data_is_lower(self):
        """calculate_data_quality_score on monthly data (missing tx_id etc.) scores lower."""
        from cashflow.utils import calculate_data_quality_score

        # Monthly aggregated data — missing most required fields
        monthly_df = pd.DataFrame({
            "month_key": ["2024-01", "2024-02", "2024-03"],
            "necf": [1000.0, 1100.0, 900.0],
            "deterministic_base": [800.0, 800.0, 800.0],
            "residual": [200.0, 300.0, 100.0],
        })

        score = calculate_data_quality_score(monthly_df)
        # Score should be lower because tx_id, customer_id, etc. are missing
        assert score < 80

    def test_engine_uses_external_df_for_quality(self):
        """Engine should use _external_df (transaction data) for quality scoring."""
        import inspect
        from cashflow.engine.forecast import ForecastEngine

        source = inspect.getsource(ForecastEngine._generate_explainability)
        assert "_external_df" in source


class TestBUG6_CustomerIdConsistency:
    """BUG-6: run_from_dataframe should accept customer_id like run() does."""

    def test_run_from_dataframe_accepts_customer_id(self):
        """run_from_dataframe should have customer_id parameter."""
        import inspect
        from cashflow.engine.forecast import ForecastEngine

        sig = inspect.signature(ForecastEngine.run_from_dataframe)
        assert "customer_id" in sig.parameters

    def test_customer_id_passed_to_aggregate(self):
        """customer_id should be passed to aggregate_monthly in run_from_dataframe."""
        import inspect
        from cashflow.engine.forecast import ForecastEngine

        source = inspect.getsource(ForecastEngine.run_from_dataframe)
        assert "aggregate_monthly(external_df, customer_id)" in source


class TestISSUE1_NoCoVarnames:
    """ISSUE-1: Model exog detection should use isinstance, not co_varnames."""

    def test_no_co_varnames_in_selection(self):
        """selection.py should not use co_varnames for exog detection."""
        import inspect
        from cashflow.models.selection import ModelSelector

        source = inspect.getsource(ModelSelector.evaluate_model)
        assert "co_varnames" not in source
        assert "isinstance" in source


class TestISSUE2_NoDeprecatedDict:
    """ISSUE-2: No deprecated .dict() calls — should use .model_dump()."""

    def test_health_endpoint_uses_model_dump(self):
        """health.py should use .model_dump() not .dict()."""
        from pathlib import Path

        health_path = Path(__file__).parent.parent.parent / "src" / "cashflow" / "web" / "routes" / "health.py"
        content = health_path.read_text()

        assert ".dict()" not in content
        assert ".model_dump()" in content
