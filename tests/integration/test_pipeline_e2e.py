"""End-to-end integration tests for the forecast pipeline."""

import pandas as pd
import pytest

from cashflow.engine import ForecastEngine, ForecastConfig
from cashflow.pipeline import (
    clean_utf,
    detect_transfers,
    net_transfers,
    aggregate_monthly,
    decompose_cashflow,
)
from cashflow.outliers.treatment import apply_residual_treatment


class TestPipelineE2E:
    """End-to-end pipeline tests."""

    def test_full_pipeline_runs(self, sample_utf_data):
        """Full pipeline should run without errors."""
        # Clean
        cleaned = clean_utf(sample_utf_data)
        assert len(cleaned) > 0

        # Transfer detection and netting
        with_transfers = detect_transfers(cleaned)
        external, summary = net_transfers(with_transfers)

        # Should have removed some internal transfers
        assert summary["num_transfers_removed"] > 0

        # Aggregate monthly
        monthly = aggregate_monthly(external)
        assert len(monthly) == 24  # 24 months of data

        # Decompose
        decomposed = decompose_cashflow(monthly, external)
        assert "deterministic_base" in decomposed.columns
        assert "residual" in decomposed.columns

        # Outlier treatment
        treated = apply_residual_treatment(decomposed)
        assert "residual_clean" in treated.columns
        assert "is_outlier" in treated.columns

    def test_outliers_detected_correctly(self, sample_utf_data):
        """Pipeline should detect the known outliers."""
        cleaned = clean_utf(sample_utf_data)
        with_transfers = detect_transfers(cleaned)
        external, _ = net_transfers(with_transfers)
        monthly = aggregate_monthly(external)
        decomposed = decompose_cashflow(monthly, external)
        treated = apply_residual_treatment(decomposed)

        # We added outliers in July and August 2024
        outlier_months = treated[treated["is_outlier"]]["month_key"].tolist()

        # Should detect at least some of our injected outliers
        assert len(outlier_months) > 0


class TestForecastEngine:
    """Tests for the main forecast engine."""

    def test_engine_from_dataframe(self, sample_utf_data):
        """Engine should produce forecast from DataFrame."""
        config = ForecastConfig(
            forecast_horizon=6,  # Shorter for faster tests
            models_to_evaluate=["ets"],  # Just ETS for speed
        )
        engine = ForecastEngine(config)

        result = engine.run_from_dataframe(sample_utf_data)

        # Check result structure
        assert result.model_selected is not None
        assert result.wmape_winner > 0
        assert len(result.forecast_results) == 6
        assert result.forecast_start is not None
        assert result.forecast_end is not None

    def test_explainability_payload_complete(self, sample_utf_data):
        """Explainability payload should have all required fields."""
        config = ForecastConfig(
            forecast_horizon=6,
            models_to_evaluate=["ets"],
        )
        engine = ForecastEngine(config)

        result = engine.run_from_dataframe(sample_utf_data)

        # Check mandatory fields per SDD 15.3.1
        assert result.model_selected is not None
        assert len(result.model_candidates) > 0
        assert result.wmape_winner is not None
        assert isinstance(result.meets_threshold, bool)
        assert result.forecast_start is not None
        assert result.forecast_end is not None
        assert result.confidence_level in ["High", "Medium", "Low"]

        # Check decomposition summary
        assert result.decomposition_summary.avg_necf is not None
        assert result.decomposition_summary.avg_deterministic_base is not None
        assert result.decomposition_summary.avg_residual is not None

        # Check transfer netting summary
        assert result.transfer_netting_summary.num_transfers_removed >= 0

    def test_forecast_recomposition(self, sample_utf_data):
        """Forecast should properly recompose components."""
        config = ForecastConfig(
            forecast_horizon=6,
            models_to_evaluate=["ets"],
        )
        engine = ForecastEngine(config)

        result = engine.run_from_dataframe(sample_utf_data)

        for forecast in result.forecast_results:
            # Verify recomposition formula
            expected_total = (
                forecast.forecast_residual
                + forecast.deterministic_base
                + forecast.known_future_delta
            )
            assert abs(forecast.forecast_total - expected_total) < 0.01

            # Verify CI order
            assert forecast.lower_ci < forecast.forecast_total < forecast.upper_ci
