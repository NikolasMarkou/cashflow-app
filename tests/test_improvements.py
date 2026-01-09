"""Tests for the three architectural improvements."""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestRecurrenceDetection:
    """Test Layer 0.5 internal recurrence detection."""

    def test_discover_recurring_patterns(self):
        """Test pattern discovery from transaction data."""
        from cashflow.pipeline.recurrence import discover_recurring_patterns

        # Create synthetic data with clear recurring pattern
        transactions = []
        for month in range(1, 13):
            # Salary - should be detected as recurring
            transactions.append({
                "tx_id": f"SAL_{month}",
                "tx_date": datetime(2024, month, 25),
                "amount": 3000 + np.random.normal(0, 10),  # Small variation
                "category": "SALARY",
                "description_raw": "MONTHLY SALARY",
                "is_recurring_flag": False,  # Upstream flag says NO
            })
            # Rent - should be detected as recurring
            transactions.append({
                "tx_id": f"RENT_{month}",
                "tx_date": datetime(2024, month, 1),
                "amount": -1200,  # Fixed amount
                "category": "RENT",
                "description_raw": "MONTHLY RENT",
                "is_recurring_flag": False,  # Upstream flag says NO
            })
            # Random expense - should NOT be detected as recurring
            transactions.append({
                "tx_id": f"MISC_{month}",
                "tx_date": datetime(2024, month, 15),
                "amount": -np.random.uniform(50, 500),  # High variation
                "category": "MISCELLANEOUS",
                "description_raw": "RANDOM PURCHASE",
                "is_recurring_flag": False,
            })

        df = pd.DataFrame(transactions)
        df["tx_date"] = pd.to_datetime(df["tx_date"])

        patterns = discover_recurring_patterns(df)

        assert len(patterns) > 0, "Should discover at least one pattern"
        pattern_categories = patterns["pattern_key"].tolist()

        # SALARY and RENT should be discovered as recurring
        assert "SALARY" in pattern_categories or any("SALARY" in str(p) for p in pattern_categories)

    def test_apply_discovered_recurrence(self):
        """Test applying discovered patterns to tag transactions."""
        from cashflow.pipeline.recurrence import (
            discover_recurring_patterns,
            apply_discovered_recurrence,
        )

        # Create data where upstream flag is wrong
        transactions = []
        for month in range(1, 13):
            transactions.append({
                "tx_id": f"SAL_{month}",
                "tx_date": datetime(2024, month, 25),
                "amount": 3000,
                "category": "SALARY",
                "is_recurring_flag": False,  # Wrong!
            })

        df = pd.DataFrame(transactions)
        df["tx_date"] = pd.to_datetime(df["tx_date"])

        patterns = discover_recurring_patterns(df)
        df_with_discovery = apply_discovered_recurrence(df, patterns)

        assert "is_recurring_discovered" in df_with_discovery.columns
        # Some transactions should now be marked as recurring
        discovered_count = df_with_discovery["is_recurring_discovered"].sum()
        assert discovered_count > 0, "Should discover some recurring transactions"


class TestTrendAdjustedProjection:
    """Test trend-adjusted deterministic projection."""

    def test_compute_deterministic_projection_with_trend(self):
        """Test projection captures upward trend."""
        from cashflow.pipeline.decomposition import compute_deterministic_projection

        # Create data with clear upward trend (salary raises)
        months = pd.date_range("2024-01-01", periods=12, freq="MS")
        base_values = [1600 + i * 50 for i in range(12)]  # +50/month trend

        df = pd.DataFrame({
            "month_key": months.strftime("%Y-%m"),
            "deterministic_base": base_values,
        })

        projection = compute_deterministic_projection(df)

        assert projection.monthly_trend > 0, "Should detect upward trend"
        assert projection.base_value > 1600, "Base should reflect recent values"

        # Project 6 months ahead should be higher than current
        future_value = projection.project(6)
        assert future_value > base_values[-1], "Future projection should be higher"

    def test_detect_level_shift(self):
        """Test detection of salary raise (level shift)."""
        from cashflow.pipeline.decomposition import compute_deterministic_projection

        # Create data with level shift at month 8 (salary raise)
        months = pd.date_range("2024-01-01", periods=12, freq="MS")
        base_values = [1600] * 7 + [2000] * 5  # Jump at month 8

        df = pd.DataFrame({
            "month_key": months.strftime("%Y-%m"),
            "deterministic_base": base_values,
        })

        projection = compute_deterministic_projection(df)

        # Should use post-shift values
        assert projection.base_value >= 1900, "Should use post-shift base"
        assert projection.method == "level_shift_adjusted", "Should detect level shift"


class TestExogenousIntegration:
    """Test SARIMAX exogenous variable integration."""

    def test_sarimax_accepts_exog(self):
        """Test SARIMAX model accepts exogenous variables."""
        from cashflow.models.sarima import SARIMAXModel
        import pandas as pd

        # Create simple series
        periods = pd.period_range("2024-01", periods=24, freq="M")
        series = pd.Series(
            [100 + np.random.normal(0, 10) for _ in range(24)],
            index=periods
        )

        # Create exogenous variables
        exog = pd.DataFrame(
            {"known_delta": [0] * 20 + [100, 0, 0, 0]},
            index=periods
        )

        model = SARIMAXModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 12))

        # Should not raise
        model.fit(series, exog=exog)

        # Should be able to predict with future exog
        future_exog = pd.DataFrame(
            {"known_delta": [0] * 12},
            index=pd.period_range("2026-01", periods=12, freq="M")
        )
        forecast = model.predict(12, exog_future=future_exog)

        assert len(forecast.forecast_mean) == 12

    def test_model_selector_handles_exog(self):
        """Test ModelSelector passes exog to SARIMAX."""
        from cashflow.models.selection import ModelSelector
        from cashflow.models.sarima import SARIMAXModel
        import pandas as pd

        # Create series
        periods = pd.period_range("2024-01", periods=24, freq="M")
        full_series = pd.Series(
            [100 + np.random.normal(0, 10) for _ in range(24)],
            index=periods
        )

        train = full_series[:20]
        test = full_series[20:]

        # Create exog
        train_exog = pd.DataFrame({"known_delta": [0] * 20}, index=train.index)
        test_exog = pd.DataFrame({"known_delta": [0] * 4}, index=test.index)
        future_exog = pd.DataFrame(
            {"known_delta": [0] * 12},
            index=pd.period_range("2026-01", periods=12, freq="M")
        )

        selector = ModelSelector()
        model = SARIMAXModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 12))

        result = selector.evaluate_model(
            model=model,
            train_series=train,
            test_series=test,
            forecast_steps=12,
            train_exog=train_exog,
            test_exog=test_exog,
            future_exog=future_exog,
        )

        assert result.wmape < float("inf"), "Should complete without error"


class TestIntegratedPipeline:
    """Test all improvements work together in the full pipeline."""

    def test_full_pipeline_with_improvements(self):
        """Test ForecastEngine uses all three improvements."""
        from cashflow.engine import ForecastEngine, ForecastConfig

        # Create synthetic data
        transactions = []
        tx_id = 1
        for year in [2024, 2025]:
            for month in range(1, 13):
                if year == 2025 and month > 6:
                    break

                # Salary with slight trend
                salary = 3000 + (year - 2024) * 12 * 50 + month * 5
                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN",
                    "tx_date": datetime(year, month, 25),
                    "amount": salary,
                    "currency": "EUR",
                    "category": "SALARY",
                    "is_recurring_flag": False,  # Wrong flag - should be discovered
                })
                tx_id += 1

                # Rent
                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN",
                    "tx_date": datetime(year, month, 1),
                    "amount": -1200,
                    "currency": "EUR",
                    "category": "RENT",
                    "is_recurring_flag": True,
                })
                tx_id += 1

                # Variable expenses
                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN",
                    "tx_date": datetime(year, month, 15),
                    "amount": -np.random.uniform(100, 300),
                    "currency": "EUR",
                    "category": "GROCERIES",
                    "is_recurring_flag": False,
                })
                tx_id += 1

        df = pd.DataFrame(transactions)
        df["tx_date"] = pd.to_datetime(df["tx_date"])

        # Run with improvements
        engine = ForecastEngine(ForecastConfig())
        payload = engine.run_from_dataframe(df)

        # Verify improvements are active
        assert payload.model_selected is not None
        assert payload.wmape_winner is not None
        assert len(payload.forecast_results) == 12

        # Check recurrence discovery was used
        assert engine._recurrence_summary is not None

        # Check trend-adjusted projection was used
        assert engine._deterministic_projection is not None
        assert engine._deterministic_projection.method in [
            "exponential_weighted_trend",
            "level_shift_adjusted",
            "mean_fallback",
        ]
