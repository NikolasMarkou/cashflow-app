"""Tests verifying fixes for all 9 code review findings.

Each test class corresponds to one finding:
- C1: Hardcoded data quality score
- C2: Recurrence source tag ordering
- C3: run() missing Layer 0.5
- H1: ETS constant-width confidence intervals
- H2: float(inf) WMAPE in explainability payload
- H3: Dead exog code removed
- M1: IQR threshold dispatch
- M2: TransferLinkID with 3+ transactions
- L1: Duplicate import
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date


class TestC1_DataQualityScore:
    """C1: data_quality_score must be calculated, not hardcoded."""

    def test_calculate_data_quality_score_returns_number(self):
        """calculate_data_quality_score should return a float between 0-100."""
        from cashflow.utils import calculate_data_quality_score

        df = pd.DataFrame({
            "tx_id": ["T1", "T2", "T3"],
            "customer_id": ["C1", "C1", "C1"],
            "account_id": ["A1", "A1", "A1"],
            "tx_date": [date(2024, 1, 1)] * 3,
            "amount": [100.0, -50.0, 200.0],
            "currency": ["EUR"] * 3,
            "direction": ["CREDIT", "DEBIT", "CREDIT"],
        })

        score = calculate_data_quality_score(df)
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_low_quality_data_gets_low_score(self):
        """Mostly-NaN data should get a low quality score."""
        from cashflow.utils import calculate_data_quality_score

        df = pd.DataFrame({
            "tx_id": [None, None, "T3"],
            "customer_id": [None, None, "C1"],
            "account_id": [None, None, "A1"],
            "tx_date": [None, None, date(2024, 1, 1)],
            "amount": [None, None, 100.0],
            "currency": [None, None, "EUR"],
            "direction": [None, None, "CREDIT"],
        })

        score = calculate_data_quality_score(df)
        assert score < 60  # Low quality data

    def test_engine_uses_calculated_score(self):
        """ForecastEngine._generate_explainability should not use hardcoded 95.0."""
        import inspect
        from cashflow.engine.forecast import ForecastEngine

        source = inspect.getsource(ForecastEngine._generate_explainability)
        # Should NOT contain the hardcoded value
        assert "data_quality_score=95.0" not in source
        # Should call the calculation function
        assert "calculate_data_quality_score" in source


class TestC2_RecurrenceSourceTag:
    """C2: upstream_flag source must be assigned before union."""

    def test_upstream_flag_source_assigned(self):
        """Transactions from upstream is_recurring_flag should get source 'upstream_flag'."""
        from cashflow.pipeline.recurrence import apply_discovered_recurrence

        # Create transactions: some from upstream flag, some from discovery
        df = pd.DataFrame({
            "tx_id": ["T1", "T2", "T3", "T4"],
            "customer_id": ["C1"] * 4,
            "month_key": ["2024-01"] * 4,
            "amount": [3000.0, -1200.0, -100.0, -50.0],
            "category": ["SALARY", "RENT_MORTGAGE", "GROCERIES", "OTHER"],
            "is_recurring_flag": [True, True, False, False],
        })

        # Patterns only discover SALARY (not RENT_MORTGAGE)
        patterns = pd.DataFrame([{
            "pattern_type": "category",
            "pattern_key": "SALARY",
            "avg_amount": 3000.0,
            "std_amount": 50.0,
            "coefficient_of_variation": 0.02,
            "months_active": 12,
            "transaction_count": 12,
            "confidence_score": 0.98,
            "detection_method": "category_stability",
        }])

        result = apply_discovered_recurrence(df, patterns)

        # RENT_MORTGAGE was only from upstream flag → source should be "upstream_flag"
        rent_row = result[result["category"] == "RENT_MORTGAGE"].iloc[0]
        assert rent_row["recurrence_detection_source"] == "upstream_flag"
        assert rent_row["is_recurring_discovered"] == True

    def test_discovered_source_assigned(self):
        """Transactions found by discovery should get source 'category_discovery'."""
        from cashflow.pipeline.recurrence import apply_discovered_recurrence

        df = pd.DataFrame({
            "tx_id": ["T1", "T2"],
            "customer_id": ["C1"] * 2,
            "month_key": ["2024-01"] * 2,
            "amount": [3000.0, -100.0],
            "category": ["SALARY", "GROCERIES"],
            "is_recurring_flag": [False, False],
        })

        patterns = pd.DataFrame([{
            "pattern_type": "category",
            "pattern_key": "SALARY",
            "avg_amount": 3000.0,
            "std_amount": 50.0,
            "coefficient_of_variation": 0.02,
            "months_active": 12,
            "transaction_count": 12,
            "confidence_score": 0.98,
            "detection_method": "category_stability",
        }])

        result = apply_discovered_recurrence(df, patterns)

        salary_row = result[result["category"] == "SALARY"].iloc[0]
        assert salary_row["recurrence_detection_source"] == "category_discovery"
        assert salary_row["is_recurring_discovered"] == True


class TestC3_RunLayerHalf:
    """C3: run() method must include Layer 0.5 recurrence discovery."""

    def test_run_method_has_recurrence_discovery(self):
        """run() should call discover_recurring_patterns like run_from_dataframe."""
        import inspect
        from cashflow.engine.forecast import ForecastEngine

        source = inspect.getsource(ForecastEngine.run)
        assert "discover_recurring_patterns" in source
        assert "apply_discovered_recurrence" in source

    def test_run_from_dataframe_sets_recurrence_summary(self, sample_utf_data):
        """run_from_dataframe should populate _recurrence_summary."""
        from cashflow.engine import ForecastEngine, ForecastConfig

        config = ForecastConfig(
            forecast_horizon=6,
            models_to_evaluate=["ets"],
        )
        engine = ForecastEngine(config)
        engine.run_from_dataframe(sample_utf_data)

        assert engine._recurrence_summary is not None
        assert "newly_discovered" in engine._recurrence_summary


class TestH1_ETSHorizonCI:
    """H1: ETS confidence intervals must widen with forecast horizon."""

    def test_ci_width_increases_with_horizon(self, sample_time_series):
        """Later forecast months should have wider confidence intervals."""
        from cashflow.models.ets import ETSModel

        model = ETSModel(trend="add", seasonal=None)
        model.fit(sample_time_series)
        output = model.predict(steps=12)

        widths = output.forecast_upper - output.forecast_lower

        # Each month's CI should be wider than the previous
        for i in range(1, len(widths)):
            assert widths[i] > widths[i - 1], (
                f"CI width at step {i+1} ({widths[i]:.2f}) should be wider than "
                f"step {i} ({widths[i-1]:.2f})"
            )

    def test_ci_scaling_factor_is_sqrt(self, sample_time_series):
        """CI width should scale approximately as sqrt(step)."""
        from cashflow.models.ets import ETSModel

        model = ETSModel(trend="add", seasonal=None)
        model.fit(sample_time_series)
        output = model.predict(steps=12)

        widths = output.forecast_upper - output.forecast_lower

        # Width ratio between step N and step 1 should be ~sqrt(N)
        width_1 = widths[0]
        for step in [4, 9]:  # sqrt(4)=2, sqrt(9)=3
            expected_ratio = np.sqrt(step + 1)
            actual_ratio = widths[step] / width_1
            assert abs(actual_ratio - expected_ratio) < 0.1, (
                f"Step {step+1}: expected ratio ~{expected_ratio:.2f}, got {actual_ratio:.2f}"
            )


class TestH2_InfWMAPE:
    """H2: Failed models (inf WMAPE) must not appear in ExplainabilityPayload."""

    def test_failed_models_filtered_from_payload(self):
        """Models with inf WMAPE should not appear in model_candidates."""
        from cashflow.engine.forecast import ForecastEngine
        from cashflow.schemas.forecast import ForecastResult

        engine = ForecastEngine()

        model_summary = {
            "winner": "ETS",
            "winner_wmape": 15.0,
            "meets_threshold": True,
            "all_results": [
                {"model": "ETS", "wmape": 15.0, "error": None, "is_fallback": False},
                {"model": "SARIMA", "wmape": float("inf"), "error": "convergence failed", "is_fallback": False},
            ],
        }

        # Set required internal state
        engine._transfer_summary = {"num_transfers_removed": 0, "total_volume_removed": 0.0}

        historical_df = pd.DataFrame({
            "month_key": ["2024-01"],
            "necf": [1000.0],
            "deterministic_base": [800.0],
            "residual": [200.0],
        })

        forecast_results = [
            ForecastResult(
                month_key="2026-01",
                forecast_total=1000.0,
                forecast_residual=200.0,
                deterministic_base=800.0,
                lower_ci=900.0,
                upper_ci=1100.0,
            )
        ]

        payload = engine._generate_explainability(forecast_results, historical_df, model_summary)

        # Only ETS should appear (SARIMA had inf WMAPE / error)
        assert len(payload.model_candidates) == 1
        assert payload.model_candidates[0].model_name == "ETS"


class TestH3_DeadExogRemoved:
    """H3: _build_exog_matrix and exog parameter should be removed."""

    def test_no_build_exog_matrix_method(self):
        """ForecastEngine should not have _build_exog_matrix method."""
        from cashflow.engine.forecast import ForecastEngine

        assert not hasattr(ForecastEngine, "_build_exog_matrix")

    def test_no_exog_matrix_attribute(self):
        """ForecastEngine instance should not have _exog_matrix attribute."""
        from cashflow.engine.forecast import ForecastEngine

        engine = ForecastEngine()
        assert not hasattr(engine, "_exog_matrix")

    def test_train_and_select_model_no_exog_param(self):
        """_train_and_select_model should not accept exog parameter."""
        import inspect
        from cashflow.engine.forecast import ForecastEngine

        sig = inspect.signature(ForecastEngine._train_and_select_model)
        assert "exog" not in sig.parameters


class TestM1_IQRThreshold:
    """M1: IQR method should use its own default multiplier (1.5)."""

    def test_iqr_uses_own_default(self):
        """IQR method should use multiplier=1.5, not the global 3.5 default."""
        from cashflow.outliers.detector import detect_outliers

        # Data with a clear outlier
        data = list(range(1, 21)) + [100]
        series = pd.Series(data, dtype=float)

        # Using IQR with default threshold=3.5 (global default for modified_zscore)
        is_outlier_iqr, _ = detect_outliers(series, method="iqr", threshold=3.5)

        # Using IQR directly with multiplier=1.5
        from cashflow.outliers.detector import iqr_outliers
        is_outlier_direct, _ = iqr_outliers(series, multiplier=1.5)

        # Both should produce the same result (IQR always uses 1.5)
        assert is_outlier_iqr.sum() == is_outlier_direct.sum()

    def test_iqr_detects_outlier_at_standard_threshold(self):
        """IQR with standard 1.5 multiplier should detect extreme values."""
        from cashflow.outliers.detector import detect_outliers

        series = pd.Series(list(range(1, 21)) + [100], dtype=float)
        is_outlier, _ = detect_outliers(series, method="iqr")

        assert is_outlier.iloc[-1] == True  # Value 100 should be outlier


class TestM2_TransferLink3Plus:
    """M2: TransferLinkID groups with 3+ transactions should match pairs."""

    def test_three_transactions_same_link_id(self):
        """Groups of 3+ should still match opposite-amount pairs."""
        from cashflow.pipeline.transfer import detect_transfers

        df = pd.DataFrame({
            "tx_id": ["T1", "T2", "T3"],
            "customer_id": ["C1"] * 3,
            "account_id": ["A1", "A2", "A1"],
            "tx_date": [date(2024, 1, 15)] * 3,
            "amount": [500.0, -500.0, 100.0],  # T1 and T2 are a pair, T3 is unmatched
            "currency": ["EUR"] * 3,
            "direction": ["CREDIT", "DEBIT", "CREDIT"],
            "category": ["OTHER"] * 3,
            "transfer_link_id": ["LINK001"] * 3,
        })

        result = detect_transfers(df)

        # T1 (+500) and T2 (-500) should be matched as transfer pair
        assert result.loc[result["tx_id"] == "T1", "is_internal_transfer"].iloc[0] == True
        assert result.loc[result["tx_id"] == "T2", "is_internal_transfer"].iloc[0] == True
        # T3 (100) has no opposite match, should NOT be marked
        assert result.loc[result["tx_id"] == "T3", "is_internal_transfer"].iloc[0] == False

    def test_four_transactions_two_pairs(self):
        """Groups of 4 with two opposite pairs should match both."""
        from cashflow.pipeline.transfer import detect_transfers

        df = pd.DataFrame({
            "tx_id": ["T1", "T2", "T3", "T4"],
            "customer_id": ["C1"] * 4,
            "account_id": ["A1", "A2", "A1", "A2"],
            "tx_date": [date(2024, 1, 15)] * 4,
            "amount": [500.0, -500.0, 200.0, -200.0],
            "currency": ["EUR"] * 4,
            "direction": ["CREDIT", "DEBIT", "CREDIT", "DEBIT"],
            "category": ["OTHER"] * 4,
            "transfer_link_id": ["LINK001"] * 4,
        })

        result = detect_transfers(df)

        # All 4 should be matched as two pairs
        assert result["is_internal_transfer"].sum() == 4


class TestL1_DuplicateImport:
    """L1: engine/config.py should not have duplicate import."""

    def test_no_duplicate_import(self):
        """config.py should have exactly one 'from __future__ import annotations'."""
        from pathlib import Path
        config_path = Path(__file__).parent.parent.parent / "src" / "cashflow" / "engine" / "config.py"
        content = config_path.read_text()

        count = content.count("from __future__ import annotations")
        assert count == 1, f"Expected 1 occurrence, found {count}"
