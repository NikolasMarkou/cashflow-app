"""Unit tests for outlier detection and treatment."""

import numpy as np
import pandas as pd
import pytest

from cashflow.outliers import (
    detect_outliers,
    modified_zscore,
    iqr_outliers,
    treat_outliers,
    TreatmentMethod,
)


class TestModifiedZScore:
    """Tests for Modified Z-Score outlier detection."""

    def test_no_outliers_in_normal_data(self):
        """Normal data should have no outliers."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(100, 10, 100))

        is_outlier, scores = modified_zscore(series, threshold=3.5)

        assert is_outlier.sum() == 0

    def test_detects_obvious_outlier(self):
        """Should detect an obvious outlier."""
        # Normal data with one extreme value
        data = [100, 102, 98, 101, 99, 103, 97, 100, 1000, 101]
        series = pd.Series(data)

        is_outlier, scores = modified_zscore(series, threshold=3.5)

        assert is_outlier.sum() == 1
        assert is_outlier.iloc[8] == True  # The 1000 value

    def test_threshold_affects_detection(self):
        """Lower threshold should detect more outliers."""
        data = [100, 102, 98, 150, 99, 103, 97, 100, 200, 101]
        series = pd.Series(data)

        is_outlier_strict, _ = modified_zscore(series, threshold=3.5)
        is_outlier_loose, _ = modified_zscore(series, threshold=2.0)

        assert is_outlier_loose.sum() >= is_outlier_strict.sum()

    def test_handles_empty_series(self):
        """Should handle empty series gracefully."""
        series = pd.Series(dtype=float)

        is_outlier, scores = modified_zscore(series)

        assert len(is_outlier) == 0
        assert len(scores) == 0

    def test_handles_constant_series(self):
        """Should handle series with no variance."""
        series = pd.Series([100.0] * 10)

        is_outlier, scores = modified_zscore(series)

        # With no variance, no outliers should be detected
        assert is_outlier.sum() == 0


class TestIQROutliers:
    """Tests for IQR-based outlier detection."""

    def test_detects_outliers_beyond_whiskers(self):
        """Should detect values beyond 1.5*IQR."""
        # Create data with clear outliers
        data = list(range(1, 21)) + [100]  # 1-20 plus one outlier
        series = pd.Series(data)

        is_outlier, _ = iqr_outliers(series, multiplier=1.5)

        assert is_outlier.iloc[-1] == True  # The 100 value


class TestOutlierTreatment:
    """Tests for outlier treatment."""

    def test_median_treatment(self):
        """Median treatment should replace outliers with median."""
        df = pd.DataFrame({
            "value": [100, 102, 98, 1000, 99],
            "is_outlier": [False, False, False, True, False],
        })

        result = treat_outliers(df, "value", "is_outlier", method="median")

        # Outlier should be replaced with median of non-outliers
        expected_median = pd.Series([100, 102, 98, 99]).median()
        assert result.loc[3, "value_clean"] == expected_median

    def test_preserves_original_values(self):
        """Treatment should preserve original values for audit."""
        df = pd.DataFrame({
            "value": [100, 1000],
            "is_outlier": [False, True],
        })

        result = treat_outliers(df, "value", "is_outlier", method="median")

        assert "value_original" in result.columns
        assert result.loc[1, "value_original"] == 1000

    def test_adds_treatment_tag(self):
        """Treatment should add appropriate tags."""
        df = pd.DataFrame({
            "value": [100, 1000],
            "is_outlier": [False, True],
        })

        result = treat_outliers(df, "value", "is_outlier", method="median")

        assert result.loc[0, "treatment_tag"] == "NORMAL"
        assert result.loc[1, "treatment_tag"] == "ABNORMAL_EXTERNAL_FLOW"
