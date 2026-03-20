"""Outlier detection methods - SDD Section 11.3."""

from __future__ import annotations
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import logging

logger = logging.getLogger(__name__)


class DetectionMethod(str, Enum):
    """Supported outlier detection methods."""

    IQR = "iqr"
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    ISOLATION_FOREST = "isolation_forest"


def detect_outliers(
    series: pd.Series,
    method: str = "modified_zscore",
    threshold: float = 3.5,
    **kwargs,
) -> tuple[pd.Series, pd.Series]:
    """Detect outliers in a series using the specified method.

    Per SDD Section 11.2, outlier detection is applied only to the
    Residual Series, never to deterministic components.

    Args:
        series: Pandas Series to analyze
        method: Detection method (iqr, zscore, modified_zscore, isolation_forest)
        threshold: Threshold for detection (meaning depends on method)
        **kwargs: Additional method-specific parameters

    Returns:
        Tuple of (is_outlier boolean Series, scores Series)
    """
    method = method.lower()

    if method == DetectionMethod.IQR.value:
        # IQR default multiplier is 1.5; only apply caller's threshold if they
        # explicitly changed it from the global default of 3.5
        iqr_default = 1.5
        return iqr_outliers(series, multiplier=iqr_default)
    elif method == DetectionMethod.ZSCORE.value:
        return zscore_outliers(series, threshold=threshold)
    elif method == DetectionMethod.MODIFIED_ZSCORE.value:
        return modified_zscore(series, threshold=threshold)
    elif method == DetectionMethod.ISOLATION_FOREST.value:
        return isolation_forest_outliers(series, **kwargs)
    else:
        raise ValueError(f"Unknown detection method: {method}")


def modified_zscore(
    series: pd.Series,
    threshold: float = 3.5,
) -> tuple[pd.Series, pd.Series]:
    """Detect outliers using Modified Z-Score (MAD-based).

    The Modified Z-Score uses Median Absolute Deviation (MAD) instead
    of standard deviation, making it robust to outliers themselves.

    Formula: MZ = 0.6745 * (x - median) / MAD

    Per SDD Section 11.3, this is the recommended method with
    threshold |MZ| > 3.5.

    Args:
        series: Pandas Series to analyze
        threshold: MZ-Score threshold (default 3.5 per SDD)

    Returns:
        Tuple of (is_outlier boolean Series, mz_scores Series)
    """
    series = series.dropna()

    if len(series) == 0:
        return pd.Series(dtype=bool), pd.Series(dtype=float)

    median = series.median()
    mad = (series - median).abs().median()

    # Prevent division by zero
    if mad == 0:
        logger.warning("MAD is zero - no variance in data, no outliers detected")
        mz_scores = pd.Series(0.0, index=series.index)
        is_outlier = pd.Series(False, index=series.index)
    else:
        # 0.6745 is the 0.75th quantile of the standard normal distribution
        mz_scores = 0.6745 * (series - median) / mad
        is_outlier = mz_scores.abs() > threshold

    outlier_count = is_outlier.sum()
    if outlier_count > 0:
        logger.info(f"Modified Z-Score: detected {outlier_count} outliers (threshold={threshold})")

    return is_outlier, mz_scores


def zscore_outliers(
    series: pd.Series,
    threshold: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Detect outliers using standard Z-Score.

    Z-Score measures how many standard deviations a value is from the mean.
    Less robust than Modified Z-Score as it's sensitive to outliers.

    Args:
        series: Pandas Series to analyze
        threshold: Z-Score threshold (default 3.0)

    Returns:
        Tuple of (is_outlier boolean Series, z_scores Series)
    """
    series = series.dropna()

    if len(series) == 0:
        return pd.Series(dtype=bool), pd.Series(dtype=float)

    mean = series.mean()
    std = series.std()

    if std == 0:
        z_scores = pd.Series(0.0, index=series.index)
        is_outlier = pd.Series(False, index=series.index)
    else:
        z_scores = (series - mean) / std
        is_outlier = z_scores.abs() > threshold

    return is_outlier, z_scores


def iqr_outliers(
    series: pd.Series,
    multiplier: float = 1.5,
) -> tuple[pd.Series, pd.Series]:
    """Detect outliers using Interquartile Range (IQR).

    Per SDD Section 11.3, IQR is the default baseline method.
    Outliers are values outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR].

    Args:
        series: Pandas Series to analyze
        multiplier: IQR multiplier (default 1.5 for standard, 3.0 for extreme)

    Returns:
        Tuple of (is_outlier boolean Series, iqr_scores Series)
    """
    series = series.dropna()

    if len(series) == 0:
        return pd.Series(dtype=bool), pd.Series(dtype=float)

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        # No spread in middle 50%
        iqr_scores = pd.Series(0.0, index=series.index)
        is_outlier = pd.Series(False, index=series.index)
    else:
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # Compute score as distance from nearest bound, normalized by IQR
        iqr_scores = pd.Series(0.0, index=series.index)
        below = series < lower_bound
        above = series > upper_bound
        iqr_scores[below] = (lower_bound - series[below]) / iqr
        iqr_scores[above] = (series[above] - upper_bound) / iqr

        is_outlier = below | above

    return is_outlier, iqr_scores


def isolation_forest_outliers(
    series: pd.Series,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Detect outliers using Isolation Forest.

    Per SDD Section 11.3, this is optional for high-volume customers.
    Isolation Forest is effective for multivariate outlier detection
    but here we use it univariately on the residual series.

    Args:
        series: Pandas Series to analyze
        contamination: Expected proportion of outliers (default 5%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (is_outlier boolean Series, anomaly_scores Series)
    """
    series = series.dropna()

    if len(series) < 10:
        # Not enough data for Isolation Forest
        logger.warning("Insufficient data for Isolation Forest, falling back to MZ-Score")
        return modified_zscore(series)

    # Reshape for sklearn
    X = series.values.reshape(-1, 1)

    # Fit Isolation Forest
    clf = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    predictions = clf.fit_predict(X)
    scores = clf.decision_function(X)

    # predictions: 1 = inlier, -1 = outlier
    is_outlier = pd.Series(predictions == -1, index=series.index)

    # Invert scores so higher = more anomalous (consistent with other methods)
    anomaly_scores = pd.Series(-scores, index=series.index)

    return is_outlier, anomaly_scores
