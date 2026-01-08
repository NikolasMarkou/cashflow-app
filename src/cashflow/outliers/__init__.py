"""Outlier detection and treatment - SDD Section 11."""

from __future__ import annotations
from cashflow.outliers.detector import (
    detect_outliers,
    modified_zscore,
    iqr_outliers,
    isolation_forest_outliers,
)
from cashflow.outliers.treatment import treat_outliers, TreatmentMethod

__all__ = [
    "detect_outliers",
    "modified_zscore",
    "iqr_outliers",
    "isolation_forest_outliers",
    "treat_outliers",
    "TreatmentMethod",
]
