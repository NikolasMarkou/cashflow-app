"""Forecast engine configuration - SDD operational parameters."""

from __future__ import annotations

from typing import List, Tuple
from pydantic import BaseModel, Field


class ForecastConfig(BaseModel):
    """Runtime configuration for the forecast engine.

    Contains all configurable parameters per SDD requirements.
    """

    # Data settings (SDD Section 17)
    lookback_min_months: int = Field(
        default=24,
        ge=12,
        description="Minimum required historical months",
    )
    lookback_max_months: int = Field(
        default=60,
        ge=24,
        description="Maximum historical months to use",
    )
    forecast_horizon: int = Field(
        default=12,
        ge=1,
        le=24,
        description="Number of months to forecast",
    )

    # Transfer netting (SDD Section 9.2)
    transfer_date_tolerance_days: int = Field(
        default=2,
        ge=0,
        le=7,
        description="Max days between transfer pairs",
    )

    # Outlier detection (SDD Section 11)
    outlier_method: str = Field(
        default="modified_zscore",
        description="Detection method: iqr, zscore, modified_zscore, isolation_forest",
    )
    outlier_threshold: float = Field(
        default=3.5,
        gt=0,
        description="Outlier detection threshold",
    )
    outlier_treatment: str = Field(
        default="median",
        description="Treatment method: median, rolling_median, capped",
    )

    # Model settings (SDD Section 13)
    models_to_evaluate: List[str] = Field(
        default=["ets", "sarima"],
        description="Models to evaluate: ets, sarima, sarimax, tirex",
    )
    arima_order: Tuple[int, int, int] = Field(
        default=(1, 1, 1),
        description="ARIMA (p, d, q) order",
    )
    seasonal_order: Tuple[int, int, int, int] = Field(
        default=(1, 1, 0, 12),
        description="Seasonal (P, D, Q, s) order",
    )
    enable_ml_layer: bool = Field(
        default=False,
        description="Enable Layer 2 ML models (Ridge/ElasticNet)",
    )
    wmape_threshold: float = Field(
        default=20.0,
        gt=0,
        le=100,
        description="Maximum acceptable WMAPE",
    )
    model_tie_tolerance: float = Field(
        default=0.5,
        ge=0,
        description="WMAPE tolerance for tie-breaking",
    )

    # Confidence intervals
    confidence_level: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description="Confidence level for prediction intervals",
    )

    # Train/test split
    test_size: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Number of months for test set",
    )

    # Output settings
    output_format: str = Field(
        default="json",
        description="Output format: json, csv, both",
    )

    model_config = {
        "validate_assignment": True,
    }


def get_default_config() -> ForecastConfig:
    """Get default configuration."""
    return ForecastConfig()


def load_config(path: str) -> ForecastConfig:
    """Load configuration from JSON file."""
    import json
    from pathlib import Path

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        data = json.load(f)

    return ForecastConfig(**data)
