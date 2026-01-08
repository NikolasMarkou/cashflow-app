"""Forecast output and explainability schemas - SDD Sections 14-15."""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class ForecastResult(BaseModel):
    """Single month forecast result with confidence intervals.

    Per SDD Section 14.3:
    - Monthly granularity
    - 12-month horizon
    - 95% confidence intervals
    """

    month_key: str = Field(..., pattern=r"^\d{4}-\d{2}$", description="YYYY-MM format")
    forecast_total: float = Field(..., description="Final recomposed forecast")
    forecast_residual: float = Field(..., description="Forecasted residual component")
    deterministic_base: float = Field(..., description="Known deterministic component")
    known_future_delta: float = Field(
        default=0.0, description="Adjustments from CRF events (loan endings, etc.)"
    )
    lower_ci: float = Field(..., description="Lower bound of 95% confidence interval")
    upper_ci: float = Field(..., description="Upper bound of 95% confidence interval")

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.upper_ci - self.lower_ci


class ModelCandidate(BaseModel):
    """Evaluated model candidate with performance metrics.

    Per SDD Section 13.5, model selection uses:
    1. Lowest WMAPE wins
    2. Tie-breaker: simpler model
    3. Explainability override within 0.5pp tolerance
    """

    model_name: str = Field(..., description="Model identifier (ETS, SARIMA, SARIMAX)")
    wmape: float = Field(..., ge=0, description="Weighted Mean Absolute Percentage Error")
    is_winner: bool = Field(default=False, description="Whether this model was selected")
    order: Optional[Tuple[int, ...]] = Field(None, description="Model order parameters")
    seasonal_order: Optional[Tuple[int, ...]] = Field(
        None, description="Seasonal order (for SARIMA/SARIMAX)"
    )
    params: Optional[Dict[str, float]] = Field(None, description="Fitted model parameters")


class OutlierRecord(BaseModel):
    """Record of detected outlier for audit trail."""

    month_key: str = Field(..., description="Month where outlier was detected")
    original_value: float = Field(..., description="Original residual value")
    treated_value: float = Field(..., description="Value after treatment")
    detection_method: str = Field(..., description="Method used (MZ-Score, IQR, etc.)")
    score: float = Field(..., description="Outlier detection score")
    treatment_tag: str = Field(..., description="ABNORMAL_EXTERNAL_FLOW or similar")


class TransferNettingSummary(BaseModel):
    """Summary of transfer netting operation."""

    num_transfers_removed: int = Field(..., ge=0)
    total_volume_removed: float = Field(..., ge=0)


class DecompositionSummary(BaseModel):
    """Summary statistics of cash flow decomposition."""

    avg_necf: float
    avg_deterministic_base: float
    avg_residual: float


class ExplainabilityPayload(BaseModel):
    """Full explainability JSON payload for LLM consumption.

    Per SDD Section 15.3, this is the single authoritative output
    containing all metadata for the LLM Rendering Engine.
    """

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    sdd_version: str = Field(default="v0.05")

    # Model selection (SDD 15.3.1)
    model_selected: str = Field(..., description="Final winning model name")
    model_candidates: List[ModelCandidate] = Field(..., description="All evaluated models")
    wmape_winner: float = Field(..., description="WMAPE of winning model")
    wmape_threshold: float = Field(default=20.0, description="Maximum acceptable WMAPE")
    meets_threshold: bool = Field(..., description="Whether WMAPE < threshold")

    # Forecast period
    forecast_start: str = Field(..., description="First forecast month (YYYY-MM)")
    forecast_end: str = Field(..., description="Last forecast month (YYYY-MM)")
    horizon_months: int = Field(default=12, description="Forecast horizon")
    confidence_level: str = Field(
        default="High", description="High/Medium/Low based on data quality"
    )

    # Decomposition metadata (SDD 15.3.2)
    decomposition_summary: DecompositionSummary

    # Transfer netting (SDD 15.3.3)
    transfer_netting_summary: TransferNettingSummary

    # Outliers
    outliers_detected: List[OutlierRecord] = Field(default_factory=list)

    # Forecast results
    forecast_results: List[ForecastResult]

    # Exogenous events (SDD 15.3.4)
    exogenous_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Known future events with delta_value and applied_in_model flag",
    )

    model_config = {
        "populate_by_name": True,
    }
