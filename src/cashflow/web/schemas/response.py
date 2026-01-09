"""Response schemas for the web API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

from cashflow.schemas.forecast import ExplainabilityPayload


class HistoricalData(BaseModel):
    """Historical time series data for charts."""

    months: List[str]
    necf: List[float]
    outlier_months: List[str]
    outlier_values: List[float]


class ForecastData(BaseModel):
    """Forecast time series data for charts."""

    months: List[str]
    totals: List[float]
    lower_ci: List[float]
    upper_ci: List[float]


class ComponentsData(BaseModel):
    """Forecast components breakdown for charts."""

    months: List[str]
    deterministic_base: List[float]
    residual: List[float]
    delta: List[float]
    totals: List[float]


class OutlierData(BaseModel):
    """Outlier record for charts."""

    month_key: str
    original_value: float
    treated_value: float
    score: float


class ChartDataResponse(BaseModel):
    """Optimized data structure for Plotly.js charts."""

    historical: HistoricalData
    forecast: ForecastData
    components: ComponentsData
    outliers: List[OutlierData]
    model_selected: str
    wmape: float


class ModelCandidateResponse(BaseModel):
    """Model candidate for API response."""

    model_name: str
    wmape: float
    is_winner: bool
    order: Optional[List[int]] = None
    seasonal_order: Optional[List[int]] = None


class DecompositionSummaryResponse(BaseModel):
    """Decomposition summary for API response."""

    avg_necf: float
    avg_deterministic_base: float
    avg_residual: float


class TransferNettingSummaryResponse(BaseModel):
    """Transfer netting summary for API response."""

    num_transfers_removed: int
    total_volume_removed: float


class ForecastAPIResponse(BaseModel):
    """Complete API response including chart data."""

    # From ExplainabilityPayload
    model_selected: str
    model_candidates: List[ModelCandidateResponse]
    wmape_winner: float
    wmape_threshold: float
    meets_threshold: bool
    forecast_start: str
    forecast_end: str
    horizon_months: int
    confidence_level: str
    decomposition_summary: DecompositionSummaryResponse
    transfer_netting_summary: TransferNettingSummaryResponse
    outliers_detected: List[OutlierData]

    # Chart-ready data
    chart_data: ChartDataResponse

    @classmethod
    def from_payload(
        cls,
        payload: ExplainabilityPayload,
        historical_df: pd.DataFrame,
    ) -> "ForecastAPIResponse":
        """Convert ExplainabilityPayload to API response with chart data.

        Args:
            payload: The forecast explainability payload
            historical_df: DataFrame with historical monthly data (must have month_key, necf columns)

        Returns:
            ForecastAPIResponse ready for JSON serialization
        """
        # Build historical data for charts
        hist_months = historical_df["month_key"].tolist()
        hist_necf = historical_df["necf"].tolist()

        # Extract outlier points from historical data
        outlier_months = []
        outlier_values = []
        if "is_outlier" in historical_df.columns:
            outlier_df = historical_df[historical_df["is_outlier"] == True]
            outlier_months = outlier_df["month_key"].tolist()
            outlier_values = outlier_df["necf"].tolist()

        historical = HistoricalData(
            months=hist_months,
            necf=hist_necf,
            outlier_months=outlier_months,
            outlier_values=outlier_values,
        )

        # Build forecast data
        forecast = ForecastData(
            months=[fr.month_key for fr in payload.forecast_results],
            totals=[fr.forecast_total for fr in payload.forecast_results],
            lower_ci=[fr.lower_ci for fr in payload.forecast_results],
            upper_ci=[fr.upper_ci for fr in payload.forecast_results],
        )

        # Build components data
        components = ComponentsData(
            months=[fr.month_key for fr in payload.forecast_results],
            deterministic_base=[fr.deterministic_base for fr in payload.forecast_results],
            residual=[fr.forecast_residual for fr in payload.forecast_results],
            delta=[fr.known_future_delta for fr in payload.forecast_results],
            totals=[fr.forecast_total for fr in payload.forecast_results],
        )

        # Build outliers data
        outliers = [
            OutlierData(
                month_key=o.month_key,
                original_value=o.original_value,
                treated_value=o.treated_value,
                score=o.score,
            )
            for o in payload.outliers_detected
        ]

        chart_data = ChartDataResponse(
            historical=historical,
            forecast=forecast,
            components=components,
            outliers=outliers,
            model_selected=payload.model_selected,
            wmape=payload.wmape_winner,
        )

        # Build model candidates
        model_candidates = [
            ModelCandidateResponse(
                model_name=c.model_name,
                wmape=c.wmape,
                is_winner=c.is_winner,
                order=list(c.order) if c.order else None,
                seasonal_order=list(c.seasonal_order) if c.seasonal_order else None,
            )
            for c in payload.model_candidates
        ]

        return cls(
            model_selected=payload.model_selected,
            model_candidates=model_candidates,
            wmape_winner=payload.wmape_winner,
            wmape_threshold=payload.wmape_threshold,
            meets_threshold=payload.meets_threshold,
            forecast_start=payload.forecast_start,
            forecast_end=payload.forecast_end,
            horizon_months=payload.horizon_months,
            confidence_level=payload.confidence_level,
            decomposition_summary=DecompositionSummaryResponse(
                avg_necf=payload.decomposition_summary.avg_necf,
                avg_deterministic_base=payload.decomposition_summary.avg_deterministic_base,
                avg_residual=payload.decomposition_summary.avg_residual,
            ),
            transfer_netting_summary=TransferNettingSummaryResponse(
                num_transfers_removed=payload.transfer_netting_summary.num_transfers_removed,
                total_volume_removed=payload.transfer_netting_summary.total_volume_removed,
            ),
            outliers_detected=outliers,
            chart_data=chart_data,
        )
