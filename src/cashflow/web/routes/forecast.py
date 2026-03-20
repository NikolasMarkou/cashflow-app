"""Forecast API routes."""

from __future__ import annotations

import io
import logging
from typing import List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from cashflow.engine import ForecastConfig, ForecastEngine
from cashflow.pipeline import (
    clean_utf,
    aggregate_monthly,
    decompose_cashflow,
    detect_transfers,
    net_transfers,
    discover_recurring_patterns,
    apply_discovered_recurrence,
)
from cashflow.outliers.treatment import apply_residual_treatment
from cashflow.schemas.forecast import ExplainabilityPayload
from cashflow.web.schemas.response import ForecastAPIResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def run_forecast_pipeline(
    utf_df: pd.DataFrame,
    config: ForecastConfig,
    crf_df: Optional[pd.DataFrame] = None,
) -> Tuple[ExplainabilityPayload, pd.DataFrame]:
    """Run forecast pipeline once and return both payload and historical data.

    Args:
        utf_df: UTF DataFrame (raw, not yet cleaned)
        config: Forecast configuration
        crf_df: Optional CRF DataFrame

    Returns:
        Tuple of (ExplainabilityPayload, historical DataFrame)
    """
    # Run the engine once — single pipeline execution
    engine = ForecastEngine(config)
    payload = engine.run_from_dataframe(utf_df, crf_df)

    # Reconstruct historical monthly data from the payload for chart rendering
    # Build a minimal historical DataFrame from decomposition summary + payload
    historical_df = _build_historical_df_from_engine(engine, utf_df, config, crf_df)

    return payload, historical_df


def _build_historical_df_from_engine(
    engine: ForecastEngine,
    utf_df: pd.DataFrame,
    config: ForecastConfig,
    crf_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build historical DataFrame for chart rendering by reusing pipeline steps.

    Runs only the lightweight aggregation steps (no model fitting) to get
    monthly NECF data with outlier flags for visualization.
    """
    utf_df = clean_utf(utf_df)
    utf_df = detect_transfers(utf_df, config.transfer_date_tolerance_days)
    external_df, _ = net_transfers(utf_df)

    patterns_df = discover_recurring_patterns(external_df)
    if len(patterns_df) > 0:
        external_df = apply_discovered_recurrence(external_df, patterns_df)

    monthly_df = aggregate_monthly(external_df)
    decomposed_df = decompose_cashflow(monthly_df, external_df)

    treated_df = apply_residual_treatment(
        decomposed_df,
        detection_method=config.outlier_method,
        detection_threshold=config.outlier_threshold,
        treatment_method=config.outlier_treatment,
    )
    return treated_df


@router.post("/forecast", response_model=ForecastAPIResponse)
async def run_forecast(
    file: UploadFile = File(..., description="UTF CSV file"),
    forecast_horizon: int = Form(default=12, ge=1, le=24),
    wmape_threshold: float = Form(default=20.0, gt=0, le=100),
    outlier_method: str = Form(default="modified_zscore"),
    outlier_threshold: float = Form(default=3.5, gt=0),
    outlier_treatment: str = Form(default="median"),
    models_to_evaluate: List[str] = Form(default=["ets", "sarima"]),
    confidence_level: float = Form(default=0.95, gt=0, lt=1),
) -> ForecastAPIResponse:
    """Run forecast pipeline on uploaded CSV.

    Args:
        file: UTF CSV file upload
        forecast_horizon: Number of months to forecast (1-24)
        wmape_threshold: Maximum acceptable WMAPE percentage
        outlier_method: Detection method (modified_zscore, iqr, zscore, isolation_forest)
        outlier_threshold: Outlier detection threshold
        outlier_treatment: Treatment method (median, rolling_median, capped)
        models_to_evaluate: List of models to evaluate (ets, sarima, sarimax)
        confidence_level: Confidence level for prediction intervals (0.0-1.0)

    Returns:
        ForecastAPIResponse with metrics and chart data
    """
    # Validate file
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    # Read CSV into DataFrame
    content = await file.read()
    try:
        utf_df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    # Validate required columns
    required_columns = {"tx_date", "amount"}
    # Check for common column name variations
    column_mapping = {
        "TransactionDate": "tx_date",
        "TransactionID": "tx_id",
        "Amount": "amount",
        "CustomerID": "customer_id",
        "AccountID": "account_id",
        "Currency": "currency",
        "Category": "category",
        "IsRecurringFlag": "is_recurring_flag",
        "Direction": "direction",
    }

    # Apply column mapping if needed
    for old_name, new_name in column_mapping.items():
        if old_name in utf_df.columns and new_name not in utf_df.columns:
            utf_df = utf_df.rename(columns={old_name: new_name})

    # Validate models_to_evaluate
    valid_models = {"ets", "sarima", "sarimax", "tirex"}
    models_to_evaluate = [m.lower() for m in models_to_evaluate]
    invalid_models = set(models_to_evaluate) - valid_models
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid models: {invalid_models}. Valid options: {valid_models}",
        )

    if not models_to_evaluate:
        models_to_evaluate = ["ets", "sarima"]

    # Validate outlier_method
    valid_methods = {"modified_zscore", "iqr", "zscore", "isolation_forest"}
    if outlier_method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid outlier_method: {outlier_method}. Valid options: {valid_methods}",
        )

    # Validate outlier_treatment
    valid_treatments = {"median", "rolling_median", "capped"}
    if outlier_treatment not in valid_treatments:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid outlier_treatment: {outlier_treatment}. Valid options: {valid_treatments}",
        )

    # Build config
    config = ForecastConfig(
        forecast_horizon=forecast_horizon,
        wmape_threshold=wmape_threshold,
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
        outlier_treatment=outlier_treatment,
        models_to_evaluate=models_to_evaluate,
        confidence_level=confidence_level,
    )

    # Run forecast
    try:
        payload, historical_df = run_forecast_pipeline(utf_df, config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Forecast pipeline failed")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

    # Return structured response
    return ForecastAPIResponse.from_payload(payload, historical_df)
