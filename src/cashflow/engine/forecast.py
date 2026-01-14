"""Main forecast engine - SDD orchestration."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging

import pandas as pd

from cashflow.engine.config import ForecastConfig
from cashflow.pipeline import (
    load_utf,
    load_crf,
    clean_utf,
    enrich_with_crf,
    detect_transfers,
    net_transfers,
    aggregate_monthly,
    decompose_cashflow,
    compute_deterministic_projection,
    discover_recurring_patterns,
    apply_discovered_recurrence,
    get_recurrence_summary,
)
from cashflow.outliers import detect_outliers
from cashflow.outliers.treatment import apply_residual_treatment
from cashflow.models import ETSModel, SARIMAModel
from cashflow.models.selection import ModelSelector
from cashflow.schemas.forecast import (
    ExplainabilityPayload,
    ForecastResult,
    ModelCandidate,
    OutlierRecord,
    TransferNettingSummary,
    DecompositionSummary,
)
from cashflow.utils import split_train_test, get_forecast_period, determine_confidence_level

logger = logging.getLogger(__name__)


class ForecastEngine:
    """Main orchestrator for the cash flow forecasting pipeline.

    Implements the full SDD v0.05 layered architecture:
    - Layer 0: Deterministic rules (transfer netting, recurrence)
    - Layer 1: Statistical baselines (ETS, SARIMA, SARIMAX)
    - Layer 2: ML residuals (optional)
    - Layer 3: Recomposition & explainability
    """

    def __init__(self, config: Optional[ForecastConfig] = None):
        """Initialize forecast engine.

        Args:
            config: Forecast configuration (uses defaults if not provided)
        """
        self.config = config or ForecastConfig()
        self._transfer_summary: Optional[dict] = None
        self._decomposition_summary: Optional[dict] = None
        self._outlier_records: list[dict] = []
        self._model_selector: Optional[ModelSelector] = None
        self._recurrence_summary: Optional[dict] = None
        self._deterministic_projection = None
        self._exog_matrix: Optional[pd.DataFrame] = None

    def run(
        self,
        utf_path: str | Path,
        crf_path: Optional[str | Path] = None,
        customer_id: Optional[str] = None,
    ) -> ExplainabilityPayload:
        """Run the complete forecast pipeline.

        Args:
            utf_path: Path to UTF CSV file
            crf_path: Optional path to CRF CSV file
            customer_id: Optional customer ID filter

        Returns:
            ExplainabilityPayload with full forecast results
        """
        logger.info("Starting forecast pipeline")

        # Phase 1: Data ingestion and cleaning
        utf_df = self._ingest_and_clean(utf_path, customer_id)

        # Phase 2: CRF enrichment (if available)
        crf_df = None
        if crf_path:
            crf_df = load_crf(crf_path)
            utf_df = enrich_with_crf(utf_df, crf_df)

        # Phase 3: Transfer detection and netting (Layer 0)
        utf_df = detect_transfers(utf_df, self.config.transfer_date_tolerance_days)
        external_df, self._transfer_summary = net_transfers(utf_df)

        # Phase 4: Monthly aggregation
        monthly_df = aggregate_monthly(external_df, customer_id)

        # Phase 5: Cash flow decomposition
        decomposed_df = decompose_cashflow(monthly_df, external_df)

        # Phase 6: Outlier detection and treatment (on residuals only)
        treated_df = apply_residual_treatment(
            decomposed_df,
            detection_method=self.config.outlier_method,
            detection_threshold=self.config.outlier_threshold,
            treatment_method=self.config.outlier_treatment,
        )

        # Record outliers for explainability
        self._record_outliers(treated_df)

        # Phase 7: Model training and selection (Layer 1)
        forecast_output, model_summary = self._train_and_select_model(treated_df)

        # Phase 8: Forecast recomposition (Layer 3)
        forecast_results = self._recompose_forecast(
            forecast_output,
            treated_df,
            crf_df,
        )

        # Phase 9: Generate explainability payload
        payload = self._generate_explainability(
            forecast_results,
            treated_df,
            model_summary,
        )

        logger.info("Forecast pipeline completed successfully")

        return payload

    def _ingest_and_clean(
        self,
        utf_path: str | Path,
        customer_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Ingest and clean UTF data."""
        logger.info(f"Loading UTF from {utf_path}")

        utf_df = load_utf(utf_path, customer_id)
        utf_df = clean_utf(utf_df)

        logger.info(f"Loaded {len(utf_df)} transactions")

        return utf_df

    def _record_outliers(self, df: pd.DataFrame) -> None:
        """Record outliers for explainability output."""
        self._outlier_records = []

        if "is_outlier" not in df.columns:
            return

        outliers = df[df["is_outlier"]]

        for _, row in outliers.iterrows():
            self._outlier_records.append({
                "month_key": row["month_key"],
                "original_value": row.get("residual_original", row.get("residual")),
                "treated_value": row.get("residual_clean", row.get("residual")),
                "detection_method": self.config.outlier_method,
                "score": row.get("outlier_score", 0),
                "treatment_tag": row.get("treatment_tag", "ABNORMAL_EXTERNAL_FLOW"),
            })

    def _train_and_select_model(
        self,
        df: pd.DataFrame,
        exog: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Train models and select the best one.

        Enhanced with exogenous variable support for SARIMAX.

        Args:
            df: Historical decomposed data
            exog: Optional exogenous variables for SARIMAX

        Returns:
            Tuple of (forecast_output, model_summary)
        """
        # Prepare residual series for modeling
        residual_col = "residual_clean" if "residual_clean" in df.columns else "residual"

        series = df.set_index("month_key")[residual_col]
        series.index = pd.to_datetime(series.index).to_period("M")

        # Split train/test
        train, test = split_train_test(series, self.config.test_size)

        # Split exog if available
        train_exog = None
        test_exog = None
        future_exog = None

        if exog is not None:
            # Align exog with series
            exog_aligned = exog.reindex(series.index).fillna(0.0)
            train_exog = exog_aligned.loc[train.index]
            test_exog = exog_aligned.loc[test.index]

            # Build future exog (zeros for now, could be enhanced with CRF)
            future_index = pd.period_range(
                start=series.index[-1] + 1,
                periods=self.config.forecast_horizon,
                freq="M"
            )
            future_exog = pd.DataFrame(
                {"known_delta": 0.0},
                index=future_index
            )

        # Initialize model selector
        self._model_selector = ModelSelector(
            wmape_threshold=self.config.wmape_threshold,
            tie_tolerance=self.config.model_tie_tolerance,
        )

        # Evaluate models
        if "ets" in self.config.models_to_evaluate:
            self._model_selector.evaluate_model(
                model=ETSModel(trend="add", seasonal="add"),
                train_series=train,
                test_series=test,
                forecast_steps=self.config.forecast_horizon,
            )

        if "sarima" in self.config.models_to_evaluate:
            self._model_selector.evaluate_model(
                model=SARIMAModel(
                    order=self.config.arima_order,
                    seasonal_order=self.config.seasonal_order,
                ),
                train_series=train,
                test_series=test,
                forecast_steps=self.config.forecast_horizon,
            )

        # Evaluate SARIMAX with exogenous variables if available
        if "sarimax" in self.config.models_to_evaluate and exog is not None:
            from cashflow.models.sarima import SARIMAXModel
            self._model_selector.evaluate_model(
                model=SARIMAXModel(
                    order=self.config.arima_order,
                    seasonal_order=self.config.seasonal_order,
                ),
                train_series=train,
                test_series=test,
                forecast_steps=self.config.forecast_horizon,
                train_exog=train_exog,
                test_exog=test_exog,
                future_exog=future_exog,
            )

        # Evaluate TiRex ONNX model if available
        if "tirex" in self.config.models_to_evaluate:
            try:
                from cashflow.models.tirex import TiRexModel
                self._model_selector.evaluate_model(
                    model=TiRexModel(),
                    train_series=train,
                    test_series=test,
                    forecast_steps=self.config.forecast_horizon,
                )
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning(f"TiRex model unavailable: {e}")

        winner = self._model_selector.select_winner()
        summary = self._model_selector.get_summary()

        return winner.forecast, summary

    def _recompose_forecast(
        self,
        forecast_output,
        historical_df: pd.DataFrame,
        crf_df: Optional[pd.DataFrame] = None,
    ) -> list[ForecastResult]:
        """Recompose forecast per SDD Section 14.1.

        Forecast Total = Forecast Residual + Deterministic Base + Known Future Adjustments

        Enhanced with trend-adjusted projection to fix the "Mean Fallacy":
        - Uses exponentially weighted recent values
        - Detects level shifts (salary raises, rent changes)
        - Projects forward with trend
        """
        # Compute trend-adjusted deterministic projection (fixes Mean Fallacy)
        self._deterministic_projection = compute_deterministic_projection(historical_df)

        logger.info(
            f"Deterministic projection: base={self._deterministic_projection.base_value:.2f}, "
            f"trend={self._deterministic_projection.monthly_trend:.2f}/month, "
            f"method={self._deterministic_projection.method}"
        )

        # Compute known future deltas from CRF
        from cashflow.pipeline.decomposition import compute_known_future_delta

        future_deltas = {}
        if crf_df is not None and len(crf_df) > 0:
            delta_df = compute_known_future_delta(
                crf_df,
                forecast_output.month_keys[0],
                forecast_output.month_keys[-1],
            )
            for _, row in delta_df.iterrows():
                future_deltas[row["month_key"]] = row["delta_value"]

        results = []
        for i, month_key in enumerate(forecast_output.month_keys):
            residual_forecast = forecast_output.forecast_mean[i]
            known_delta = future_deltas.get(month_key, 0.0)

            # Project deterministic base forward with trend (months_ahead = i + 1)
            projected_deterministic = self._deterministic_projection.project(i + 1)

            # Recomposition formula with trend-adjusted projection
            total = residual_forecast + projected_deterministic + known_delta

            # Adjust confidence intervals similarly
            lower = forecast_output.forecast_lower[i] + projected_deterministic + known_delta
            upper = forecast_output.forecast_upper[i] + projected_deterministic + known_delta

            results.append(
                ForecastResult(
                    month_key=month_key,
                    forecast_total=round(total, 2),
                    forecast_residual=round(residual_forecast, 2),
                    deterministic_base=round(projected_deterministic, 2),
                    known_future_delta=round(known_delta, 2),
                    lower_ci=round(lower, 2),
                    upper_ci=round(upper, 2),
                )
            )

        return results

    def _generate_explainability(
        self,
        forecast_results: list[ForecastResult],
        historical_df: pd.DataFrame,
        model_summary: dict,
    ) -> ExplainabilityPayload:
        """Generate the explainability JSON payload."""
        # Build model candidates list
        model_candidates = []
        for result in model_summary.get("all_results", []):
            model_candidates.append(
                ModelCandidate(
                    model_name=result["model"],
                    wmape=result["wmape"],
                    is_winner=(result["model"] == model_summary.get("winner")),
                )
            )

        # Build outlier records
        outlier_records = [
            OutlierRecord(**record) for record in self._outlier_records
        ]

        # Decomposition summary
        decomposition_summary = DecompositionSummary(
            avg_necf=round(historical_df["necf"].mean(), 2),
            avg_deterministic_base=round(historical_df["deterministic_base"].mean(), 2),
            avg_residual=round(historical_df["residual"].mean(), 2),
        )

        # Transfer netting summary
        transfer_summary = TransferNettingSummary(
            num_transfers_removed=self._transfer_summary.get("num_transfers_removed", 0),
            total_volume_removed=round(self._transfer_summary.get("total_volume_removed", 0), 2),
        )

        # Determine confidence level
        from cashflow.pipeline.cleaning import validate_data_quality

        confidence = determine_confidence_level(
            data_quality_score=95.0,  # Simplified
            month_count=len(historical_df),
            wmape=model_summary.get("winner_wmape", 100),
        )

        return ExplainabilityPayload(
            model_selected=model_summary.get("winner", "Unknown"),
            model_candidates=model_candidates,
            wmape_winner=round(model_summary.get("winner_wmape", 0), 3),
            meets_threshold=model_summary.get("meets_threshold", False),
            forecast_start=forecast_results[0].month_key,
            forecast_end=forecast_results[-1].month_key,
            horizon_months=len(forecast_results),
            confidence_level=confidence,
            decomposition_summary=decomposition_summary,
            transfer_netting_summary=transfer_summary,
            outliers_detected=outlier_records,
            forecast_results=forecast_results,
        )

    def run_from_dataframe(
        self,
        utf_df: pd.DataFrame,
        crf_df: Optional[pd.DataFrame] = None,
    ) -> ExplainabilityPayload:
        """Run forecast from DataFrames directly (for testing/API use).

        Enhanced pipeline with:
        - Layer 0.5: Internal recurrence detection
        - Trend-adjusted deterministic projection
        - Exogenous integration for SARIMAX
        """
        logger.info("Running forecast from DataFrames")

        # Clean UTF
        utf_df = clean_utf(utf_df)

        # CRF enrichment
        if crf_df is not None:
            utf_df = enrich_with_crf(utf_df, crf_df)

        # Transfer detection and netting
        utf_df = detect_transfers(utf_df, self.config.transfer_date_tolerance_days)
        external_df, self._transfer_summary = net_transfers(utf_df)

        # Layer 0.5: Internal recurrence detection (fixes SPOF on is_recurring_flag)
        patterns_df = discover_recurring_patterns(external_df)
        if len(patterns_df) > 0:
            external_df = apply_discovered_recurrence(external_df, patterns_df)
            self._recurrence_summary = get_recurrence_summary(external_df)
            logger.info(f"Recurrence discovery: {self._recurrence_summary}")
        else:
            self._recurrence_summary = {"newly_discovered": 0}

        # Monthly aggregation
        monthly_df = aggregate_monthly(external_df)

        # Decomposition (now uses discovered recurrence)
        decomposed_df = decompose_cashflow(monthly_df, external_df)

        # Outlier treatment
        treated_df = apply_residual_treatment(
            decomposed_df,
            detection_method=self.config.outlier_method,
            detection_threshold=self.config.outlier_threshold,
            treatment_method=self.config.outlier_treatment,
        )

        self._record_outliers(treated_df)

        # Build exogenous matrix for SARIMAX (if CRF available)
        self._exog_matrix = self._build_exog_matrix(treated_df, crf_df)

        # Model selection (with exogenous integration)
        forecast_output, model_summary = self._train_and_select_model(
            treated_df,
            exog=self._exog_matrix,
        )

        # Recomposition (with trend-adjusted projection)
        forecast_results = self._recompose_forecast(forecast_output, treated_df, crf_df)

        # Explainability
        return self._generate_explainability(forecast_results, treated_df, model_summary)

    def _build_exog_matrix(
        self,
        historical_df: pd.DataFrame,
        crf_df: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """Build exogenous variable matrix for SARIMAX.

        This implements proper exogenous integration so SARIMAX can learn
        the relationship between contract events and cash flow jumps.

        Args:
            historical_df: Historical decomposed data
            crf_df: Optional CRF data with contract information

        Returns:
            DataFrame with exogenous variables indexed by period
        """
        if crf_df is None or len(crf_df) == 0:
            return None

        from cashflow.pipeline.decomposition import compute_known_future_delta

        # Get month range from historical data
        months = historical_df["month_key"].unique()
        if len(months) < 2:
            return None

        start_month = min(months)
        end_month = max(months)

        # Compute known deltas for historical period
        delta_df = compute_known_future_delta(crf_df, start_month, end_month)

        if len(delta_df) == 0:
            return None

        # Create exogenous matrix
        exog = pd.DataFrame(index=pd.PeriodIndex(months, freq="M"))
        exog["known_delta"] = 0.0

        # Fill in known deltas
        for _, row in delta_df.iterrows():
            period = pd.Period(row["month_key"], freq="M")
            if period in exog.index:
                exog.loc[period, "known_delta"] = row["delta_value"]

        logger.info(f"Built exogenous matrix with {(exog['known_delta'] != 0).sum()} non-zero entries")

        return exog
