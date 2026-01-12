"""Metrics collection and export for production monitoring.

Phase 4.2: Machine-readable metrics for dashboards and alerts.
Supports JSON serialization and Prometheus format stubs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager


@dataclass
class ForecastMetrics:
    """Metrics for a single forecast execution.

    Captures all key metrics from a forecast run for
    monitoring and alerting purposes.
    """
    # Identification
    forecast_id: str
    customer_id: str
    account_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Accuracy metrics
    wmape: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None

    # Execution metrics
    pipeline_duration_ms: float = 0.0
    model_fit_duration_ms: float = 0.0
    model_predict_duration_ms: float = 0.0

    # Data quality metrics
    data_quality_score: float = 0.0
    missing_rate: float = 0.0
    transactions_count: int = 0
    months_history: int = 0

    # Detection metrics
    outliers_detected_count: int = 0
    outliers_treated_count: int = 0
    transfers_netted_count: int = 0
    recurring_patterns_count: int = 0

    # Model selection
    selected_model: str = ""
    fallback_used: bool = False
    models_evaluated_count: int = 0

    # Confidence metrics
    confidence_score: float = 0.0
    confidence_level: str = ""
    ci_width_mean: float = 0.0

    # Forecast summary
    forecast_horizon_months: int = 12
    forecast_total: float = 0.0
    forecast_mean: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "identification": {
                "forecast_id": self.forecast_id,
                "customer_id": self.customer_id,
                "account_id": self.account_id,
                "timestamp": self.timestamp,
            },
            "accuracy": {
                "wmape": round(self.wmape, 2) if self.wmape is not None else None,
                "rmse": round(self.rmse, 2) if self.rmse is not None else None,
                "mape": round(self.mape, 2) if self.mape is not None else None,
            },
            "execution": {
                "pipeline_duration_ms": round(self.pipeline_duration_ms, 2),
                "model_fit_duration_ms": round(self.model_fit_duration_ms, 2),
                "model_predict_duration_ms": round(self.model_predict_duration_ms, 2),
            },
            "data_quality": {
                "score": round(self.data_quality_score, 1),
                "missing_rate": round(self.missing_rate, 4),
                "transactions_count": self.transactions_count,
                "months_history": self.months_history,
            },
            "detection": {
                "outliers_detected": self.outliers_detected_count,
                "outliers_treated": self.outliers_treated_count,
                "transfers_netted": self.transfers_netted_count,
                "recurring_patterns": self.recurring_patterns_count,
            },
            "model_selection": {
                "selected_model": self.selected_model,
                "fallback_used": self.fallback_used,
                "models_evaluated": self.models_evaluated_count,
            },
            "confidence": {
                "score": round(self.confidence_score, 1),
                "level": self.confidence_level,
                "ci_width_mean": round(self.ci_width_mean, 2),
            },
            "forecast": {
                "horizon_months": self.forecast_horizon_months,
                "total": round(self.forecast_total, 2),
                "mean": round(self.forecast_mean, 2),
            },
        }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus exposition format.

        Returns string in Prometheus format, ready for scraping.
        Labels include customer_id and forecast_id for cardinality.
        """
        lines = []
        labels = f'customer_id="{self.customer_id}",forecast_id="{self.forecast_id}"'

        # Accuracy metrics
        if self.wmape is not None:
            lines.append(f'cashflow_forecast_wmape{{{labels}}} {self.wmape:.4f}')
        if self.rmse is not None:
            lines.append(f'cashflow_forecast_rmse{{{labels}}} {self.rmse:.4f}')

        # Execution timing
        lines.append(f'cashflow_pipeline_duration_ms{{{labels}}} {self.pipeline_duration_ms:.2f}')
        lines.append(f'cashflow_model_fit_duration_ms{{{labels}}} {self.model_fit_duration_ms:.2f}')
        lines.append(f'cashflow_model_predict_duration_ms{{{labels}}} {self.model_predict_duration_ms:.2f}')

        # Data quality
        lines.append(f'cashflow_data_quality_score{{{labels}}} {self.data_quality_score:.1f}')
        lines.append(f'cashflow_transactions_count{{{labels}}} {self.transactions_count}')
        lines.append(f'cashflow_months_history{{{labels}}} {self.months_history}')

        # Detection counts
        lines.append(f'cashflow_outliers_detected{{{labels}}} {self.outliers_detected_count}')
        lines.append(f'cashflow_transfers_netted{{{labels}}} {self.transfers_netted_count}')
        lines.append(f'cashflow_recurring_patterns{{{labels}}} {self.recurring_patterns_count}')

        # Confidence
        lines.append(f'cashflow_confidence_score{{{labels}}} {self.confidence_score:.1f}')

        # Fallback indicator (1 = used fallback, 0 = primary model)
        lines.append(f'cashflow_fallback_used{{{labels}}} {1 if self.fallback_used else 0}')

        return "\n".join(lines)


class MetricsCollector:
    """Collector for aggregating metrics across multiple forecasts.

    Usage:
        collector = MetricsCollector()

        with collector.track_forecast("FC001", "CUST001") as metrics:
            # Run forecast pipeline
            metrics.wmape = 15.5
            metrics.selected_model = "ets"

        # Export all metrics
        print(collector.to_dict())
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._forecasts: List[ForecastMetrics] = []
        self._start_time: Optional[float] = None
        self._current_forecast: Optional[ForecastMetrics] = None

    @contextmanager
    def track_forecast(
        self,
        forecast_id: str,
        customer_id: str,
        account_id: Optional[str] = None,
    ):
        """Context manager for tracking a forecast execution.

        Args:
            forecast_id: Unique forecast identifier
            customer_id: Customer identifier
            account_id: Optional account identifier

        Yields:
            ForecastMetrics instance to populate
        """
        metrics = ForecastMetrics(
            forecast_id=forecast_id,
            customer_id=customer_id,
            account_id=account_id,
        )
        self._current_forecast = metrics
        start_time = time.perf_counter()

        try:
            yield metrics
        finally:
            # Record duration
            metrics.pipeline_duration_ms = (time.perf_counter() - start_time) * 1000
            self._forecasts.append(metrics)
            self._current_forecast = None

    @contextmanager
    def track_stage(self, stage_name: str):
        """Context manager for tracking a pipeline stage duration.

        Args:
            stage_name: Name of the pipeline stage

        Yields:
            Dict to store stage metrics
        """
        start_time = time.perf_counter()
        stage_metrics: Dict[str, Any] = {}

        try:
            yield stage_metrics
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            stage_metrics["duration_ms"] = duration_ms

            # Update current forecast if available
            if self._current_forecast is not None:
                if stage_name == "model_fit":
                    self._current_forecast.model_fit_duration_ms = duration_ms
                elif stage_name == "model_predict":
                    self._current_forecast.model_predict_duration_ms = duration_ms

    def record_forecast(self, metrics: ForecastMetrics) -> None:
        """Record a forecast metrics entry.

        Args:
            metrics: ForecastMetrics to record
        """
        self._forecasts.append(metrics)

    def get_forecasts(self) -> List[ForecastMetrics]:
        """Get all recorded forecast metrics."""
        return self._forecasts.copy()

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self._forecasts.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics as dictionary.

        Returns:
            Dictionary with aggregated metrics and individual forecasts
        """
        if not self._forecasts:
            return {
                "summary": {},
                "forecasts": [],
            }

        # Calculate aggregate statistics
        wmapes = [f.wmape for f in self._forecasts if f.wmape is not None]
        durations = [f.pipeline_duration_ms for f in self._forecasts]
        quality_scores = [f.data_quality_score for f in self._forecasts]
        confidence_scores = [f.confidence_score for f in self._forecasts]
        fallback_count = sum(1 for f in self._forecasts if f.fallback_used)

        summary = {
            "total_forecasts": len(self._forecasts),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        if wmapes:
            summary["wmape_mean"] = round(sum(wmapes) / len(wmapes), 2)
            summary["wmape_min"] = round(min(wmapes), 2)
            summary["wmape_max"] = round(max(wmapes), 2)
            summary["wmape_pass_rate"] = round(
                sum(1 for w in wmapes if w < 20) / len(wmapes), 4
            )

        if durations:
            summary["duration_ms_mean"] = round(sum(durations) / len(durations), 2)
            summary["duration_ms_p95"] = round(
                sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0], 2
            )

        if quality_scores:
            summary["data_quality_mean"] = round(sum(quality_scores) / len(quality_scores), 1)

        if confidence_scores:
            summary["confidence_mean"] = round(sum(confidence_scores) / len(confidence_scores), 1)

        summary["fallback_rate"] = round(fallback_count / len(self._forecasts), 4)

        return {
            "summary": summary,
            "forecasts": [f.to_dict() for f in self._forecasts],
        }

    def to_prometheus(self) -> str:
        """Export aggregate metrics in Prometheus format.

        Returns:
            String in Prometheus exposition format
        """
        lines = [
            "# HELP cashflow_forecasts_total Total number of forecasts processed",
            "# TYPE cashflow_forecasts_total counter",
            f"cashflow_forecasts_total {len(self._forecasts)}",
            "",
        ]

        if self._forecasts:
            wmapes = [f.wmape for f in self._forecasts if f.wmape is not None]
            if wmapes:
                lines.extend([
                    "# HELP cashflow_wmape_mean Mean WMAPE across all forecasts",
                    "# TYPE cashflow_wmape_mean gauge",
                    f"cashflow_wmape_mean {sum(wmapes) / len(wmapes):.4f}",
                    "",
                    "# HELP cashflow_wmape_pass_rate Proportion of forecasts with WMAPE < 20%",
                    "# TYPE cashflow_wmape_pass_rate gauge",
                    f"cashflow_wmape_pass_rate {sum(1 for w in wmapes if w < 20) / len(wmapes):.4f}",
                    "",
                ])

            durations = [f.pipeline_duration_ms for f in self._forecasts]
            lines.extend([
                "# HELP cashflow_pipeline_duration_ms_mean Mean pipeline duration",
                "# TYPE cashflow_pipeline_duration_ms_mean gauge",
                f"cashflow_pipeline_duration_ms_mean {sum(durations) / len(durations):.2f}",
                "",
            ])

            fallback_count = sum(1 for f in self._forecasts if f.fallback_used)
            lines.extend([
                "# HELP cashflow_fallback_rate Rate of fallback model usage",
                "# TYPE cashflow_fallback_rate gauge",
                f"cashflow_fallback_rate {fallback_count / len(self._forecasts):.4f}",
                "",
            ])

        # Add individual forecast metrics
        for forecast in self._forecasts:
            lines.append(forecast.to_prometheus())
            lines.append("")

        return "\n".join(lines)


# Singleton collector for global access
_global_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get the global metrics collector.

    Returns:
        Global MetricsCollector instance
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_collector() -> None:
    """Reset the global metrics collector."""
    global _global_collector
    _global_collector = MetricsCollector()
