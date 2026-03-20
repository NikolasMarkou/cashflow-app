"""Structured logging for production deployment.

Phase 4.1: Production-grade log output for monitoring systems.
Supports both text and JSON output formats compatible with ELK/Splunk.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Union
from enum import Enum

from loguru import logger


class LogFormat(str, Enum):
    """Log output format options."""
    TEXT = "text"
    JSON = "json"


class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogConfig:
    """Configuration for structured logging.

    Attributes:
        format: Output format ("text" or "json")
        level: Minimum log level to output
        include_timestamp: Whether to include timestamp in output
        include_component: Whether to include component name
        stream: Output stream (default: sys.stderr)
        correlation_fields: Fields to include in every log entry for correlation
    """
    format: LogFormat = LogFormat.TEXT
    level: LogLevel = LogLevel.INFO
    include_timestamp: bool = True
    include_component: bool = True
    stream: Any = None  # Will default to sys.stderr
    correlation_fields: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stream is None:
            self.stream = sys.stderr
        # Convert string to enum if needed
        if isinstance(self.format, str):
            self.format = LogFormat(self.format)
        if isinstance(self.level, str):
            self.level = LogLevel(self.level)


class JsonFormatter:
    """JSON log formatter for structured logging.

    Outputs logs in JSON format with schema:
    {
        "timestamp": "2024-01-15T10:30:00.000Z",
        "level": "INFO",
        "component": "pipeline",
        "message": "Processing started",
        "customer_id": "CUST001",
        "forecast_id": "FC001",
        "metrics": {"duration_ms": 150}
    }
    """

    def __init__(self, correlation_fields: Optional[Dict[str, str]] = None) -> None:
        self.correlation_fields = correlation_fields or {}

    def format(self, record: Any) -> str:
        """Format log record as JSON."""
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.get("level", "INFO") if isinstance(record, dict) else str(getattr(record, "level", "INFO")),
            "component": record.get("component", "") if isinstance(record, dict) else getattr(record, "component", ""),
            "message": record.get("message", "") if isinstance(record, dict) else str(getattr(record, "message", "")),
        }

        # Add correlation fields
        for key, value in self.correlation_fields.items():
            log_entry[key] = value

        # Add extra fields from record
        extra = record if isinstance(record, dict) else {}
        for field_name in ["customer_id", "forecast_id", "account_id"]:
            if field_name in extra:
                log_entry[field_name] = extra[field_name]

        # Add metrics if present
        if "metrics" in extra and extra["metrics"]:
            log_entry["metrics"] = extra["metrics"]

        # Add exception info if present
        if "exception" in extra:
            log_entry["exception"] = extra["exception"]

        # Add any additional context
        if "context" in extra and extra["context"]:
            log_entry["context"] = extra["context"]

        return json.dumps(log_entry, default=str)


class TextFormatter:
    """Human-readable text formatter for development.

    Output format:
    2024-01-15 10:30:00 [INFO] [pipeline] Processing started (customer_id=CUST001)
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_component: bool = True,
        correlation_fields: Optional[Dict[str, str]] = None,
    ) -> None:
        self.include_timestamp = include_timestamp
        self.include_component = include_component
        self.correlation_fields = correlation_fields or {}

    def format(self, record: Any) -> str:
        """Format log record as human-readable text."""
        parts = []

        if self.include_timestamp:
            parts.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        level = record.get("level", "INFO") if isinstance(record, dict) else str(getattr(record, "level", "INFO"))
        parts.append(f"[{level}]")

        if self.include_component:
            component = record.get("component", "") if isinstance(record, dict) else getattr(record, "component", "")
            parts.append(f"[{component}]")

        message = record.get("message", "") if isinstance(record, dict) else str(getattr(record, "message", ""))
        parts.append(message)

        # Add correlation info
        extra_parts = []
        for key, value in self.correlation_fields.items():
            extra_parts.append(f"{key}={value}")

        extra = record if isinstance(record, dict) else {}
        if "customer_id" in extra:
            extra_parts.append(f"customer_id={extra['customer_id']}")
        if "forecast_id" in extra:
            extra_parts.append(f"forecast_id={extra['forecast_id']}")

        if extra_parts:
            parts.append(f"({', '.join(extra_parts)})")

        # Add metrics summary
        if "metrics" in extra and extra["metrics"]:
            metrics_str = ", ".join(f"{k}={v}" for k, v in extra["metrics"].items())
            parts.append(f"[metrics: {metrics_str}]")

        result = " ".join(parts)

        # Add exception info
        if "exception" in extra:
            result += "\n" + str(extra["exception"])

        return result


class StructuredLogger:
    """Structured logger with context support.

    Provides logging with automatic inclusion of correlation IDs
    and metrics. Supports both text and JSON output formats.

    Usage:
        logger = StructuredLogger("pipeline")
        logger.info("Processing started", customer_id="CUST001")
        logger.with_metrics({"duration_ms": 150}).info("Step completed")
    """

    def __init__(
        self,
        name: str,
        config: Optional[LogConfig] = None,
    ) -> None:
        """Initialize structured logger.

        Args:
            name: Logger name (component identifier)
            config: Log configuration
        """
        self.name = name
        self.config = config or LogConfig()
        self._logger = logger.bind(component=name)
        self._context: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}

    def with_context(self, **kwargs: Any) -> "StructuredLogger":
        """Return logger with additional context fields.

        Args:
            **kwargs: Context fields to include

        Returns:
            New logger instance with context
        """
        new_logger = StructuredLogger(self.name, self.config)
        new_logger._context = {**self._context, **kwargs}
        new_logger._metrics = self._metrics.copy()
        return new_logger

    def with_metrics(self, metrics: Dict[str, Any]) -> "StructuredLogger":
        """Return logger with metrics to include.

        Args:
            metrics: Metrics dictionary

        Returns:
            New logger instance with metrics
        """
        new_logger = StructuredLogger(self.name, self.config)
        new_logger._context = self._context.copy()
        new_logger._metrics = {**self._metrics, **metrics}
        return new_logger

    def _log(
        self,
        level: str,
        msg: str,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Internal logging method."""
        # Build bound logger with extra fields
        bound = self._logger.bind(
            context={**self._context, **kwargs},
            metrics=self._metrics,
        )

        # Add common correlation fields
        for key in ["customer_id", "forecast_id", "account_id"]:
            if key in kwargs:
                bound = bound.bind(**{key: kwargs[key]})
            elif key in self._context:
                bound = bound.bind(**{key: self._context[key]})

        if exc_info:
            bound.opt(exception=exc_info).log(level, msg)
        else:
            bound.log(level, msg)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log("DEBUG", msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log("INFO", msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log("WARNING", msg, **kwargs)

    def error(self, msg: str, exc_info: Any = None, **kwargs: Any) -> None:
        """Log error message."""
        self._log("ERROR", msg, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, exc_info: Any = None, **kwargs: Any) -> None:
        """Log critical message."""
        self._log("CRITICAL", msg, exc_info=exc_info, **kwargs)

    # Pipeline-specific logging methods
    def pipeline_start(self, customer_id: str, forecast_id: str) -> None:
        """Log pipeline start event."""
        self.info(
            "Pipeline started",
            customer_id=customer_id,
            forecast_id=forecast_id,
        )

    def pipeline_stage(
        self,
        stage: str,
        customer_id: str,
        forecast_id: str,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Log pipeline stage completion."""
        metrics = {}
        if duration_ms is not None:
            metrics["duration_ms"] = round(duration_ms, 2)

        self.with_metrics(metrics).info(
            f"Pipeline stage completed: {stage}",
            customer_id=customer_id,
            forecast_id=forecast_id,
        )

    def pipeline_complete(
        self,
        customer_id: str,
        forecast_id: str,
        total_duration_ms: float,
        wmape: Optional[float] = None,
    ) -> None:
        """Log pipeline completion."""
        metrics = {"total_duration_ms": round(total_duration_ms, 2)}
        if wmape is not None:
            metrics["wmape"] = round(wmape, 2)

        self.with_metrics(metrics).info(
            "Pipeline completed",
            customer_id=customer_id,
            forecast_id=forecast_id,
        )

    def model_selection(
        self,
        selected_model: str,
        wmape: float,
        candidates: Dict[str, float],
        customer_id: str,
        forecast_id: str,
        is_fallback: bool = False,
    ) -> None:
        """Log model selection decision."""
        metrics = {
            "wmape": round(wmape, 2),
            "num_candidates": len(candidates),
        }

        msg = f"Model selected: {selected_model}"
        if is_fallback:
            msg += " (fallback)"

        self.with_metrics(metrics).info(
            msg,
            customer_id=customer_id,
            forecast_id=forecast_id,
        )

    def threshold_violation(
        self,
        metric_name: str,
        actual: float,
        threshold: float,
        customer_id: str,
        forecast_id: str,
    ) -> None:
        """Log threshold violation warning."""
        self.warning(
            f"Threshold violation: {metric_name}={actual:.2f} exceeds {threshold:.2f}",
            customer_id=customer_id,
            forecast_id=forecast_id,
        )

    def data_quality_issue(
        self,
        issue_type: str,
        details: str,
        severity: str = "warning",
        customer_id: Optional[str] = None,
    ) -> None:
        """Log data quality issue."""
        kwargs = {}
        if customer_id:
            kwargs["customer_id"] = customer_id

        if severity == "error":
            self.error(f"Data quality issue ({issue_type}): {details}", **kwargs)
        else:
            self.warning(f"Data quality issue ({issue_type}): {details}", **kwargs)


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}
_global_config: Optional[LogConfig] = None


def configure_logging(config: LogConfig) -> None:
    """Configure global logging settings.

    Args:
        config: Log configuration to use
    """
    global _global_config
    _global_config = config

    # Remove existing handlers
    logger.remove()

    # Create formatter based on config
    if config.format == LogFormat.JSON:
        formatter = JsonFormatter(config.correlation_fields)
        logger.add(
            config.stream,
            level=config.level.value,
            format=lambda record: formatter.format(record) + "\n",
            serialize=False,
        )
    else:
        fmt = ""
        if config.include_timestamp:
            fmt += "{time:YYYY-MM-DD HH:mm:ss} "
        fmt += "[{level}]"
        if config.include_component:
            fmt += " [{extra[component]}]" if "component" in "{extra}" else " [{name}]"
        fmt += " {message}"
        logger.add(
            config.stream,
            level=config.level.value,
            format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {name}: {message}",
        )


def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger.

    Args:
        name: Logger name (component identifier)

    Returns:
        StructuredLogger instance
    """
    global _loggers, _global_config

    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, _global_config)

    return _loggers[name]


def configure_json_logging() -> None:
    """Convenience function to configure JSON logging for production."""
    configure_logging(LogConfig(format=LogFormat.JSON))


def configure_text_logging() -> None:
    """Convenience function to configure text logging for development."""
    configure_logging(LogConfig(format=LogFormat.TEXT))
