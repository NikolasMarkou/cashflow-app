"""Structured logging for production deployment.

Phase 4.1: Production-grade log output for monitoring systems.
Supports both text and JSON output formats compatible with ELK/Splunk.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Union
from enum import Enum


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


class JsonFormatter(logging.Formatter):
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
        super().__init__()
        self.correlation_fields = correlation_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "message": record.getMessage(),
        }

        # Add correlation fields
        for key, value in self.correlation_fields.items():
            log_entry[key] = value

        # Add extra fields from record
        if hasattr(record, "customer_id"):
            log_entry["customer_id"] = record.customer_id
        if hasattr(record, "forecast_id"):
            log_entry["forecast_id"] = record.forecast_id
        if hasattr(record, "account_id"):
            log_entry["account_id"] = record.account_id

        # Add metrics if present
        if hasattr(record, "metrics") and record.metrics:
            log_entry["metrics"] = record.metrics

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any additional context
        if hasattr(record, "context") and record.context:
            log_entry["context"] = record.context

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
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
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_component = include_component
        self.correlation_fields = correlation_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as human-readable text."""
        parts = []

        if self.include_timestamp:
            parts.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        parts.append(f"[{record.levelname}]")

        if self.include_component:
            component = getattr(record, "component", record.name)
            parts.append(f"[{component}]")

        parts.append(record.getMessage())

        # Add correlation info
        extra_parts = []
        for key, value in self.correlation_fields.items():
            extra_parts.append(f"{key}={value}")

        if hasattr(record, "customer_id"):
            extra_parts.append(f"customer_id={record.customer_id}")
        if hasattr(record, "forecast_id"):
            extra_parts.append(f"forecast_id={record.forecast_id}")

        if extra_parts:
            parts.append(f"({', '.join(extra_parts)})")

        # Add metrics summary
        if hasattr(record, "metrics") and record.metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in record.metrics.items())
            parts.append(f"[metrics: {metrics_str}]")

        result = " ".join(parts)

        # Add exception info
        if record.exc_info:
            result += "\n" + self.formatException(record.exc_info)

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
        self._logger = logging.getLogger(f"cashflow.{name}")
        self._context: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}

        # Set level
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        self._logger.setLevel(level_map[self.config.level])

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
        level: int,
        msg: str,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Internal logging method."""
        # Create custom record with extra fields
        extra = {
            "component": self.name,
            "context": {**self._context, **kwargs},
            "metrics": self._metrics,
        }

        # Add common correlation fields
        for key in ["customer_id", "forecast_id", "account_id"]:
            if key in kwargs:
                extra[key] = kwargs[key]
            elif key in self._context:
                extra[key] = self._context[key]

        self._logger.log(level, msg, exc_info=exc_info, extra=extra)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, exc_info: Any = None, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, msg, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, exc_info: Any = None, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, msg, exc_info=exc_info, **kwargs)

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

    # Set up root logger
    root_logger = logging.getLogger("cashflow")
    root_logger.handlers.clear()

    # Create handler with appropriate formatter
    handler = logging.StreamHandler(config.stream)

    if config.format == LogFormat.JSON:
        formatter = JsonFormatter(config.correlation_fields)
    else:
        formatter = TextFormatter(
            include_timestamp=config.include_timestamp,
            include_component=config.include_component,
            correlation_fields=config.correlation_fields,
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Set level
    level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL,
    }
    root_logger.setLevel(level_map[config.level])


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
