"""Health check endpoint for production monitoring.

Phase 4.3: Service health for orchestration systems.
Provides detailed health status for load balancers and monitoring.
"""

from __future__ import annotations

import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

# SDD version
SDD_VERSION = "0.05"
APP_VERSION = "0.5.0"


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: HealthStatus
    timestamp: str
    version: str
    sdd_version: str
    uptime_seconds: float
    checks: Dict[str, ComponentHealth]


# Track start time for uptime calculation
_start_time: float = time.time()


def _check_model_availability() -> ComponentHealth:
    """Check if forecasting models are available.

    Returns:
        ComponentHealth with model availability status
    """
    try:
        start = time.perf_counter()
        # Try importing model classes
        from cashflow.models import ETSModel, SARIMAModel, SARIMAXModel

        latency = (time.perf_counter() - start) * 1000

        return ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="All models available",
            latency_ms=round(latency, 2),
        )
    except ImportError as e:
        return ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=f"Model import failed: {str(e)}",
        )
    except Exception as e:
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message=f"Model check error: {str(e)}",
        )


def _check_pipeline_availability() -> ComponentHealth:
    """Check if pipeline components are available.

    Returns:
        ComponentHealth with pipeline availability status
    """
    try:
        start = time.perf_counter()
        # Try importing pipeline components
        from cashflow.pipeline import (
            clean_utf,
            detect_transfers,
            net_transfers,
            aggregate_monthly,
        )
        from cashflow.pipeline.decomposition import decompose_cashflow

        latency = (time.perf_counter() - start) * 1000

        return ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="Pipeline components available",
            latency_ms=round(latency, 2),
        )
    except ImportError as e:
        return ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=f"Pipeline import failed: {str(e)}",
        )
    except Exception as e:
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message=f"Pipeline check error: {str(e)}",
        )


def _check_engine_availability() -> ComponentHealth:
    """Check if forecast engine is available.

    Returns:
        ComponentHealth with engine availability status
    """
    try:
        start = time.perf_counter()
        # Try importing and instantiating engine
        from cashflow.engine import ForecastEngine, ForecastConfig

        # Just check that config can be created
        config = ForecastConfig()

        latency = (time.perf_counter() - start) * 1000

        return ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="Forecast engine available",
            latency_ms=round(latency, 2),
        )
    except ImportError as e:
        return ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=f"Engine import failed: {str(e)}",
        )
    except Exception as e:
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message=f"Engine check error: {str(e)}",
        )


def _check_data_validation() -> ComponentHealth:
    """Check if data validation is available.

    Returns:
        ComponentHealth with validation availability status
    """
    try:
        start = time.perf_counter()
        from cashflow.pipeline.validation import (
            DataQualityContract,
            DEFAULT_CONTRACT,
        )

        # Check that contract can be instantiated
        contract = DataQualityContract()

        latency = (time.perf_counter() - start) * 1000

        return ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="Data validation available",
            latency_ms=round(latency, 2),
        )
    except ImportError as e:
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message=f"Validation import failed: {str(e)}",
        )
    except Exception as e:
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message=f"Validation check error: {str(e)}",
        )


def _check_monitoring() -> ComponentHealth:
    """Check if monitoring components are available.

    Returns:
        ComponentHealth with monitoring availability status
    """
    try:
        start = time.perf_counter()
        from loguru import logger  # noqa: F811

        latency = (time.perf_counter() - start) * 1000

        return ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="Loguru monitoring available",
            latency_ms=round(latency, 2),
        )
    except ImportError as e:
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message=f"Monitoring import failed: {str(e)}",
        )
    except Exception as e:
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message=f"Monitoring check error: {str(e)}",
        )


def _determine_overall_status(checks: Dict[str, ComponentHealth]) -> HealthStatus:
    """Determine overall health status from component checks.

    Args:
        checks: Dictionary of component health statuses

    Returns:
        Overall health status
    """
    statuses = [c.status for c in checks.values()]

    if any(s == HealthStatus.UNHEALTHY for s in statuses):
        return HealthStatus.UNHEALTHY
    elif any(s == HealthStatus.DEGRADED for s in statuses):
        return HealthStatus.DEGRADED
    else:
        return HealthStatus.HEALTHY


# Create router
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health status for orchestration systems.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"},
    },
)
async def health_check(response: Response) -> HealthResponse:
    """Health check endpoint.

    Performs component checks and returns overall health status.
    Returns 200 for healthy/degraded, 503 for unhealthy.
    """
    # Run all health checks
    checks = {
        "models": _check_model_availability(),
        "pipeline": _check_pipeline_availability(),
        "engine": _check_engine_availability(),
        "validation": _check_data_validation(),
        "monitoring": _check_monitoring(),
    }

    # Determine overall status
    overall_status = _determine_overall_status(checks)

    # Set HTTP status code
    if overall_status == HealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    # Calculate uptime
    uptime = time.time() - _start_time

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        version=APP_VERSION,
        sdd_version=SDD_VERSION,
        uptime_seconds=round(uptime, 2),
        checks={k: v for k, v in checks.items()},
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Simple liveness check for Kubernetes.",
    responses={
        200: {"description": "Service is alive"},
    },
)
async def liveness_probe() -> Dict[str, str]:
    """Kubernetes liveness probe.

    Simple endpoint that returns 200 if the service is running.
    No dependency checks - just confirms the process is alive.
    """
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Readiness check for Kubernetes.",
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
)
async def readiness_probe(response: Response) -> Dict[str, Any]:
    """Kubernetes readiness probe.

    Checks if the service is ready to accept traffic.
    Verifies critical components (models, engine) are available.
    """
    # Check critical components only
    critical_checks = {
        "models": _check_model_availability(),
        "engine": _check_engine_availability(),
    }

    # Service is ready only if critical components are healthy
    is_ready = all(
        c.status == HealthStatus.HEALTHY
        for c in critical_checks.values()
    )

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "ready": is_ready,
        "checks": {k: v.model_dump() for k, v in critical_checks.items()},
    }
