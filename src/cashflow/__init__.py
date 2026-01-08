"""Cash Flow Forecasting Predictive Engine - SDD v0.05."""

from __future__ import annotations
__version__ = "0.5.0"

from cashflow.engine.forecast import ForecastEngine
from cashflow.engine.config import ForecastConfig

__all__ = ["ForecastEngine", "ForecastConfig", "__version__"]
