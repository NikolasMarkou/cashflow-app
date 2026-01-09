"""Page routes for serving HTML templates."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

# Templates directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the main forecast page."""

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "outlier_methods": [
                {"value": "modified_zscore", "label": "Modified Z-Score (MAD)"},
                {"value": "iqr", "label": "Interquartile Range (IQR)"},
                {"value": "zscore", "label": "Standard Z-Score"},
                {"value": "isolation_forest", "label": "Isolation Forest"},
            ],
            "outlier_treatments": [
                {"value": "median", "label": "Median Replacement"},
                {"value": "rolling_median", "label": "Rolling Median"},
                {"value": "capped", "label": "Capped (Percentile)"},
            ],
            "models": [
                {"value": "ets", "label": "ETS (Holt-Winters)", "default": True},
                {"value": "sarima", "label": "SARIMA", "default": True},
                {"value": "sarimax", "label": "SARIMAX (with exogenous)", "default": False},
            ],
            "confidence_levels": [
                {"value": "0.90", "label": "90%"},
                {"value": "0.95", "label": "95%"},
                {"value": "0.99", "label": "99%"},
            ],
        },
    )
