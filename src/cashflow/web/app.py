"""FastAPI application for cashflow forecasting web interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

if TYPE_CHECKING:
    from fastapi import Response

logger = logging.getLogger(__name__)

# Package paths
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Cash Flow Forecasting Engine",
        description="SDD v0.05 - Production-grade cash flow forecasting with interactive visualization",
        version="0.5.0",
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Register exception handlers
    _register_exception_handlers(app)

    # Include routers
    from cashflow.web.routes import forecast, pages

    app.include_router(pages.router, tags=["pages"])
    app.include_router(forecast.router, prefix="/api", tags=["forecast"])

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers."""

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(
        request: Request, exc: ValidationError
    ) -> Response:
        """Handle Pydantic validation errors with user-friendly messages."""
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Validation Error",
                "errors": exc.errors(),
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> Response:
        """Handle value errors from the forecast pipeline."""
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> Response:
        """Handle unexpected errors gracefully."""
        logger.exception("Unhandled error in request")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected error occurred. Please try again.",
                "error_type": type(exc).__name__,
            },
        )


# Create the app instance
app = create_app()


def main() -> None:
    """Run the web application with uvicorn."""
    import uvicorn

    uvicorn.run(
        "cashflow.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
