"""Command-line interface for the Cash Flow Forecasting Engine."""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from cashflow import __version__
from cashflow.engine import ForecastConfig, ForecastEngine
from cashflow.explainability import save_explainability_json


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(version=__version__)
def cli():
    """Cash Flow Forecasting Predictive Engine - SDD v0.05

    A production-grade cash flow forecasting system with layered architecture,
    transfer netting, and LLM-ready explainability output.
    """
    pass


@cli.command()
@click.option(
    "--utf",
    required=True,
    type=click.Path(exists=True),
    help="Path to UTF (Unified Transaction Feed) CSV file",
)
@click.option(
    "--crf",
    type=click.Path(exists=True),
    help="Optional path to CRF (Counterparty Reference Feed) CSV file",
)
@click.option(
    "--output",
    "-o",
    default="./output",
    type=click.Path(),
    help="Output directory for results",
)
@click.option(
    "--customer-id",
    help="Filter to specific customer ID",
)
@click.option(
    "--horizon",
    default=12,
    type=int,
    help="Forecast horizon in months (default: 12)",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration JSON file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def forecast(
    utf: str,
    crf: Optional[str],
    output: str,
    customer_id: Optional[str],
    horizon: int,
    config: Optional[str],
    verbose: bool,
):
    """Run cash flow forecast.

    Executes the full forecasting pipeline:
    1. Load and clean UTF data
    2. Enrich with CRF (if provided)
    3. Detect and net internal transfers
    4. Decompose into deterministic and residual
    5. Detect and treat outliers
    6. Train and select best model
    7. Generate forecast with confidence intervals
    8. Output explainability JSON
    """
    setup_logging(verbose)
    logger = logging.getLogger("cashflow.cli")

    try:
        # Load or create configuration
        if config:
            from cashflow.engine.config import load_config

            forecast_config = load_config(config)
        else:
            forecast_config = ForecastConfig(forecast_horizon=horizon)

        # Initialize engine
        engine = ForecastEngine(config=forecast_config)

        # Run forecast
        logger.info(f"Starting forecast for UTF: {utf}")
        result = engine.run(utf, crf, customer_id)

        # Create output directory
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        json_path = output_dir / "forecast_summary.json"
        save_explainability_json(result, json_path)

        # Also save CSV of forecast
        import pandas as pd

        forecast_df = pd.DataFrame([r.model_dump() for r in result.forecast_results])
        csv_path = output_dir / "forecast_results.csv"
        forecast_df.to_csv(csv_path, index=False)

        # Print summary
        click.echo(f"\n{'='*60}")
        click.echo("FORECAST COMPLETE")
        click.echo(f"{'='*60}")
        click.echo(f"Model Selected: {result.model_selected}")
        click.echo(f"WMAPE: {result.wmape_winner:.3f}%")
        click.echo(f"Meets Threshold: {'Yes' if result.meets_threshold else 'No'}")
        click.echo(f"Confidence: {result.confidence_level}")
        click.echo(f"Forecast Period: {result.forecast_start} to {result.forecast_end}")
        click.echo(f"Outliers Detected: {len(result.outliers_detected)}")
        click.echo(f"Transfers Netted: {result.transfer_netting_summary.num_transfers_removed}")
        click.echo(f"\nOutput saved to: {output_dir}")
        click.echo(f"  - {json_path.name}")
        click.echo(f"  - {csv_path.name}")

    except Exception as e:
        logger.exception("Forecast failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--utf",
    required=True,
    type=click.Path(exists=True),
    help="Path to UTF CSV file to validate",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed validation results",
)
def validate(utf: str, verbose: bool):
    """Validate UTF data against schema.

    Checks for:
    - Required columns
    - Valid data types
    - Missing values
    - Duplicate transactions
    """
    setup_logging(verbose)
    logger = logging.getLogger("cashflow.cli")

    from cashflow.pipeline import load_utf, clean_utf
    from cashflow.pipeline.cleaning import validate_data_quality

    try:
        # Load raw data
        df = load_utf(utf)
        click.echo(f"Loaded {len(df)} raw records from {utf}")

        # Clean and validate
        cleaned = clean_utf(df)
        click.echo(f"After cleaning: {len(cleaned)} valid records")

        # Quality report
        quality = validate_data_quality(cleaned)

        click.echo(f"\nData Quality Report:")
        click.echo(f"  Total rows: {quality['total_rows']}")
        click.echo(f"  Unique customers: {quality['unique_customers']}")
        click.echo(f"  Unique accounts: {quality['unique_accounts']}")

        if quality["date_range"]:
            click.echo(
                f"  Date range: {quality['date_range']['min']} to {quality['date_range']['max']}"
            )

        click.echo(f"  Quality score: {quality['data_quality_score']:.1f}%")

        if quality["missing_values"] and verbose:
            click.echo("\n  Missing values by column:")
            for col, count in quality["missing_values"].items():
                click.echo(f"    - {col}: {count}")

        if quality["data_quality_score"] >= 95:
            click.echo("\n✓ Data quality is excellent")
        elif quality["data_quality_score"] >= 80:
            click.echo("\n⚠ Data quality is acceptable but could be improved")
        else:
            click.echo("\n✗ Data quality issues detected")

    except Exception as e:
        logger.exception("Validation failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--output",
    "-o",
    default="./config.json",
    type=click.Path(),
    help="Output path for configuration file",
)
def init_config(output: str):
    """Generate a default configuration file.

    Creates a JSON configuration file with all default settings
    that can be customized for your use case.
    """
    config = ForecastConfig()
    output_path = Path(output)

    with open(output_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)

    click.echo(f"Default configuration written to: {output_path}")
    click.echo("\nEdit this file to customize forecast settings.")


if __name__ == "__main__":
    cli()
