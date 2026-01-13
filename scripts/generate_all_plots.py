#!/usr/bin/env python
"""Generate all visualization plots for client presentation."""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import csv
import numpy as np
from datetime import datetime


def collect_forecast_metrics(payload, historical_df) -> dict:
    """Extract key metrics from forecast payload."""
    # Model metrics
    metrics = {
        "model_selected": payload.model_selected,
        "wmape_winner": payload.wmape_winner,
        "wmape_threshold": payload.wmape_threshold,
        "confidence_level": payload.confidence_level,
    }

    # Add all model candidates
    for candidate in payload.model_candidates:
        metrics[f"wmape_{candidate.model_name}"] = candidate.wmape

    # Forecast statistics
    forecast_totals = [fr.forecast_total for fr in payload.forecast_results]
    forecast_lower = [fr.lower_ci for fr in payload.forecast_results]
    forecast_upper = [fr.upper_ci for fr in payload.forecast_results]

    metrics["forecast_mean"] = np.mean(forecast_totals)
    metrics["forecast_std"] = np.std(forecast_totals)
    metrics["forecast_min"] = np.min(forecast_totals)
    metrics["forecast_max"] = np.max(forecast_totals)

    # CI width
    ci_widths = [u - l for u, l in zip(forecast_upper, forecast_lower)]
    metrics["ci_width_mean"] = np.mean(ci_widths)
    metrics["ci_width_std"] = np.std(ci_widths)

    # Component breakdown (first month as example)
    if payload.forecast_results:
        fr = payload.forecast_results[0]
        metrics["deterministic_base_m1"] = fr.deterministic_base
        metrics["forecast_residual_m1"] = fr.forecast_residual
        metrics["known_future_delta_m1"] = fr.known_future_delta

    # Outliers
    metrics["outliers_detected"] = len(payload.outliers_detected) if payload.outliers_detected else 0

    # Historical data stats
    if historical_df is not None and "necf" in historical_df.columns:
        metrics["historical_mean"] = historical_df["necf"].mean()
        metrics["historical_std"] = historical_df["necf"].std()
        metrics["historical_months"] = len(historical_df)

    # Pass/fail
    metrics["passes_threshold"] = payload.wmape_winner < payload.wmape_threshold

    return metrics


def write_summary_csv(metrics: dict, output_path: Path) -> None:
    """Write metrics to CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            if isinstance(value, float):
                writer.writerow([key, f"{value:.4f}"])
            else:
                writer.writerow([key, value])
    print(f"Saved: {output_path}")


def main():
    """Run all visualization scripts and collect metrics."""
    print("=" * 60)
    print("CASH FLOW FORECASTING - VISUALIZATION SUITE")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    # First, run the forecast to collect metrics
    print("\n" + "-" * 60)
    print("RUNNING FORECAST FOR METRICS COLLECTION")
    print("-" * 60)
    from visualize_forecast import load_utf_data, run_forecast, DEFAULT_UTF_PATH

    print(f"Loading UTF data from: {DEFAULT_UTF_PATH}")
    utf_df = load_utf_data()
    print(f"Loaded {len(utf_df)} transactions")

    print("Running forecast...")
    payload, historical_df = run_forecast(utf_df)

    # Collect metrics
    metrics = collect_forecast_metrics(payload, historical_df)
    metrics["generated_at"] = datetime.now().isoformat()

    # Import and run each visualization module
    print("\n" + "-" * 60)
    print("1. FORECAST VISUALIZATIONS")
    print("-" * 60)
    from visualize_forecast import main as forecast_main
    forecast_main()

    print("\n" + "-" * 60)
    print("2. FORECAST ANALYSIS")
    print("-" * 60)
    from visualize_scenarios import main as scenarios_main
    scenarios_main()

    print("\n" + "-" * 60)
    print("3. DECOMPOSITION & OUTLIER ANALYSIS")
    print("-" * 60)
    from visualize_decomposition import main as decomposition_main
    decomposition_main()

    # Write summary CSV
    print("\n" + "-" * 60)
    print("4. GENERATING SUMMARY CSV")
    print("-" * 60)
    csv_path = output_dir / "forecast_summary.csv"
    write_summary_csv(metrics, csv_path)

    # Summary
    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 60)

    # List generated files
    plot_files = list(output_dir.glob("*.png"))
    csv_files = list(output_dir.glob("*.csv"))
    print(f"\nGenerated {len(plot_files)} plots and {len(csv_files)} CSV(s) in: {output_dir}")
    print()
    for f in sorted(plot_files):
        print(f"  - {f.name}")
    for f in sorted(csv_files):
        print(f"  - {f.name}")

    # Print key metrics
    print("\n" + "-" * 60)
    print("KEY METRICS SUMMARY")
    print("-" * 60)
    print(f"  Model Selected:    {metrics['model_selected']}")
    print(f"  WMAPE Winner:      {metrics['wmape_winner']:.2f}%")
    print(f"  Threshold:         {metrics['wmape_threshold']:.2f}%")
    print(f"  Passes Threshold:  {'Yes' if metrics['passes_threshold'] else 'No'}")
    print(f"  Forecast Mean:     {metrics['forecast_mean']:.2f} EUR")
    print(f"  CI Width Mean:     {metrics['ci_width_mean']:.2f} EUR")
    print(f"  Outliers Detected: {metrics['outliers_detected']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
