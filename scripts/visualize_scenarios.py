"""Multi-scenario visualization - comparing forecasts across different random conditions."""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import List, Dict

from cashflow.engine import ForecastEngine, ForecastConfig
from cashflow.schemas.forecast import ExplainabilityPayload

# Import data generator from forecast visualization
from visualize_forecast import generate_synthetic_data, run_forecast, COLORS, DPI

# Extended color palette for multiple scenarios
SCENARIO_COLORS = [
    "#2E86AB",  # Steel blue
    "#28A745",  # Green
    "#FFC107",  # Amber
    "#DC3545",  # Red
    "#6F42C1",  # Purple
    "#17A2B8",  # Cyan
    "#FD7E14",  # Orange
    "#20C997",  # Teal
]

FIG_SIZE = (14, 8)
FIG_SIZE_TALL = (14, 10)


def run_multiple_scenarios(seeds: List[int]) -> Dict[int, tuple]:
    """Run forecasts with multiple random seeds."""
    results = {}
    for seed in seeds:
        print(f"  Running scenario with seed={seed}...")
        utf_df = generate_synthetic_data(seed=seed)
        payload, historical_df = run_forecast(utf_df)
        results[seed] = (payload, historical_df)
    return results


def plot_scenario_comparison(
    scenarios: Dict[int, tuple],
    output_path: str,
) -> None:
    """Plot multiple forecast scenarios overlaid (spaghetti plot)."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    all_forecasts = []
    seeds = list(scenarios.keys())

    # Plot each scenario
    for i, (seed, (payload, historical_df)) in enumerate(scenarios.items()):
        color = SCENARIO_COLORS[i % len(SCENARIO_COLORS)]

        # Get forecast data
        forecast_months = pd.to_datetime([f"{fr.month_key}-01" for fr in payload.forecast_results])
        forecast_values = [fr.forecast_total for fr in payload.forecast_results]

        all_forecasts.append(forecast_values)

        # Plot individual scenario
        ax.plot(forecast_months, forecast_values,
                color=color, linewidth=1.5, alpha=0.7,
                label=f"Seed {seed} (WMAPE: {payload.wmape_winner:.2f}%)")

    # Calculate and plot mean forecast
    mean_forecast = np.mean(all_forecasts, axis=0)
    std_forecast = np.std(all_forecasts, axis=0)

    ax.plot(forecast_months, mean_forecast,
            color="#212529", linewidth=3, linestyle="-",
            label="Mean Forecast", zorder=10)

    # Confidence band (±1 std)
    ax.fill_between(forecast_months,
                    mean_forecast - std_forecast,
                    mean_forecast + std_forecast,
                    color="#21252920", edgecolor="#212529",
                    linewidth=1, label="±1 Std Dev")

    # Styling
    ax.set_xlabel("Forecast Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Net External Cash Flow (EUR)", fontsize=12, fontweight="bold")
    ax.set_title(f"Forecast Robustness: {len(seeds)} Different Data Scenarios\nShowing variance across random conditions",
                fontsize=14, fontweight="bold", pad=20)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha="right")

    ax.legend(loc="upper left", fontsize=9, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_wmape_distribution(
    scenarios: Dict[int, tuple],
    output_path: str,
) -> None:
    """Plot distribution of WMAPE scores across scenarios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    seeds = list(scenarios.keys())
    wmapes = [scenarios[s][0].wmape_winner for s in seeds]
    models = [scenarios[s][0].model_selected for s in seeds]

    # Bar chart of WMAPE by seed
    colors = [SCENARIO_COLORS[i % len(SCENARIO_COLORS)] for i in range(len(seeds))]
    bars = ax1.bar(range(len(seeds)), wmapes, color=colors, edgecolor="white", linewidth=1.5)

    ax1.set_xlabel("Scenario (Seed)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("WMAPE (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Model Accuracy Across Scenarios", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(len(seeds)))
    ax1.set_xticklabels([str(s) for s in seeds])

    # Add value labels
    for bar, wmape, model in zip(bars, wmapes, models):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{wmape:.2f}%\n({model})", ha="center", va="bottom", fontsize=9)

    ax1.axhline(y=20, color=COLORS["threshold"], linestyle="--",
                linewidth=2, label="20% Threshold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_facecolor("#FAFAFA")

    # Histogram of WMAPE distribution
    ax2.hist(wmapes, bins=10, color=COLORS["actual"], edgecolor="white",
             linewidth=1.5, alpha=0.7)
    ax2.axvline(x=np.mean(wmapes), color=COLORS["forecast"], linewidth=2.5,
                linestyle="-", label=f"Mean: {np.mean(wmapes):.2f}%")
    ax2.axvline(x=20, color=COLORS["threshold"], linewidth=2,
                linestyle="--", label="Threshold: 20%")

    ax2.set_xlabel("WMAPE (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax2.set_title("WMAPE Distribution", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_confidence_fan_chart(
    payload: ExplainabilityPayload,
    historical_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Plot fan chart showing expanding uncertainty over forecast horizon."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Historical data
    hist_months = pd.to_datetime(historical_df["month_key"].astype(str) + "-01")
    hist_values = historical_df["necf"].values

    # Forecast data
    forecast_months = pd.to_datetime([f"{fr.month_key}-01" for fr in payload.forecast_results])
    forecast_values = np.array([fr.forecast_total for fr in payload.forecast_results])
    forecast_lower = np.array([fr.lower_ci for fr in payload.forecast_results])
    forecast_upper = np.array([fr.upper_ci for fr in payload.forecast_results])

    # Calculate different CI levels (approximated from 95% CI)
    ci_width = forecast_upper - forecast_lower
    ci_90_lower = forecast_values - ci_width * 0.45
    ci_90_upper = forecast_values + ci_width * 0.45
    ci_99_lower = forecast_values - ci_width * 0.70
    ci_99_upper = forecast_values + ci_width * 0.70

    # Plot historical
    ax.plot(hist_months, hist_values,
            color=COLORS["actual"], linewidth=2.5, label="Historical",
            marker="o", markersize=4)

    # Plot forecast with expanding fan
    ax.fill_between(forecast_months, ci_99_lower, ci_99_upper,
                    color="#28A74515", label="99% CI")
    ax.fill_between(forecast_months, forecast_lower, forecast_upper,
                    color="#28A74530", label="95% CI")
    ax.fill_between(forecast_months, ci_90_lower, ci_90_upper,
                    color="#28A74550", label="90% CI")

    ax.plot(forecast_months, forecast_values,
            color=COLORS["forecast"], linewidth=2.5, label="Forecast",
            marker="s", markersize=5)

    # Forecast start line
    ax.axvline(x=forecast_months[0], color="#6C757D", linestyle="--",
               linewidth=1.5, alpha=0.7)

    # Styling
    ax.set_xlabel("Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Net External Cash Flow (EUR)", fontsize=12, fontweight="bold")
    ax.set_title("Forecast Uncertainty Fan Chart\nConfidence intervals expand with forecast horizon",
                fontsize=14, fontweight="bold", pad=20)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")

    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all scenario comparison plots."""
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    # Define scenarios with different random seeds
    seeds = [42, 123, 456, 789, 2024]

    print(f"Running {len(seeds)} forecast scenarios...")
    scenarios = run_multiple_scenarios(seeds)

    print("\nGenerating plots...")

    # Plot 1: Scenario comparison (spaghetti plot)
    plot_scenario_comparison(
        scenarios,
        str(output_dir / "scenario_comparison.png")
    )

    # Plot 2: WMAPE distribution
    plot_wmape_distribution(
        scenarios,
        str(output_dir / "wmape_distribution.png")
    )

    # Plot 3: Confidence fan chart (using first scenario)
    first_seed = seeds[0]
    payload, historical_df = scenarios[first_seed]
    plot_confidence_fan_chart(
        payload, historical_df,
        str(output_dir / "confidence_fan_chart.png")
    )

    print("\nDone! Scenario plots saved to:", output_dir)


if __name__ == "__main__":
    main()
