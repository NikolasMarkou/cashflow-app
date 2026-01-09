"""Forecast analysis visualization - confidence intervals and model performance."""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from cashflow.schemas.forecast import ExplainabilityPayload

# Import from forecast visualization
from visualize_forecast import load_utf_data, run_forecast, COLORS, DPI, DEFAULT_UTF_PATH

FIG_SIZE = (14, 8)


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


def plot_model_performance(
    payload: ExplainabilityPayload,
    output_path: str,
) -> None:
    """Plot detailed model performance comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = [c.model_name for c in payload.model_candidates]
    wmapes = [c.wmape for c in payload.model_candidates]
    is_winner = [c.is_winner for c in payload.model_candidates]

    # Bar chart - WMAPE comparison
    colors = [COLORS["winner"] if w else COLORS["loser"] for w in is_winner]
    bars = ax1.bar(models, wmapes, color=colors, edgecolor="white", linewidth=2, width=0.5)

    for bar, wmape, winner in zip(bars, wmapes, is_winner):
        height = bar.get_height()
        label = f"{wmape:.2f}%"
        if winner:
            label += "\n(Winner)"
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax1.axhline(y=payload.wmape_threshold, color=COLORS["threshold"],
                linewidth=2, linestyle="--", label=f"Threshold ({payload.wmape_threshold}%)")

    ax1.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax1.set_ylabel("WMAPE (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Model WMAPE Comparison\nLower is Better", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_facecolor("#FAFAFA")
    ax1.set_ylim(0, max(wmapes) * 1.5)

    # Gauge-style visualization for winner
    winner_wmape = payload.wmape_winner
    threshold = payload.wmape_threshold

    # Create a simple gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1

    # Background arc
    ax2.fill_between(theta, 0, r, alpha=0.1, color="#6C757D")

    # Threshold zone (red)
    threshold_angle = np.pi * (1 - threshold / 30)  # Scale to 30% max
    ax2.fill_between(theta[theta < threshold_angle], 0, r, alpha=0.3, color=COLORS["threshold"])

    # Good zone (green)
    ax2.fill_between(theta[theta >= threshold_angle], 0, r, alpha=0.3, color=COLORS["winner"])

    # Needle
    needle_angle = np.pi * (1 - winner_wmape / 30)
    ax2.annotate("", xy=(needle_angle, 0.9), xytext=(np.pi/2, 0),
                arrowprops=dict(arrowstyle="->", color="#212529", lw=3))

    # Labels
    ax2.text(np.pi/2, -0.15, f"{winner_wmape:.2f}%", ha="center", fontsize=16, fontweight="bold")
    ax2.text(np.pi/2, -0.35, f"Model: {payload.model_selected}", ha="center", fontsize=12)
    ax2.text(0, 0.1, "30%", ha="center", fontsize=10)
    ax2.text(np.pi, 0.1, "0%", ha="center", fontsize=10)

    ax2.set_xlim(-0.1, np.pi + 0.1)
    ax2.set_ylim(-0.5, 1.1)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Winner WMAPE Gauge", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_forecast_summary(
    payload: ExplainabilityPayload,
    output_path: str,
) -> None:
    """Plot summary statistics of the forecast."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract forecast data
    months = [fr.month_key for fr in payload.forecast_results]
    totals = [fr.forecast_total for fr in payload.forecast_results]
    lowers = [fr.lower_ci for fr in payload.forecast_results]
    uppers = [fr.upper_ci for fr in payload.forecast_results]

    x = np.arange(len(months))

    # Plot with error bars
    ax.errorbar(x, totals, yerr=[np.array(totals) - np.array(lowers),
                                  np.array(uppers) - np.array(totals)],
                fmt="o-", color=COLORS["forecast"], linewidth=2,
                markersize=8, capsize=5, capthick=2,
                label="Forecast ± 95% CI")

    # Add summary statistics
    avg_forecast = np.mean(totals)
    ax.axhline(y=avg_forecast, color=COLORS["actual"], linestyle="--",
               linewidth=2, label=f"Average: {avg_forecast:.2f} EUR")

    # Annotations
    ax.fill_between(x, lowers, uppers, alpha=0.2, color=COLORS["forecast"])

    # Styling
    ax.set_xlabel("Forecast Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Net External Cash Flow (EUR)", fontsize=12, fontweight="bold")
    ax.set_title(f"12-Month Forecast Summary\n{payload.model_selected} Model | WMAPE: {payload.wmape_winner:.2f}%",
                fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")

    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate forecast analysis plots from real data."""
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading UTF data from: {DEFAULT_UTF_PATH}")
    utf_df = load_utf_data()
    print(f"Loaded {len(utf_df)} transactions")

    print("Running forecast...")
    payload, historical_df = run_forecast(utf_df)

    print("\nGenerating plots...")

    # Plot 1: Confidence fan chart
    plot_confidence_fan_chart(
        payload, historical_df,
        str(output_dir / "confidence_fan_chart.png")
    )

    # Plot 2: Model performance comparison
    plot_model_performance(
        payload,
        str(output_dir / "model_performance.png")
    )

    # Plot 3: Forecast summary with error bars
    plot_forecast_summary(
        payload,
        str(output_dir / "forecast_summary.png")
    )

    print("\nDone! Analysis plots saved to:", output_dir)


if __name__ == "__main__":
    main()
