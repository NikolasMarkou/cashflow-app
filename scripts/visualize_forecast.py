"""Main forecast visualization - time series, components, and model comparison."""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from cashflow.engine import ForecastEngine, ForecastConfig
from cashflow.schemas.forecast import ExplainabilityPayload

# Professional color palette
COLORS = {
    "actual": "#2E86AB",        # Steel blue
    "forecast": "#28A745",      # Green
    "ci_fill": "#28A74540",     # Green transparent
    "ci_edge": "#28A745",       # Green
    "outlier": "#DC3545",       # Red
    "deterministic": "#17A2B8", # Cyan
    "residual": "#FFC107",      # Amber
    "delta": "#6F42C1",         # Purple
    "threshold": "#DC3545",     # Red
    "winner": "#28A745",        # Green
    "loser": "#6C757D",         # Gray
}

# Plot settings
FIG_SIZE = (14, 7)
FIG_SIZE_WIDE = (16, 6)
DPI = 150


def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic UTF transaction data with controlled patterns."""
    np.random.seed(seed)

    transactions = []
    tx_id = 1

    # Generate 24 months of data
    for year in [2024, 2025]:
        for month in range(1, 13):
            if year == 2025 and month > 12:
                break

            month_start = datetime(year, month, 1)

            # Salary (recurring income) - 1st of month
            salary = 3000 + np.random.normal(0, 50)
            transactions.append({
                "tx_id": f"TX{tx_id:06d}",
                "customer_id": "CUST001",
                "account_id": "MAIN_CHECKING",
                "tx_date": month_start,
                "amount": salary,
                "currency": "EUR",
                "direction": "CREDIT",
                "category": "SALARY",
                "description_raw": f"SALARY {year}-{month:02d}",
                "is_recurring_flag": True,
                "is_variable_amount": False,
            })
            tx_id += 1

            # Rent (recurring expense) - 1st of month
            transactions.append({
                "tx_id": f"TX{tx_id:06d}",
                "customer_id": "CUST001",
                "account_id": "MAIN_CHECKING",
                "tx_date": month_start,
                "amount": -1200,
                "currency": "EUR",
                "direction": "DEBIT",
                "category": "RENT_MORTGAGE",
                "description_raw": f"RENT {year}-{month:02d}",
                "is_recurring_flag": True,
                "is_variable_amount": False,
            })
            tx_id += 1

            # Utilities (seasonal variation)
            winter_factor = 1.5 if month in [11, 12, 1, 2] else 1.0
            utilities = -(120 * winter_factor + np.random.normal(0, 20))
            transactions.append({
                "tx_id": f"TX{tx_id:06d}",
                "customer_id": "CUST001",
                "account_id": "MAIN_CHECKING",
                "tx_date": datetime(year, month, 5),
                "amount": utilities,
                "currency": "EUR",
                "direction": "DEBIT",
                "category": "UTILITIES",
                "description_raw": f"UTILITIES {year}-{month:02d}",
                "is_recurring_flag": True,
                "is_variable_amount": True,
            })
            tx_id += 1

            # Groceries (variable, multiple per month)
            for _ in range(np.random.randint(3, 6)):
                day = np.random.randint(1, 28)
                amount = -(80 + np.random.exponential(40))
                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN_CHECKING",
                    "tx_date": datetime(year, month, day),
                    "amount": amount,
                    "currency": "EUR",
                    "direction": "DEBIT",
                    "category": "GROCERIES",
                    "description_raw": f"GROCERIES PURCHASE",
                    "is_recurring_flag": False,
                    "is_variable_amount": True,
                })
                tx_id += 1

            # Internal transfer (to be netted)
            transactions.append({
                "tx_id": f"TX{tx_id:06d}",
                "customer_id": "CUST001",
                "account_id": "MAIN_CHECKING",
                "tx_date": datetime(year, month, 15),
                "amount": -500,
                "currency": "EUR",
                "direction": "DEBIT",
                "category": "TRANSFER_OUT",
                "description_raw": "SAVINGS TRANSFER",
                "is_recurring_flag": True,
                "is_variable_amount": False,
            })
            tx_id += 1

            transactions.append({
                "tx_id": f"TX{tx_id:06d}",
                "customer_id": "CUST001",
                "account_id": "SAVINGS",
                "tx_date": datetime(year, month, 15),
                "amount": 500,
                "currency": "EUR",
                "direction": "CREDIT",
                "category": "TRANSFER_IN",
                "description_raw": "SAVINGS TRANSFER",
                "is_recurring_flag": True,
                "is_variable_amount": False,
            })
            tx_id += 1

            # Outliers (anomalies)
            # Tax refund in August 2024
            if year == 2024 and month == 8:
                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN_CHECKING",
                    "tx_date": datetime(year, month, 15),
                    "amount": 5000,
                    "currency": "EUR",
                    "direction": "CREDIT",
                    "category": "TAX_REFUND",
                    "description_raw": "TAX REFUND",
                    "is_recurring_flag": False,
                    "is_variable_amount": False,
                })
                tx_id += 1

            # Vacation expenses in July
            if month == 7:
                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN_CHECKING",
                    "tx_date": datetime(year, month, 20),
                    "amount": -1800,
                    "currency": "EUR",
                    "direction": "DEBIT",
                    "category": "TRAVEL",
                    "description_raw": "VACATION EXPENSES",
                    "is_recurring_flag": False,
                    "is_variable_amount": False,
                })
                tx_id += 1

    df = pd.DataFrame(transactions)
    df["tx_date"] = pd.to_datetime(df["tx_date"])
    return df


def run_forecast(utf_df: pd.DataFrame) -> tuple[ExplainabilityPayload, pd.DataFrame]:
    """Run forecast and return payload plus historical monthly data."""
    from cashflow.pipeline import clean_utf, detect_transfers, net_transfers, aggregate_monthly
    from cashflow.pipeline.decomposition import decompose_cashflow

    # Clean data
    utf_df = clean_utf(utf_df)

    # Detect and net transfers
    utf_df = detect_transfers(utf_df, date_tolerance_days=2)
    external_df, _ = net_transfers(utf_df)

    # Aggregate monthly
    monthly_df = aggregate_monthly(external_df)

    # Get decomposed for historical reference
    decomposed_df = decompose_cashflow(monthly_df, external_df)

    # Run full engine
    engine = ForecastEngine(ForecastConfig())
    payload = engine.run_from_dataframe(utf_df)

    return payload, decomposed_df


def plot_forecast_timeseries(
    historical_df: pd.DataFrame,
    payload: ExplainabilityPayload,
    output_path: str,
    show_outliers: bool = True,
) -> None:
    """Plot historical data with forecast and confidence intervals."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Prepare historical data
    hist_months = pd.to_datetime(historical_df["month_key"].astype(str) + "-01")
    hist_values = historical_df["necf"].values

    # Prepare forecast data
    forecast_months = pd.to_datetime([f"{fr.month_key}-01" for fr in payload.forecast_results])
    forecast_values = [fr.forecast_total for fr in payload.forecast_results]
    forecast_lower = [fr.lower_ci for fr in payload.forecast_results]
    forecast_upper = [fr.upper_ci for fr in payload.forecast_results]

    # Plot historical
    ax.plot(hist_months, hist_values,
            color=COLORS["actual"], linewidth=2.5, label="Historical (Actual)",
            marker="o", markersize=5)

    # Plot forecast
    ax.plot(forecast_months, forecast_values,
            color=COLORS["forecast"], linewidth=2.5, label="Forecast",
            marker="s", markersize=6)

    # Confidence interval
    ax.fill_between(forecast_months, forecast_lower, forecast_upper,
                    color=COLORS["ci_fill"], edgecolor=COLORS["ci_edge"],
                    linewidth=1, alpha=0.4, label="95% Confidence Interval")

    # Vertical line at forecast start
    forecast_start = forecast_months[0]
    ax.axvline(x=forecast_start, color="#6C757D", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(forecast_start, ax.get_ylim()[1] * 0.95, "  Forecast Start",
            fontsize=10, color="#6C757D", va="top")

    # Mark outliers
    if show_outliers and payload.outliers_detected:
        outlier_months = [o.month_key for o in payload.outliers_detected]
        for om in outlier_months:
            om_dt = pd.to_datetime(f"{om}-01")
            if om_dt in hist_months.values:
                idx = hist_months.tolist().index(om_dt)
                ax.scatter([om_dt], [hist_values[idx]],
                          color=COLORS["outlier"], s=150, zorder=5,
                          marker="o", edgecolors="white", linewidths=2)
        ax.scatter([], [], color=COLORS["outlier"], s=100, label="Detected Outliers",
                  marker="o", edgecolors="white", linewidths=2)

    # Styling
    ax.set_xlabel("Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Net External Cash Flow (EUR)", fontsize=12, fontweight="bold")
    ax.set_title(f"Cash Flow Forecast - 12 Month Horizon\nModel: {payload.model_selected} | WMAPE: {payload.wmape_winner:.2f}%",
                fontsize=14, fontweight="bold", pad=20)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")

    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add subtle background
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_component_breakdown(
    payload: ExplainabilityPayload,
    output_path: str,
) -> None:
    """Plot stacked bar chart of forecast components."""
    fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)

    months = [fr.month_key for fr in payload.forecast_results]
    deterministic = [fr.deterministic_base for fr in payload.forecast_results]
    residual = [fr.forecast_residual for fr in payload.forecast_results]
    delta = [fr.known_future_delta for fr in payload.forecast_results]
    total = [fr.forecast_total for fr in payload.forecast_results]

    x = np.arange(len(months))
    width = 0.6

    # Stacked bars
    bars1 = ax.bar(x, deterministic, width, label="Deterministic Base",
                   color=COLORS["deterministic"], edgecolor="white", linewidth=0.5)

    # Handle negative residuals
    residual_pos = [max(0, r) for r in residual]
    residual_neg = [min(0, r) for r in residual]

    bars2 = ax.bar(x, residual_pos, width, bottom=deterministic,
                   label="Residual (Positive)", color=COLORS["residual"],
                   edgecolor="white", linewidth=0.5)

    bars3 = ax.bar(x, residual_neg, width, bottom=deterministic,
                   label="Residual (Negative)", color="#E57373",
                   edgecolor="white", linewidth=0.5)

    # Total line
    ax.plot(x, total, color="#212529", linewidth=2.5, marker="D",
            markersize=6, label="Forecast Total", zorder=5)

    # Styling
    ax.set_xlabel("Forecast Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cash Flow (EUR)", fontsize=12, fontweight="bold")
    ax.set_title("Forecast Component Breakdown\nTotal = Deterministic Base + Residual + Known Future Delta",
                fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")

    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="#6C757D", linewidth=1, linestyle="-")
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison(
    payload: ExplainabilityPayload,
    output_path: str,
) -> None:
    """Plot bar chart comparing model WMAPE scores."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [c.model_name for c in payload.model_candidates]
    wmapes = [c.wmape for c in payload.model_candidates]
    is_winner = [c.is_winner for c in payload.model_candidates]

    colors = [COLORS["winner"] if w else COLORS["loser"] for w in is_winner]

    x = np.arange(len(models))
    bars = ax.bar(x, wmapes, color=colors, edgecolor="white", linewidth=2, width=0.5)

    # Add value labels
    for bar, wmape, winner in zip(bars, wmapes, is_winner):
        height = bar.get_height()
        label = f"{wmape:.2f}%"
        if winner:
            label += "\n(Winner)"
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Threshold line
    threshold = payload.wmape_threshold
    ax.axhline(y=threshold, color=COLORS["threshold"], linewidth=2,
               linestyle="--", label=f"Threshold ({threshold}%)")

    # Styling
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("WMAPE (%)", fontsize=12, fontweight="bold")
    ax.set_title("Model Selection: WMAPE Comparison\nLower is Better",
                fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, max(wmapes) * 1.3)

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all main forecast plots."""
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    print("Generating synthetic data...")
    utf_df = generate_synthetic_data(seed=42)

    print("Running forecast...")
    payload, historical_df = run_forecast(utf_df)

    print("\nGenerating plots...")

    # Plot 1: Time series
    plot_forecast_timeseries(
        historical_df, payload,
        str(output_dir / "forecast_timeseries.png")
    )

    # Plot 2: Components
    plot_component_breakdown(
        payload,
        str(output_dir / "forecast_components.png")
    )

    # Plot 3: Model comparison
    plot_model_comparison(
        payload,
        str(output_dir / "model_comparison.png")
    )

    print("\nDone! Plots saved to:", output_dir)


if __name__ == "__main__":
    main()
