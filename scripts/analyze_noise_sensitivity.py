"""Noise sensitivity analysis - evaluate model performance under increasing noise levels."""

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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from cashflow.engine import ForecastEngine, ForecastConfig
from cashflow.schemas.forecast import ExplainabilityPayload
from cashflow.pipeline import clean_utf, detect_transfers, net_transfers, aggregate_monthly
from cashflow.pipeline.decomposition import decompose_cashflow

# Plot settings
COLORS = {
    "actual": "#2E86AB",
    "forecast": "#28A745",
    "threshold": "#DC3545",
    "noise_gradient": ["#28A745", "#7CB342", "#C0CA33", "#FDD835", "#FFB300", "#FB8C00", "#F4511E", "#E53935"],
}
FIG_SIZE = (14, 8)
FIG_SIZE_WIDE = (16, 6)
DPI = 150


@dataclass
class NoiseConfig:
    """Configuration for noise levels in synthetic data."""
    name: str
    salary_std: float          # Standard deviation of salary noise
    expense_std: float         # Standard deviation of expense noise
    grocery_std: float         # Exponential scale for groceries
    outlier_magnitude: float   # Multiplier for outlier amounts
    random_expense_prob: float # Probability of random unexpected expense
    random_expense_max: float  # Max amount for random expenses


# Define noise levels from clean to very noisy
NOISE_LEVELS = [
    NoiseConfig("Baseline (No Noise)", 0, 0, 0, 1.0, 0, 0),
    NoiseConfig("Very Low Noise", 25, 10, 20, 1.0, 0.05, 200),
    NoiseConfig("Low Noise", 50, 20, 40, 1.2, 0.10, 400),
    NoiseConfig("Moderate Noise", 100, 40, 60, 1.5, 0.15, 600),
    NoiseConfig("High Noise", 200, 80, 100, 2.0, 0.20, 1000),
]


def generate_synthetic_data(noise_config: NoiseConfig, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic UTF transaction data with configurable noise levels.

    Args:
        noise_config: Configuration for noise parameters
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic transactions
    """
    np.random.seed(seed)

    transactions = []
    tx_id = 1

    # Base amounts (no noise reference)
    base_salary = 3000
    base_rent = 1200
    base_utilities = 120
    base_groceries = 80

    # Generate 24 months of data
    for year in [2024, 2025]:
        for month in range(1, 13):
            if year == 2025 and month > 12:
                break

            month_start = datetime(year, month, 1)

            # Salary (recurring income) - with noise
            salary_noise = np.random.normal(0, noise_config.salary_std) if noise_config.salary_std > 0 else 0
            salary = base_salary + salary_noise
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

            # Rent (recurring expense) - fixed
            transactions.append({
                "tx_id": f"TX{tx_id:06d}",
                "customer_id": "CUST001",
                "account_id": "MAIN_CHECKING",
                "tx_date": month_start,
                "amount": -base_rent,
                "currency": "EUR",
                "direction": "DEBIT",
                "category": "RENT_MORTGAGE",
                "description_raw": f"RENT {year}-{month:02d}",
                "is_recurring_flag": True,
                "is_variable_amount": False,
            })
            tx_id += 1

            # Utilities (seasonal variation + noise)
            winter_factor = 1.5 if month in [11, 12, 1, 2] else 1.0
            utility_noise = np.random.normal(0, noise_config.expense_std) if noise_config.expense_std > 0 else 0
            utilities = -(base_utilities * winter_factor + utility_noise)
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
            num_grocery_trips = np.random.randint(3, 6)
            for _ in range(num_grocery_trips):
                day = np.random.randint(1, 28)
                grocery_var = np.random.exponential(noise_config.grocery_std) if noise_config.grocery_std > 0 else 0
                amount = -(base_groceries + grocery_var)
                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN_CHECKING",
                    "tx_date": datetime(year, month, day),
                    "amount": amount,
                    "currency": "EUR",
                    "direction": "DEBIT",
                    "category": "GROCERIES",
                    "description_raw": "GROCERIES PURCHASE",
                    "is_recurring_flag": False,
                    "is_variable_amount": True,
                })
                tx_id += 1

            # Internal transfer (to be netted out)
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

            # Random unexpected expenses (noise-dependent probability)
            if np.random.random() < noise_config.random_expense_prob:
                random_amount = -np.random.uniform(100, noise_config.random_expense_max)
                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN_CHECKING",
                    "tx_date": datetime(year, month, np.random.randint(1, 28)),
                    "amount": random_amount,
                    "currency": "EUR",
                    "direction": "DEBIT",
                    "category": "MISCELLANEOUS",
                    "description_raw": "UNEXPECTED EXPENSE",
                    "is_recurring_flag": False,
                    "is_variable_amount": True,
                })
                tx_id += 1

            # Outliers (scaled by noise level)
            # Tax refund in August 2024
            if year == 2024 and month == 8:
                outlier_amount = 5000 * noise_config.outlier_magnitude
                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN_CHECKING",
                    "tx_date": datetime(year, month, 15),
                    "amount": outlier_amount,
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
                vacation_amount = -1800 * noise_config.outlier_magnitude
                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN_CHECKING",
                    "tx_date": datetime(year, month, 20),
                    "amount": vacation_amount,
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


def run_forecast(utf_df: pd.DataFrame) -> Tuple[ExplainabilityPayload, pd.DataFrame]:
    """Run forecast and return payload plus historical monthly data."""
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


@dataclass
class NoiseResult:
    """Results from running forecast at a specific noise level."""
    noise_config: NoiseConfig
    wmape: float
    model_selected: str
    meets_threshold: bool
    num_outliers: int
    avg_ci_width: float
    historical_std: float
    forecast_values: List[float]


def analyze_noise_level(noise_config: NoiseConfig, seed: int = 42) -> NoiseResult:
    """Run analysis for a single noise level."""
    utf_df = generate_synthetic_data(noise_config, seed=seed)
    payload, historical_df = run_forecast(utf_df)

    # Calculate metrics
    forecast_values = [fr.forecast_total for fr in payload.forecast_results]
    ci_widths = [fr.upper_ci - fr.lower_ci for fr in payload.forecast_results]

    return NoiseResult(
        noise_config=noise_config,
        wmape=payload.wmape_winner,
        model_selected=payload.model_selected,
        meets_threshold=payload.meets_threshold,
        num_outliers=len(payload.outliers_detected),
        avg_ci_width=np.mean(ci_widths),
        historical_std=historical_df["necf"].std(),
        forecast_values=forecast_values,
    )


def run_noise_analysis(seeds: List[int] = None) -> Dict[str, List[NoiseResult]]:
    """Run analysis across all noise levels with multiple seeds."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 2024, 1337, 9999, 5555, 7777, 3141,
                 1111, 2222, 3333, 4444, 6666, 8888, 1234, 5678, 9012, 3456,
                 1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 1010]

    results = {}

    for noise_config in NOISE_LEVELS:
        print(f"  Analyzing: {noise_config.name}...")
        results[noise_config.name] = []

        for seed in seeds:
            try:
                result = analyze_noise_level(noise_config, seed=seed)
                results[noise_config.name].append(result)
            except Exception as e:
                print(f"    Warning: Seed {seed} failed - {e}")

    return results


def plot_wmape_vs_noise(results: Dict[str, List[NoiseResult]], output_path: str) -> None:
    """Plot WMAPE degradation as noise increases."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    noise_levels = list(results.keys())

    # Calculate mean and std of WMAPE for each noise level
    means = []
    stds = []
    for level in noise_levels:
        wmapes = [r.wmape for r in results[level]]
        means.append(np.mean(wmapes))
        stds.append(np.std(wmapes))

    x = np.arange(len(noise_levels))
    colors = COLORS["noise_gradient"][:len(noise_levels)]

    # Bar chart with error bars
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="white", linewidth=2, alpha=0.8)

    # Threshold line
    ax.axhline(y=20, color=COLORS["threshold"], linewidth=2.5,
               linestyle="--", label="Acceptance Threshold (20%)")

    # Value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
               f"{mean:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Styling
    ax.set_xlabel("Noise Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("WMAPE (%)", fontsize=12, fontweight="bold")
    ax.set_title("Model Accuracy Degradation Under Increasing Noise\n(Mean ± Std across 30 random seeds)",
                fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in noise_levels], fontsize=9, rotation=0)
    ax.set_ylim(0, max(means) * 1.4)

    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_forecast_comparison(results: Dict[str, List[NoiseResult]], output_path: str) -> None:
    """Plot forecast trajectories for different noise levels."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    noise_levels = list(results.keys())
    colors = COLORS["noise_gradient"][:len(noise_levels)]

    # Use first seed result for each noise level
    for i, (level, color) in enumerate(zip(noise_levels, colors)):
        if results[level]:
            forecast = results[level][0].forecast_values
            months = range(1, len(forecast) + 1)

            alpha = 1.0 - (i * 0.08)  # Fade with noise
            linewidth = 2.5 - (i * 0.2)

            ax.plot(months, forecast, color=color, linewidth=max(1, linewidth),
                   alpha=max(0.4, alpha), marker="o", markersize=4,
                   label=f"{level} (WMAPE: {results[level][0].wmape:.1f}%)")

    # Styling
    ax.set_xlabel("Forecast Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Cash Flow (EUR)", fontsize=12, fontweight="bold")
    ax.set_title("Forecast Trajectories Under Different Noise Levels",
                fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(range(1, 13))
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ci_width_vs_noise(results: Dict[str, List[NoiseResult]], output_path: str) -> None:
    """Plot confidence interval width expansion with noise."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    noise_levels = list(results.keys())
    colors = COLORS["noise_gradient"][:len(noise_levels)]

    # Left plot: Average CI width
    ci_means = []
    ci_stds = []
    for level in noise_levels:
        widths = [r.avg_ci_width for r in results[level]]
        ci_means.append(np.mean(widths))
        ci_stds.append(np.std(widths))

    x = np.arange(len(noise_levels))
    ax1.bar(x, ci_means, yerr=ci_stds, capsize=4, color=colors,
           edgecolor="white", linewidth=1.5, alpha=0.8)

    ax1.set_xlabel("Noise Level", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average CI Width (EUR)", fontsize=11, fontweight="bold")
    ax1.set_title("Confidence Interval Expansion", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.split()[0] for n in noise_levels], fontsize=8, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_facecolor("#FAFAFA")

    # Right plot: Historical volatility
    hist_stds = []
    for level in noise_levels:
        stds = [r.historical_std for r in results[level]]
        hist_stds.append(np.mean(stds))

    ax2.bar(x, hist_stds, color=colors, edgecolor="white", linewidth=1.5, alpha=0.8)

    ax2.set_xlabel("Noise Level", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Historical NECF Std Dev (EUR)", fontsize=11, fontweight="bold")
    ax2.set_title("Historical Data Volatility", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.split()[0] for n in noise_levels], fontsize=8, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_selection_distribution(results: Dict[str, List[NoiseResult]], output_path: str) -> None:
    """Plot which models win at different noise levels."""
    fig, ax = plt.subplots(figsize=(12, 6))

    noise_levels = list(results.keys())

    # Count model selections
    ets_counts = []
    sarima_counts = []

    for level in noise_levels:
        ets = sum(1 for r in results[level] if r.model_selected == "ETS")
        sarima = sum(1 for r in results[level] if r.model_selected == "SARIMA")
        ets_counts.append(ets)
        sarima_counts.append(sarima)

    x = np.arange(len(noise_levels))
    width = 0.35

    bars1 = ax.bar(x - width/2, ets_counts, width, label="ETS",
                   color=COLORS["actual"], edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width/2, sarima_counts, width, label="SARIMA",
                   color=COLORS["forecast"], edgecolor="white", linewidth=1.5)

    # Add value labels
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Noise Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Wins (out of 30 seeds)", fontsize=12, fontweight="bold")
    ax.set_title("Model Selection Under Different Noise Conditions",
                fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in noise_levels], fontsize=9)
    ax.set_ylim(0, 6)

    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_outlier_detection_rate(results: Dict[str, List[NoiseResult]], output_path: str) -> None:
    """Plot outlier detection across noise levels."""
    fig, ax = plt.subplots(figsize=(12, 6))

    noise_levels = list(results.keys())
    colors = COLORS["noise_gradient"][:len(noise_levels)]

    # Calculate average outliers detected
    avg_outliers = []
    for level in noise_levels:
        outliers = [r.num_outliers for r in results[level]]
        avg_outliers.append(np.mean(outliers))

    x = np.arange(len(noise_levels))
    bars = ax.bar(x, avg_outliers, color=colors, edgecolor="white", linewidth=2, alpha=0.8)

    # Value labels
    for bar, val in zip(bars, avg_outliers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Noise Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg Outliers Detected", fontsize=12, fontweight="bold")
    ax.set_title("Outlier Detection Rate Under Increasing Noise",
                fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in noise_levels], fontsize=9)

    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_threshold_pass_rate(results: Dict[str, List[NoiseResult]], output_path: str) -> None:
    """Plot percentage of runs meeting the 20% WMAPE threshold."""
    fig, ax = plt.subplots(figsize=(12, 6))

    noise_levels = list(results.keys())
    colors = COLORS["noise_gradient"][:len(noise_levels)]

    # Calculate pass rates
    pass_rates = []
    for level in noise_levels:
        passes = sum(1 for r in results[level] if r.meets_threshold)
        total = len(results[level])
        pass_rates.append(100 * passes / total if total > 0 else 0)

    x = np.arange(len(noise_levels))
    bars = ax.bar(x, pass_rates, color=colors, edgecolor="white", linewidth=2, alpha=0.8)

    # Color bars based on pass rate
    for bar, rate in zip(bars, pass_rates):
        if rate < 50:
            bar.set_color(COLORS["threshold"])
        elif rate < 80:
            bar.set_color("#FFC107")

    # Value labels
    for bar, rate in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f"{rate:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Reference lines
    ax.axhline(y=100, color=COLORS["forecast"], linewidth=2, linestyle="-", alpha=0.5)
    ax.axhline(y=50, color="#FFC107", linewidth=1.5, linestyle="--", alpha=0.5)

    ax.set_xlabel("Noise Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pass Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Percentage of Runs Meeting WMAPE < 20% Threshold\n(30 random seeds per noise level)",
                fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in noise_levels], fontsize=9)
    ax.set_ylim(0, 115)

    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(results: Dict[str, List[NoiseResult]], output_path: str) -> None:
    """Generate a summary table of results."""
    rows = []

    for level in results.keys():
        level_results = results[level]
        if not level_results:
            continue

        wmapes = [r.wmape for r in level_results]
        ci_widths = [r.avg_ci_width for r in level_results]
        outliers = [r.num_outliers for r in level_results]
        passes = sum(1 for r in level_results if r.meets_threshold)

        ets_wins = sum(1 for r in level_results if r.model_selected == "ETS")

        rows.append({
            "Noise Level": level,
            "WMAPE Mean": f"{np.mean(wmapes):.2f}%",
            "WMAPE Std": f"{np.std(wmapes):.2f}%",
            "CI Width": f"{np.mean(ci_widths):.0f}",
            "Outliers": f"{np.mean(outliers):.1f}",
            "Pass Rate": f"{100*passes/len(level_results):.0f}%",
            "ETS Wins": f"{ets_wins}/{len(level_results)}",
        })

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_path.replace(".png", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Create table plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colColours=["#E3F2FD"] * len(df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#1976D2")
            cell.set_text_props(color="white", fontweight="bold")

    plt.title("Noise Sensitivity Analysis Summary", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Run complete noise sensitivity analysis."""
    output_dir = Path(__file__).parent.parent / "plots" / "noise_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NOISE SENSITIVITY ANALYSIS")
    print("=" * 60)

    print(f"\nAnalyzing {len(NOISE_LEVELS)} noise levels with 30 random seeds each...")
    results = run_noise_analysis()

    print("\nGenerating plots...")

    # Plot 1: WMAPE vs Noise
    plot_wmape_vs_noise(results, str(output_dir / "wmape_vs_noise.png"))

    # Plot 2: Forecast comparison
    plot_forecast_comparison(results, str(output_dir / "forecast_trajectories.png"))

    # Plot 3: CI width expansion
    plot_ci_width_vs_noise(results, str(output_dir / "ci_width_vs_noise.png"))

    # Plot 4: Model selection distribution
    plot_model_selection_distribution(results, str(output_dir / "model_selection.png"))

    # Plot 5: Outlier detection rate
    plot_outlier_detection_rate(results, str(output_dir / "outlier_detection.png"))

    # Plot 6: Threshold pass rate
    plot_threshold_pass_rate(results, str(output_dir / "threshold_pass_rate.png"))

    # Generate summary table
    generate_summary_table(results, str(output_dir / "summary_table.png"))

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")

    # Print summary
    print("\n--- Quick Summary ---")
    for level in results.keys():
        if results[level]:
            wmapes = [r.wmape for r in results[level]]
            print(f"{level}: WMAPE = {np.mean(wmapes):.2f}% ± {np.std(wmapes):.2f}%")


if __name__ == "__main__":
    main()
