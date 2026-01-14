#!/usr/bin/env python3
"""
Cash Flow Forecasting Framework Test Runner

Main entry point for running the standardized testing framework
as specified in docs/2026_01_13_framework.md

Usage:
    python scripts/run_framework_tests.py           # Full suite (120 runs)
    python scripts/run_framework_tests.py --quick   # Quick validation (36 runs)
    python scripts/run_framework_tests.py --account-type personal
    python scripts/run_framework_tests.py --randomness low
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from framework_config import (
    AccountType,
    RandomnessLevel,
    TestConfig,
    get_all_test_configs,
)
from data_generator import (
    generate_synthetic_data,
    generate_full_period_data,
    aggregate_monthly_actuals,
)

from cashflow.engine import ForecastEngine, ForecastConfig
from cashflow.pipeline import clean_utf, detect_transfers, net_transfers, aggregate_monthly
from cashflow.pipeline.decomposition import decompose_cashflow
from cashflow.utils import calculate_wmape


# Output directories
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = Path(__file__).parent.parent / "plots"


def run_single_test(config: TestConfig) -> Dict:
    """Run a single forecast test and return metrics.

    Args:
        config: Test configuration

    Returns:
        Dictionary with test results and metrics
    """
    start_time = time.time()

    try:
        # Generate continuous 36-month data (24 training + 12 holdout)
        start_date = datetime(2024, 1, 1)
        training_df, holdout_df = generate_full_period_data(config, start_date)

        # Run forecast pipeline
        engine = ForecastEngine(ForecastConfig(
            lookback_min_months=12,
            forecast_horizon=config.forecast_horizon,
            models_to_evaluate=["tirex"],
        ))

        # Clean and prepare training data
        training_df = clean_utf(training_df)
        training_df = detect_transfers(training_df, date_tolerance_days=2)
        external_df, _ = net_transfers(training_df)

        # Aggregate historical monthly data for plotting (with decomposition)
        historical_monthly = aggregate_monthly_actuals(external_df, decompose=True)
        historical_values = historical_monthly["net_cash_flow"].values.tolist()
        historical_predictable = historical_monthly["predictable"].values.tolist()
        historical_residual = historical_monthly["residual"].values.tolist()

        # Aggregate holdout with decomposition
        holdout_monthly = aggregate_monthly_actuals(holdout_df, decompose=True)
        holdout_predictable = holdout_monthly["predictable"].values[:config.forecast_horizon].tolist()
        holdout_residual = holdout_monthly["residual"].values[:config.forecast_horizon].tolist()

        # Run forecast
        payload = engine.run_from_dataframe(external_df)

        # Extract forecast values
        forecast_values = []
        for result in payload.forecast_results:
            forecast_values.append(result.forecast_total)

        forecast_values = np.array(forecast_values[:config.forecast_horizon])

        # Get actual values from holdout
        actual_values = holdout_monthly["net_cash_flow"].values[:config.forecast_horizon]

        # Ensure same length
        min_len = min(len(forecast_values), len(actual_values))
        forecast_values = forecast_values[:min_len]
        actual_values = actual_values[:min_len]

        # Calculate WMAPE metrics
        wmape_metrics = calculate_wmape_metrics(forecast_values, actual_values)

        # Get model info (use model_dump() for pydantic v2)
        payload_dict = payload.model_dump()
        model_selected = payload_dict.get("model_selected", "Unknown")
        passes_threshold = payload_dict.get("meets_threshold", False)

        # Calculate CI width
        ci_widths = [r.ci_width for r in payload.forecast_results]
        ci_width_avg = np.mean(ci_widths) if ci_widths else 0

        runtime = time.time() - start_time

        return {
            "account_type": config.account_type.value,
            "randomness": config.randomness_level.value,
            "seed": config.seed,
            "config_id": config.config_id,
            **wmape_metrics,
            "model_selected": model_selected,
            "passes_threshold": passes_threshold,
            "ci_width_avg": round(ci_width_avg, 2),
            "runtime_seconds": round(runtime, 2),
            "status": "success",
            "error": None,
            "historical_values": historical_values,
            "forecast_values": forecast_values.tolist(),
            "actual_values": actual_values.tolist(),
            # Decomposition data
            "historical_predictable": historical_predictable,
            "historical_residual": historical_residual,
            "holdout_predictable": holdout_predictable,
            "holdout_residual": holdout_residual,
        }

    except Exception as e:
        runtime = time.time() - start_time
        return {
            "account_type": config.account_type.value,
            "randomness": config.randomness_level.value,
            "seed": config.seed,
            "config_id": config.config_id,
            "wmape_3m": None,
            "wmape_6m": None,
            "wmape_12m": None,
            **{f"wmape_m{i}": None for i in range(1, 13)},
            "model_selected": None,
            "passes_threshold": False,
            "ci_width_avg": None,
            "runtime_seconds": round(runtime, 2),
            "status": "error",
            "error": str(e),
            "historical_values": [],
            "forecast_values": [],
            "actual_values": [],
            "historical_predictable": [],
            "historical_residual": [],
            "holdout_predictable": [],
            "holdout_residual": [],
        }


def calculate_wmape_metrics(
    forecast: np.ndarray,
    actual: np.ndarray,
) -> Dict:
    """Calculate WMAPE at multiple horizons and per-period.

    Args:
        forecast: Forecast values array
        actual: Actual values array

    Returns:
        Dictionary with WMAPE metrics
    """
    metrics = {}

    # Ensure arrays
    forecast = np.array(forecast)
    actual = np.array(actual)

    n_periods = min(len(forecast), len(actual), 12)

    # Cumulative WMAPE at key horizons
    for horizon, name in [(3, "wmape_3m"), (6, "wmape_6m"), (12, "wmape_12m")]:
        h = min(horizon, n_periods)
        if h > 0:
            wmape = calculate_wmape(actual[:h], forecast[:h])
            metrics[name] = round(wmape, 2)
        else:
            metrics[name] = None

    # Per-period WMAPE
    for i in range(12):
        if i < n_periods and actual[i] != 0:
            period_wmape = 100 * abs(actual[i] - forecast[i]) / abs(actual[i])
            metrics[f"wmape_m{i + 1}"] = round(period_wmape, 2)
        else:
            metrics[f"wmape_m{i + 1}"] = None

    return metrics


def run_full_suite(
    account_types: Optional[List[AccountType]] = None,
    randomness_levels: Optional[List[RandomnessLevel]] = None,
    seeds: int = 10,
) -> pd.DataFrame:
    """Run the full test suite.

    Args:
        account_types: List of account types to test (None = all)
        randomness_levels: List of randomness levels to test (None = all)
        seeds: Number of seeds per configuration

    Returns:
        DataFrame with all test results
    """
    # Filter configurations
    all_configs = get_all_test_configs(seeds)

    if account_types:
        all_configs = [c for c in all_configs if c.account_type in account_types]

    if randomness_levels:
        all_configs = [c for c in all_configs if c.randomness_level in randomness_levels]

    print(f"Running {len(all_configs)} test configurations...")
    print("-" * 60)

    results = []
    for i, config in enumerate(all_configs):
        print(f"[{i + 1}/{len(all_configs)}] {config.config_id} seed={config.seed}...", end=" ")
        result = run_single_test(config)
        status = result["status"]
        wmape = result.get("wmape_12m", "N/A")
        print(f"{status} (WMAPE: {wmape}%)")
        results.append(result)

    print("-" * 60)
    print(f"Completed {len(results)} tests")

    return pd.DataFrame(results)


def export_results(results_df: pd.DataFrame) -> None:
    """Export results to CSV files.

    Args:
        results_df: DataFrame with test results
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Export detailed results
    detail_cols = [
        "account_type", "randomness", "seed", "config_id",
        "wmape_3m", "wmape_6m", "wmape_12m",
    ] + [f"wmape_m{i}" for i in range(1, 13)] + [
        "model_selected", "passes_threshold", "ci_width_avg", "runtime_seconds",
        "status", "error"
    ]

    detail_df = results_df[[c for c in detail_cols if c in results_df.columns]]
    detail_df.to_csv(RESULTS_DIR / "wmape_results.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'wmape_results.csv'}")

    # Generate summary statistics
    summary_data = []
    for account_type in AccountType:
        for randomness in RandomnessLevel:
            subset = results_df[
                (results_df["account_type"] == account_type.value) &
                (results_df["randomness"] == randomness.value) &
                (results_df["status"] == "success")
            ]

            if len(subset) == 0:
                continue

            summary_data.append({
                "account_type": account_type.value,
                "randomness": randomness.value,
                "n_runs": len(subset),
                "wmape_3m_mean": round(subset["wmape_3m"].mean(), 2),
                "wmape_3m_std": round(subset["wmape_3m"].std(), 2),
                "wmape_6m_mean": round(subset["wmape_6m"].mean(), 2),
                "wmape_6m_std": round(subset["wmape_6m"].std(), 2),
                "wmape_12m_mean": round(subset["wmape_12m"].mean(), 2),
                "wmape_12m_std": round(subset["wmape_12m"].std(), 2),
                "pass_rate": round(100 * subset["passes_threshold"].mean(), 1),
                "model_ets_pct": round(100 * (subset["model_selected"] == "ETS").mean(), 1),
                "model_sarima_pct": round(100 * (subset["model_selected"] == "SARIMA").mean(), 1),
                "model_tirex_pct": round(100 * (subset["model_selected"] == "TiRex").mean(), 1),
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(RESULTS_DIR / "wmape_summary.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'wmape_summary.csv'}")


def generate_plots(results_df: pd.DataFrame) -> None:
    """Generate all visualization plots.

    Args:
        results_df: DataFrame with test results
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to successful runs
    df = results_df[results_df["status"] == "success"].copy()

    if len(df) == 0:
        print("No successful runs to plot")
        return

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    # 1. WMAPE by Account Type
    _plot_wmape_by_account_type(df)

    # 2. WMAPE by Randomness Level
    _plot_wmape_by_randomness(df)

    # 3. WMAPE Heatmap
    _plot_wmape_heatmap(df)

    # 4. WMAPE Horizon Degradation
    _plot_horizon_degradation(df)

    # 5. Pass Rate Matrix
    _plot_pass_rate_matrix(df)

    # 6. Forecast Trajectories
    _plot_forecast_trajectories(results_df)

    # 7. Decomposition Analysis
    _plot_decomposition(results_df)


def _plot_wmape_by_account_type(df: pd.DataFrame) -> None:
    """Plot WMAPE grouped by account type."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    plot_data = df.groupby(["account_type", "randomness"])["wmape_12m"].mean().reset_index()

    # Create grouped bar chart
    account_types = ["personal", "sme", "corporate"]
    randomness_levels = ["none", "low", "medium", "high"]
    x = np.arange(len(account_types))
    width = 0.2

    for i, rand in enumerate(randomness_levels):
        values = [
            plot_data[(plot_data["account_type"] == acc) & (plot_data["randomness"] == rand)]["wmape_12m"].values
            for acc in account_types
        ]
        values = [v[0] if len(v) > 0 else 0 for v in values]
        ax.bar(x + i * width, values, width, label=rand.capitalize())

    ax.set_xlabel("Account Type", fontsize=12)
    ax.set_ylabel("WMAPE (%)", fontsize=12)
    ax.set_title("12-Month WMAPE by Account Type and Randomness Level", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([t.upper() for t in account_types])
    ax.axhline(y=20, color="red", linestyle="--", label="20% Threshold")
    ax.legend(title="Randomness")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "wmape_by_account_type.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'wmape_by_account_type.png'}")


def _plot_wmape_by_randomness(df: pd.DataFrame) -> None:
    """Plot WMAPE grouped by randomness level."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    plot_data = df.groupby(["randomness", "account_type"])["wmape_12m"].mean().reset_index()

    # Create grouped bar chart
    randomness_levels = ["none", "low", "medium", "high"]
    account_types = ["personal", "sme", "corporate"]
    x = np.arange(len(randomness_levels))
    width = 0.25

    for i, acc in enumerate(account_types):
        values = [
            plot_data[(plot_data["randomness"] == rand) & (plot_data["account_type"] == acc)]["wmape_12m"].values
            for rand in randomness_levels
        ]
        values = [v[0] if len(v) > 0 else 0 for v in values]
        ax.bar(x + i * width, values, width, label=acc.upper())

    ax.set_xlabel("Randomness Level", fontsize=12)
    ax.set_ylabel("WMAPE (%)", fontsize=12)
    ax.set_title("12-Month WMAPE by Randomness Level and Account Type", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([r.capitalize() for r in randomness_levels])
    ax.axhline(y=20, color="red", linestyle="--", label="20% Threshold")
    ax.legend(title="Account Type")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "wmape_by_randomness.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'wmape_by_randomness.png'}")


def _plot_wmape_heatmap(df: pd.DataFrame) -> None:
    """Plot WMAPE heatmap (account type vs randomness)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pivot data for heatmap
    pivot_data = df.pivot_table(
        values="wmape_12m",
        index="account_type",
        columns="randomness",
        aggfunc="mean"
    )

    # Reorder
    row_order = ["personal", "sme", "corporate"]
    col_order = ["none", "low", "medium", "high"]
    pivot_data = pivot_data.reindex(index=row_order, columns=col_order)

    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        center=20,
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={"label": "WMAPE (%)"}
    )

    ax.set_xlabel("Randomness Level", fontsize=12)
    ax.set_ylabel("Account Type", fontsize=12)
    ax.set_title("12-Month WMAPE Heatmap", fontsize=14)
    ax.set_xticklabels([c.capitalize() for c in col_order])
    ax.set_yticklabels([r.upper() for r in row_order], rotation=0)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "wmape_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'wmape_heatmap.png'}")


def _plot_horizon_degradation(df: pd.DataFrame) -> None:
    """Plot WMAPE degradation over forecast horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    account_types = ["personal", "sme", "corporate"]
    randomness_levels = ["none", "low", "medium", "high"]
    colors = {"none": "green", "low": "blue", "medium": "orange", "high": "red"}

    for ax, acc_type in zip(axes, account_types):
        for rand in randomness_levels:
            subset = df[(df["account_type"] == acc_type) & (df["randomness"] == rand)]

            if len(subset) == 0:
                continue

            # Get per-month WMAPE columns
            wmape_cols = [f"wmape_m{i}" for i in range(1, 13)]
            monthly_wmape = subset[wmape_cols].mean().values

            months = range(1, len(monthly_wmape) + 1)
            ax.plot(months, monthly_wmape, marker="o", label=rand.capitalize(),
                    color=colors[rand], linewidth=2, markersize=4)

        ax.axhline(y=20, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Forecast Month", fontsize=10)
        ax.set_ylabel("WMAPE (%)", fontsize=10)
        ax.set_title(f"{acc_type.upper()}", fontsize=12)
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(0, None)

    fig.suptitle("WMAPE Degradation Over Forecast Horizon", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(PLOTS_DIR / "wmape_horizon_degradation.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'wmape_horizon_degradation.png'}")


def _plot_pass_rate_matrix(df: pd.DataFrame) -> None:
    """Plot pass rate heatmap (account type vs randomness)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate pass rates
    pass_rates = df.groupby(["account_type", "randomness"])["passes_threshold"].mean() * 100

    # Pivot for heatmap
    pivot_data = pass_rates.unstack()

    # Reorder
    row_order = ["personal", "sme", "corporate"]
    col_order = ["none", "low", "medium", "high"]
    pivot_data = pivot_data.reindex(index=row_order, columns=col_order)

    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={"label": "Pass Rate (%)"}
    )

    ax.set_xlabel("Randomness Level", fontsize=12)
    ax.set_ylabel("Account Type", fontsize=12)
    ax.set_title("Pass Rate Matrix (WMAPE < 20%)", fontsize=14)
    ax.set_xticklabels([c.capitalize() for c in col_order])
    ax.set_yticklabels([r.upper() for r in row_order], rotation=0)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pass_rate_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'pass_rate_matrix.png'}")


def _plot_forecast_trajectories(df: pd.DataFrame) -> None:
    """Plot sample forecast trajectories with historical data for each configuration."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    account_types = ["personal", "sme", "corporate"]
    randomness_levels = ["none", "low", "medium", "high"]

    for i, acc_type in enumerate(account_types):
        for j, rand in enumerate(randomness_levels):
            ax = axes[i, j]

            # Get first successful run for this config
            subset = df[
                (df["account_type"] == acc_type) &
                (df["randomness"] == rand) &
                (df["status"] == "success")
            ]

            if len(subset) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{acc_type.upper()} - {rand.capitalize()}")
                continue

            row = subset.iloc[0]
            historical = row.get("historical_values", [])
            forecast = row["forecast_values"]
            actual = row["actual_values"]

            if len(forecast) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{acc_type.upper()} - {rand.capitalize()}")
                continue

            # Historical period (months 1-24)
            n_hist = len(historical)
            hist_months = list(range(1, n_hist + 1))

            # Forecast period (months 25-36)
            n_forecast = min(len(forecast), len(actual))
            forecast_months = list(range(n_hist + 1, n_hist + n_forecast + 1))

            # Plot historical data
            if len(historical) > 0:
                ax.plot(hist_months, historical, "o-", label="Historical",
                        color="gray", linewidth=1.5, markersize=4, alpha=0.7)

            # Plot forecast and actual holdout
            if len(actual) > 0:
                ax.plot(forecast_months, actual[:n_forecast], "o-", label="Actual (Holdout)",
                        color="blue", linewidth=2, markersize=5)
            ax.plot(forecast_months, forecast[:n_forecast], "s--", label="Forecast",
                    color="orange", linewidth=2, markersize=5)

            # Add vertical line to separate historical from forecast
            if n_hist > 0:
                ax.axvline(x=n_hist + 0.5, color="red", linestyle=":", alpha=0.5, label="Forecast Start")

            ax.set_xlabel("Month")
            ax.set_ylabel("Net Cash Flow (EUR)")
            ax.set_title(f"{acc_type.upper()} - {rand.capitalize()}")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Historical Data + Forecast vs Actual Trajectories", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOTS_DIR / "forecast_trajectories.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'forecast_trajectories.png'}")


def _plot_decomposition(df: pd.DataFrame) -> None:
    """Plot decomposition analysis showing predictable vs residual components.

    Creates a grid of plots showing:
    - Historical predictable (recurring) component
    - Historical residual (variable) component
    - Holdout predictable vs residual
    - Forecast vs each component
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))

    account_types = ["personal", "sme", "corporate"]
    randomness_levels = ["none", "low", "medium", "high"]

    for i, acc_type in enumerate(account_types):
        for j, rand in enumerate(randomness_levels):
            ax = axes[i, j]

            # Get first successful run for this config
            subset = df[
                (df["account_type"] == acc_type) &
                (df["randomness"] == rand) &
                (df["status"] == "success")
            ]

            if len(subset) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{acc_type.upper()} - {rand.capitalize()}")
                continue

            row = subset.iloc[0]

            # Get decomposition data
            hist_predictable = row.get("historical_predictable", [])
            hist_residual = row.get("historical_residual", [])
            holdout_predictable = row.get("holdout_predictable", [])
            holdout_residual = row.get("holdout_residual", [])
            forecast = row.get("forecast_values", [])
            actual = row.get("actual_values", [])

            if len(hist_predictable) == 0:
                ax.text(0.5, 0.5, "No decomposition data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{acc_type.upper()} - {rand.capitalize()}")
                continue

            n_hist = len(hist_predictable)
            n_forecast = min(len(forecast), len(actual), len(holdout_predictable)) if holdout_predictable else min(len(forecast), len(actual))

            # X-axis values
            hist_months = list(range(1, n_hist + 1))
            forecast_months = list(range(n_hist + 1, n_hist + n_forecast + 1))

            # Plot historical components as stacked areas
            hist_pred = np.array(hist_predictable)
            hist_resid = np.array(hist_residual)

            # Plot predictable component (typically positive net from income)
            ax.fill_between(hist_months, 0, hist_pred, alpha=0.4, color="green", label="Predictable (Recurring)")
            ax.fill_between(hist_months, hist_pred, hist_pred + hist_resid, alpha=0.4, color="orange", label="Residual (Variable)")

            # Plot total historical as line
            hist_total = hist_pred + hist_resid
            ax.plot(hist_months, hist_total, "k-", linewidth=1.5, alpha=0.7, label="Total (Historical)")

            # Plot holdout period decomposition
            if len(holdout_predictable) > 0 and len(holdout_residual) > 0:
                hold_pred = np.array(holdout_predictable[:n_forecast])
                hold_resid = np.array(holdout_residual[:n_forecast])
                hold_total = hold_pred + hold_resid

                ax.fill_between(forecast_months, 0, hold_pred, alpha=0.3, color="green", hatch="//")
                ax.fill_between(forecast_months, hold_pred, hold_pred + hold_resid, alpha=0.3, color="orange", hatch="//")
                ax.plot(forecast_months, hold_total, "b-", linewidth=2, label="Actual (Holdout)")

            # Plot forecast as dashed line
            if len(forecast) > 0:
                ax.plot(forecast_months, forecast[:n_forecast], "r--", linewidth=2, marker="s", markersize=4, label="Forecast")

            # Add vertical line for forecast start
            ax.axvline(x=n_hist + 0.5, color="gray", linestyle=":", alpha=0.7)

            ax.set_xlabel("Month", fontsize=9)
            ax.set_ylabel("Cash Flow (EUR)", fontsize=9)
            ax.set_title(f"{acc_type.upper()} - {rand.capitalize()}", fontsize=11)
            ax.legend(fontsize=6, loc="best")
            ax.grid(True, alpha=0.3)

            # Add zero line
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    fig.suptitle("Cash Flow Decomposition: Predictable (Recurring) vs Residual (Variable)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOTS_DIR / "decomposition_analysis.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'decomposition_analysis.png'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cash Flow Forecasting Framework Test Runner"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation with 3 seeds instead of 10"
    )
    parser.add_argument(
        "--account-type",
        choices=["personal", "sme", "corporate"],
        help="Run only for specific account type"
    )
    parser.add_argument(
        "--randomness",
        choices=["none", "low", "medium", "high"],
        help="Run only for specific randomness level"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of seeds per configuration (default: 10)"
    )

    args = parser.parse_args()

    # Determine seeds
    seeds = 3 if args.quick else args.seeds

    # Filter account types
    account_types = None
    if args.account_type:
        account_types = [AccountType(args.account_type)]

    # Filter randomness levels
    randomness_levels = None
    if args.randomness:
        randomness_levels = [RandomnessLevel(args.randomness)]

    print("=" * 60)
    print("Cash Flow Forecasting Framework Test Runner")
    print("=" * 60)
    print(f"Seeds per configuration: {seeds}")
    if account_types:
        print(f"Account types: {[a.value for a in account_types]}")
    if randomness_levels:
        print(f"Randomness levels: {[r.value for r in randomness_levels]}")
    print("=" * 60)

    # Run tests
    results_df = run_full_suite(
        account_types=account_types,
        randomness_levels=randomness_levels,
        seeds=seeds,
    )

    # Export results
    print("\nExporting results...")
    export_results(results_df)

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(results_df)

    print("\n" + "=" * 60)
    print("Test run complete!")
    print("=" * 60)

    # Print summary
    success_df = results_df[results_df["status"] == "success"]
    if len(success_df) > 0:
        print(f"\nSummary:")
        print(f"  Total runs: {len(results_df)}")
        print(f"  Successful: {len(success_df)}")
        print(f"  Failed: {len(results_df) - len(success_df)}")
        print(f"  Overall pass rate: {100 * success_df['passes_threshold'].mean():.1f}%")
        print(f"  Average WMAPE (12m): {success_df['wmape_12m'].mean():.1f}%")


if __name__ == "__main__":
    main()
