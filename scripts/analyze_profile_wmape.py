"""Profile-based WMAPE analysis across forecast horizons.

Analyzes how forecast accuracy (WMAPE) evolves across:
- 3 customer profiles: Personal, SME, Corporate
- Multiple noise scenarios
- Forecast horizons 1-12 months
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from cashflow.engine import ForecastEngine, ForecastConfig
from cashflow.pipeline import clean_utf, detect_transfers, net_transfers, aggregate_monthly

# Plot settings
COLORS = {
    "Personal": "#2196F3",      # Blue
    "SME": "#4CAF50",           # Green
    "Corporate": "#FF9800",     # Orange
    "threshold": "#DC3545",     # Red
    "actual": "#6C757D",        # Gray
}
NOISE_COLORS = {
    "Low": "#28A745",
    "Moderate": "#FFC107",
    "High": "#DC3545",
}
FIG_SIZE = (14, 8)
FIG_SIZE_WIDE = (16, 6)
DPI = 150


@dataclass
class ProfileConfig:
    """Configuration for different customer profiles."""
    name: str
    # Income characteristics
    base_income: float
    income_count: int            # Number of income sources per month
    income_std_ratio: float      # Std as ratio of base income
    # Expense characteristics
    base_fixed_expenses: float   # Rent, loan, etc.
    fixed_expense_count: int
    base_variable_expenses: float  # Groceries, misc
    variable_expense_count: int
    expense_std_ratio: float
    # Seasonality
    seasonal_amplitude: float    # 0 = no seasonality, 1 = 100% variation
    # Transaction volume
    transactions_per_month: Tuple[int, int]  # Min, max non-recurring transactions


# Define three distinct customer profiles
PROFILES = {
    "Personal": ProfileConfig(
        name="Personal",
        base_income=3000,
        income_count=1,           # Single salary
        income_std_ratio=0.02,    # Stable salary
        base_fixed_expenses=1500,
        fixed_expense_count=4,    # Rent, utilities, subscription, loan
        base_variable_expenses=100,
        variable_expense_count=8,
        expense_std_ratio=0.3,
        seasonal_amplitude=0.15,  # Light seasonality (holidays)
        transactions_per_month=(10, 20),
    ),
    "SME": ProfileConfig(
        name="SME",
        base_income=25000,
        income_count=5,           # Multiple client payments
        income_std_ratio=0.25,    # Variable client payments
        base_fixed_expenses=8000,
        fixed_expense_count=6,    # Rent, utilities, insurance, etc.
        base_variable_expenses=500,
        variable_expense_count=15,
        expense_std_ratio=0.4,
        seasonal_amplitude=0.3,   # Moderate seasonality (business cycles)
        transactions_per_month=(30, 60),
    ),
    "Corporate": ProfileConfig(
        name="Corporate",
        base_income=500000,
        income_count=20,          # Many revenue streams
        income_std_ratio=0.15,    # Some variability
        base_fixed_expenses=200000,
        fixed_expense_count=15,   # Payroll, rent, utilities, insurance, etc.
        base_variable_expenses=5000,
        variable_expense_count=50,
        expense_std_ratio=0.2,
        seasonal_amplitude=0.1,   # Lower seasonality (diversified)
        transactions_per_month=(100, 200),
    ),
}


@dataclass
class NoiseScenario:
    """Noise configuration for analysis."""
    name: str
    income_noise_multiplier: float
    expense_noise_multiplier: float
    flag_corruption_rate: float
    outlier_probability: float
    outlier_magnitude: float


NOISE_SCENARIOS = [
    NoiseScenario("Low", 1.0, 1.0, 0.05, 0.02, 2.0),
    NoiseScenario("Moderate", 2.0, 2.0, 0.15, 0.05, 3.0),
    NoiseScenario("High", 4.0, 4.0, 0.30, 0.10, 5.0),
]


@dataclass
class HorizonResult:
    """Results for a single forecast run."""
    profile: str
    noise_scenario: str
    seed: int
    wmape_total: float
    wmape_by_horizon: List[float]  # WMAPE at each step 1-12
    forecast_values: List[float]
    actual_values: List[float]
    model_selected: str


def generate_profile_data(
    profile: ProfileConfig,
    noise: NoiseScenario,
    seed: int = 42,
    months: int = 36,  # 24 training + 12 holdout
) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:
    """Generate synthetic data for a customer profile.

    Uses a consistent base pattern so holdout is predictable from training.

    Returns:
        Tuple of (training_df, holdout_df, expected_monthly_necf)
        expected_monthly_necf contains the "true" underlying pattern for all 36 months
    """
    np.random.seed(seed)

    transactions = []
    tx_id = 1
    expected_monthly_necf = []  # Track expected net cash flow per month

    for month_idx in range(months):
        year = 2023 + month_idx // 12
        month = (month_idx % 12) + 1

        # Seasonal factor (peaks in Q4, dips in Q1 for business)
        seasonal_factor = 1 + profile.seasonal_amplitude * np.sin(
            2 * np.pi * (month - 3) / 12
        )

        month_income = 0.0
        month_expense = 0.0

        # === INCOME TRANSACTIONS ===
        for income_idx in range(profile.income_count):
            base = profile.base_income / profile.income_count
            expected_amount = base * seasonal_factor
            noise_std = base * profile.income_std_ratio * noise.income_noise_multiplier
            amount = expected_amount + np.random.normal(0, noise_std)
            amount = max(amount, 0)  # Income is positive

            month_income += amount

            # Corrupt flag with probability
            is_recurring = True
            if np.random.random() < noise.flag_corruption_rate:
                is_recurring = False

            transactions.append({
                "tx_id": f"TX{tx_id:08d}",
                "customer_id": f"CUST_{profile.name.upper()}",
                "account_id": f"CUST_{profile.name.upper()}_CHECKING",
                "tx_date": datetime(year, month, min(income_idx + 1, 28)),
                "amount": amount,
                "currency": "EUR",
                "direction": "CREDIT",
                "category": "SALARY" if profile.name == "Personal" else "REVENUE",
                "description_raw": f"INCOME_{income_idx + 1}",
                "counterparty_key": f"INCOME_SOURCE_{income_idx + 1}",
                "is_recurring_flag": is_recurring,
                "is_variable_amount": True,
            })
            tx_id += 1

        # === FIXED EXPENSE TRANSACTIONS ===
        categories = ["RENT_MORTGAGE", "UTILITIES", "INSURANCE", "LOAN_PAYMENT",
                      "SUBSCRIPTION", "PAYROLL"]
        for exp_idx in range(profile.fixed_expense_count):
            base = profile.base_fixed_expenses / profile.fixed_expense_count
            noise_std = base * 0.05 * noise.expense_noise_multiplier  # Fixed expenses are stable
            amount = -(base + np.random.normal(0, noise_std))

            month_expense += amount

            is_recurring = True
            if np.random.random() < noise.flag_corruption_rate:
                is_recurring = False

            cat_idx = exp_idx % len(categories)
            transactions.append({
                "tx_id": f"TX{tx_id:08d}",
                "customer_id": f"CUST_{profile.name.upper()}",
                "account_id": f"CUST_{profile.name.upper()}_CHECKING",
                "tx_date": datetime(year, month, min(exp_idx + 5, 28)),
                "amount": amount,
                "currency": "EUR",
                "direction": "DEBIT",
                "category": categories[cat_idx],
                "description_raw": f"FIXED_EXPENSE_{exp_idx + 1}",
                "counterparty_key": f"FIXED_VENDOR_{exp_idx + 1}",
                "is_recurring_flag": is_recurring,
                "is_variable_amount": False,
            })
            tx_id += 1

        # === VARIABLE EXPENSE TRANSACTIONS ===
        # Use consistent number of transactions per seed for reproducibility
        n_var = profile.transactions_per_month[0] + (
            (seed + month_idx) % (profile.transactions_per_month[1] - profile.transactions_per_month[0] + 1)
        )
        var_categories = ["GROCERIES", "SHOPPING", "RESTAURANT", "TRANSPORT",
                         "ENTERTAINMENT", "MISCELLANEOUS", "SUPPLIES"]

        for var_idx in range(n_var):
            base = profile.base_variable_expenses
            noise_std = base * profile.expense_std_ratio * noise.expense_noise_multiplier
            amount = -(base + np.abs(np.random.normal(0, noise_std)))

            month_expense += amount

            day = np.random.randint(1, 28)
            cat_idx = var_idx % len(var_categories)

            transactions.append({
                "tx_id": f"TX{tx_id:08d}",
                "customer_id": f"CUST_{profile.name.upper()}",
                "account_id": f"CUST_{profile.name.upper()}_CHECKING",
                "tx_date": datetime(year, month, day),
                "amount": amount,
                "currency": "EUR",
                "direction": "DEBIT",
                "category": var_categories[cat_idx],
                "description_raw": f"VARIABLE_EXPENSE",
                "counterparty_key": f"VENDOR_{var_idx}",
                "is_recurring_flag": False,
                "is_variable_amount": True,
            })
            tx_id += 1

        # === OUTLIERS (only in training period to avoid contamination) ===
        if month_idx < 24 and np.random.random() < noise.outlier_probability:
            outlier_type = np.random.choice(["positive", "negative"])
            if outlier_type == "positive":
                # Large unexpected income (tax refund, bonus, big sale)
                amount = profile.base_income * noise.outlier_magnitude
                category = "TAX_REFUND" if profile.name == "Personal" else "REVENUE"
                direction = "CREDIT"
                month_income += amount
            else:
                # Large unexpected expense
                amount = -profile.base_fixed_expenses * noise.outlier_magnitude
                category = "MISCELLANEOUS"
                direction = "DEBIT"
                month_expense += amount

            transactions.append({
                "tx_id": f"TX{tx_id:08d}",
                "customer_id": f"CUST_{profile.name.upper()}",
                "account_id": f"CUST_{profile.name.upper()}_CHECKING",
                "tx_date": datetime(year, month, 15),
                "amount": amount,
                "currency": "EUR",
                "direction": direction,
                "category": category,
                "description_raw": "OUTLIER_TRANSACTION",
                "counterparty_key": "OUTLIER",
                "is_recurring_flag": False,
                "is_variable_amount": True,
            })
            tx_id += 1

        expected_monthly_necf.append(month_income + month_expense)

    df = pd.DataFrame(transactions)
    df["tx_date"] = pd.to_datetime(df["tx_date"])

    # Split into training (24 months) and holdout (12 months)
    cutoff_date = datetime(2025, 1, 1)  # 24 months from 2023-01
    train_df = df[df["tx_date"] < cutoff_date].copy()
    holdout_df = df[df["tx_date"] >= cutoff_date].copy()

    return train_df, holdout_df, expected_monthly_necf


def compute_monthly_actuals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly net external cash flow from transactions."""
    df = clean_utf(df)
    df = detect_transfers(df, date_tolerance_days=2)
    external_df, _ = net_transfers(df)
    monthly_df = aggregate_monthly(external_df)
    return monthly_df


def compute_wmape_by_horizon(
    forecast_values: List[float],
    actual_values: List[float],
) -> Tuple[float, List[float]]:
    """Compute overall and per-horizon WMAPE.

    Returns:
        Tuple of (total_wmape, list of wmape at each horizon)
    """
    n = min(len(forecast_values), len(actual_values))
    if n == 0:
        return 0.0, []

    forecast = np.array(forecast_values[:n])
    actual = np.array(actual_values[:n])

    # Per-horizon WMAPE (cumulative up to that point)
    wmape_by_horizon = []
    for h in range(1, n + 1):
        f_slice = forecast[:h]
        a_slice = actual[:h]
        total_abs_actual = np.sum(np.abs(a_slice))
        if total_abs_actual > 0:
            wmape = 100 * np.sum(np.abs(f_slice - a_slice)) / total_abs_actual
        else:
            wmape = 0.0
        wmape_by_horizon.append(wmape)

    # Total WMAPE
    total_abs_actual = np.sum(np.abs(actual))
    if total_abs_actual > 0:
        total_wmape = 100 * np.sum(np.abs(forecast - actual)) / total_abs_actual
    else:
        total_wmape = 0.0

    return total_wmape, wmape_by_horizon


def run_single_analysis(
    profile: ProfileConfig,
    noise: NoiseScenario,
    seed: int,
) -> Optional[HorizonResult]:
    """Run forecast for a single profile/noise/seed combination."""
    try:
        # Generate data with holdout and expected values
        train_df, holdout_df, expected_necf = generate_profile_data(profile, noise, seed)

        # Use actual holdout data (with noise) as ground truth
        actual_monthly = compute_monthly_actuals(holdout_df)
        if "necf" in actual_monthly.columns:
            actual_values = actual_monthly["necf"].tolist()
        else:
            # Fallback to expected values for holdout period (months 24-35)
            actual_values = expected_necf[24:36]

        # Ensure we have 12 months of actuals
        if len(actual_values) < 12:
            # Pad with expected values if needed
            for i in range(len(actual_values), 12):
                if i + 24 < len(expected_necf):
                    actual_values.append(expected_necf[i + 24])
                else:
                    actual_values.append(np.mean(actual_values) if actual_values else 0)

        actual_values = actual_values[:12]  # Exactly 12 months

        # Run forecast on training data
        engine = ForecastEngine(ForecastConfig())
        payload = engine.run_from_dataframe(train_df)

        forecast_values = [fr.forecast_total for fr in payload.forecast_results][:12]

        # Compute WMAPE by horizon
        total_wmape, wmape_by_horizon = compute_wmape_by_horizon(
            forecast_values, actual_values
        )

        return HorizonResult(
            profile=profile.name,
            noise_scenario=noise.name,
            seed=seed,
            wmape_total=total_wmape,
            wmape_by_horizon=wmape_by_horizon,
            forecast_values=forecast_values,
            actual_values=actual_values,
            model_selected=payload.model_selected,
        )
    except Exception as e:
        print(f"    Warning: {profile.name}/{noise.name}/seed={seed} failed: {e}")
        return None


def run_full_analysis(
    seeds: List[int] = None,
    profiles: Dict[str, ProfileConfig] = None,
    noise_scenarios: List[NoiseScenario] = None,
) -> List[HorizonResult]:
    """Run analysis across all profiles, noise levels, and seeds."""
    if seeds is None:
        seeds = list(range(42, 42 + 20))  # 20 seeds
    if profiles is None:
        profiles = PROFILES
    if noise_scenarios is None:
        noise_scenarios = NOISE_SCENARIOS

    results = []
    total = len(profiles) * len(noise_scenarios) * len(seeds)
    current = 0

    for profile_name, profile in profiles.items():
        print(f"\nProfile: {profile_name}")
        for noise in noise_scenarios:
            print(f"  Noise: {noise.name}...", end=" ", flush=True)
            count = 0
            for seed in seeds:
                current += 1
                result = run_single_analysis(profile, noise, seed)
                if result:
                    results.append(result)
                    count += 1
            print(f"({count}/{len(seeds)} successful)")

    return results


def plot_wmape_vs_horizon_by_profile(
    results: List[HorizonResult],
    output_path: str,
) -> None:
    """Plot WMAPE evolution over forecast horizon, one line per profile."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    horizons = list(range(1, 13))

    for ax, noise_name in zip(axes, ["Low", "Moderate", "High"]):
        for profile_name, color in COLORS.items():
            if profile_name in ["threshold", "actual"]:
                continue

            # Filter results
            profile_results = [
                r for r in results
                if r.profile == profile_name and r.noise_scenario == noise_name
            ]

            if not profile_results:
                continue

            # Aggregate WMAPE by horizon
            wmape_matrix = np.array([r.wmape_by_horizon for r in profile_results])
            mean_wmape = np.mean(wmape_matrix, axis=0)
            std_wmape = np.std(wmape_matrix, axis=0)

            # Plot
            ax.plot(horizons, mean_wmape, color=color, linewidth=2.5,
                   marker="o", markersize=6, label=profile_name)
            ax.fill_between(horizons, mean_wmape - std_wmape, mean_wmape + std_wmape,
                           color=color, alpha=0.2)

        # Threshold line
        ax.axhline(y=20, color=COLORS["threshold"], linewidth=2,
                  linestyle="--", alpha=0.7)

        ax.set_xlabel("Forecast Horizon (months)", fontsize=11, fontweight="bold")
        ax.set_title(f"{noise_name} Noise", fontsize=12, fontweight="bold")
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#FAFAFA")
        ax.set_xlim(0.5, 12.5)

    axes[0].set_ylabel("Cumulative WMAPE (%)", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=10)

    fig.suptitle("WMAPE Evolution Over Forecast Horizon by Customer Profile",
                fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_wmape_vs_horizon_by_noise(
    results: List[HorizonResult],
    output_path: str,
) -> None:
    """Plot WMAPE evolution over forecast horizon, one subplot per profile."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    horizons = list(range(1, 13))
    profile_names = ["Personal", "SME", "Corporate"]

    for ax, profile_name in zip(axes, profile_names):
        for noise_name, color in NOISE_COLORS.items():
            # Filter results
            noise_results = [
                r for r in results
                if r.profile == profile_name and r.noise_scenario == noise_name
            ]

            if not noise_results:
                continue

            # Aggregate WMAPE by horizon
            wmape_matrix = np.array([r.wmape_by_horizon for r in noise_results])
            mean_wmape = np.mean(wmape_matrix, axis=0)
            std_wmape = np.std(wmape_matrix, axis=0)

            # Plot
            ax.plot(horizons, mean_wmape, color=color, linewidth=2.5,
                   marker="s", markersize=6, label=f"{noise_name} Noise")
            ax.fill_between(horizons, mean_wmape - std_wmape, mean_wmape + std_wmape,
                           color=color, alpha=0.2)

        # Threshold line
        ax.axhline(y=20, color=COLORS["threshold"], linewidth=2,
                  linestyle="--", alpha=0.7, label="20% Threshold")

        ax.set_xlabel("Forecast Horizon (months)", fontsize=11, fontweight="bold")
        ax.set_title(f"{profile_name}", fontsize=12, fontweight="bold")
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#FAFAFA")
        ax.set_xlim(0.5, 12.5)

    axes[0].set_ylabel("Cumulative WMAPE (%)", fontsize=11, fontweight="bold")
    axes[2].legend(loc="upper left", fontsize=10)

    fig.suptitle("WMAPE Evolution Over Forecast Horizon by Noise Level",
                fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_heatmap_profile_noise(
    results: List[HorizonResult],
    output_path: str,
) -> None:
    """Plot heatmap of final WMAPE (month 12) by profile and noise."""
    fig, ax = plt.subplots(figsize=(10, 6))

    profile_names = ["Personal", "SME", "Corporate"]
    noise_names = ["Low", "Moderate", "High"]

    # Build matrix
    wmape_matrix = np.zeros((len(profile_names), len(noise_names)))

    for i, profile in enumerate(profile_names):
        for j, noise in enumerate(noise_names):
            matching = [
                r.wmape_total for r in results
                if r.profile == profile and r.noise_scenario == noise
            ]
            if matching:
                wmape_matrix[i, j] = np.mean(matching)

    # Create heatmap
    im = ax.imshow(wmape_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=40)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("WMAPE (%)", fontsize=11, fontweight="bold")

    # Add value annotations
    for i in range(len(profile_names)):
        for j in range(len(noise_names)):
            value = wmape_matrix[i, j]
            color = "white" if value > 25 else "black"
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center",
                   fontsize=14, fontweight="bold", color=color)

    # Labels
    ax.set_xticks(range(len(noise_names)))
    ax.set_xticklabels(noise_names, fontsize=11)
    ax.set_yticks(range(len(profile_names)))
    ax.set_yticklabels(profile_names, fontsize=11)

    ax.set_xlabel("Noise Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Customer Profile", fontsize=12, fontweight="bold")
    ax.set_title("12-Month Forecast WMAPE by Profile and Noise Level",
                fontsize=14, fontweight="bold", pad=20)

    # Add threshold annotation
    ax.axhline(y=-0.5, color="white", linewidth=0)
    fig.text(0.98, 0.02, "Green = Below 20% threshold | Red = Above threshold",
            fontsize=9, ha="right", style="italic", color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_step_wmape_bars(
    results: List[HorizonResult],
    output_path: str,
) -> None:
    """Plot per-step (not cumulative) WMAPE as grouped bar chart."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    profile_names = ["Personal", "SME", "Corporate"]
    horizons = np.arange(1, 13)
    width = 0.25

    for ax, profile_name in zip(axes, profile_names):
        for i, (noise_name, color) in enumerate(NOISE_COLORS.items()):
            # Filter results
            noise_results = [
                r for r in results
                if r.profile == profile_name and r.noise_scenario == noise_name
            ]

            if not noise_results:
                continue

            # Compute per-step (not cumulative) error
            step_wmapes = []
            for h in range(12):
                step_errors = []
                for r in noise_results:
                    if h < len(r.forecast_values) and h < len(r.actual_values):
                        f = r.forecast_values[h]
                        a = r.actual_values[h]
                        if abs(a) > 0:
                            step_errors.append(100 * abs(f - a) / abs(a))
                if step_errors:
                    step_wmapes.append(np.mean(step_errors))
                else:
                    step_wmapes.append(0)

            offset = (i - 1) * width
            ax.bar(horizons + offset, step_wmapes, width, color=color,
                  label=f"{noise_name} Noise", alpha=0.8, edgecolor="white")

        # Threshold line
        ax.axhline(y=20, color=COLORS["threshold"], linewidth=2,
                  linestyle="--", alpha=0.7)

        ax.set_ylabel("Step WMAPE (%)", fontsize=10, fontweight="bold")
        ax.set_title(f"{profile_name}", fontsize=11, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#FAFAFA")
        ax.set_ylim(0, None)

    axes[-1].set_xlabel("Forecast Horizon (months)", fontsize=11, fontweight="bold")
    axes[-1].set_xticks(horizons)
    axes[0].legend(loc="upper right", fontsize=9)

    fig.suptitle("Per-Step WMAPE by Profile and Noise Level\n(Error at each individual month)",
                fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_forecast_trajectories(
    results: List[HorizonResult],
    output_path: str,
) -> None:
    """Plot example forecast vs actual trajectories."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    profile_names = ["Personal", "SME", "Corporate"]
    noise_names = ["Low", "Moderate", "High"]
    horizons = list(range(1, 13))

    for i, profile_name in enumerate(profile_names):
        for j, noise_name in enumerate(noise_names):
            ax = axes[i, j]

            # Get first result for this combination
            matching = [
                r for r in results
                if r.profile == profile_name and r.noise_scenario == noise_name
            ]

            if not matching:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                       transform=ax.transAxes)
                continue

            # Use first result as example
            r = matching[0]

            # Normalize for comparison (divide by mean actual)
            mean_actual = np.mean(np.abs(r.actual_values)) if r.actual_values else 1
            norm_forecast = np.array(r.forecast_values) / mean_actual
            norm_actual = np.array(r.actual_values) / mean_actual

            ax.plot(horizons, norm_actual, color=COLORS["actual"], linewidth=2,
                   marker="o", markersize=5, label="Actual")
            ax.plot(horizons, norm_forecast, color=COLORS[profile_name], linewidth=2,
                   marker="s", markersize=5, label="Forecast")

            # Fill error area
            ax.fill_between(horizons, norm_forecast, norm_actual,
                           color=COLORS["threshold"], alpha=0.2)

            ax.set_title(f"{profile_name} - {noise_name}\nWMAPE: {r.wmape_total:.1f}%",
                        fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#FAFAFA")

            if i == 2:
                ax.set_xlabel("Month", fontsize=9)
            if j == 0:
                ax.set_ylabel("Normalized Value", fontsize=9)
            if i == 0 and j == 2:
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Forecast vs Actual Trajectories (Normalized)\n"
                "Seed 42 Example for Each Profile/Noise Combination",
                fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(
    results: List[HorizonResult],
    output_path: str,
) -> None:
    """Generate summary CSV and table plot."""
    rows = []

    for profile_name in ["Personal", "SME", "Corporate"]:
        for noise_name in ["Low", "Moderate", "High"]:
            matching = [
                r for r in results
                if r.profile == profile_name and r.noise_scenario == noise_name
            ]

            if not matching:
                continue

            wmapes = [r.wmape_total for r in matching]
            passes = sum(1 for w in wmapes if w < 20)

            # Get WMAPE at specific horizons
            wmape_3m = np.mean([r.wmape_by_horizon[2] for r in matching if len(r.wmape_by_horizon) > 2])
            wmape_6m = np.mean([r.wmape_by_horizon[5] for r in matching if len(r.wmape_by_horizon) > 5])
            wmape_12m = np.mean([r.wmape_by_horizon[11] for r in matching if len(r.wmape_by_horizon) > 11])

            rows.append({
                "Profile": profile_name,
                "Noise": noise_name,
                "WMAPE 3M": f"{wmape_3m:.1f}%",
                "WMAPE 6M": f"{wmape_6m:.1f}%",
                "WMAPE 12M": f"{wmape_12m:.1f}%",
                "Mean": f"{np.mean(wmapes):.1f}%",
                "Std": f"{np.std(wmapes):.1f}%",
                "Pass Rate": f"{100 * passes / len(matching):.0f}%",
            })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = output_path.replace(".png", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Create table plot
    fig, ax = plt.subplots(figsize=(14, 8))
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
        # Color code pass rate column
        elif key[1] == 7:  # Pass Rate column
            val = df.iloc[key[0] - 1]["Pass Rate"]
            pct = int(val.replace("%", ""))
            if pct >= 80:
                cell.set_facecolor("#C8E6C9")
            elif pct >= 50:
                cell.set_facecolor("#FFF9C4")
            else:
                cell.set_facecolor("#FFCDD2")

    plt.title("Profile WMAPE Analysis Summary",
             fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Run complete profile-based WMAPE analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile-based WMAPE analysis across forecast horizons"
    )
    parser.add_argument(
        "--seeds", "-s",
        type=int,
        default=20,
        help="Number of random seeds to use (default: 20)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: plots/profile_analysis)"
    )
    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "plots" / "profile_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate seeds
    seeds = list(range(42, 42 + args.seeds))

    print("=" * 70)
    print("PROFILE-BASED WMAPE ANALYSIS")
    print("=" * 70)
    print(f"\nProfiles: Personal, SME, Corporate")
    print(f"Noise levels: Low, Moderate, High")
    print(f"Seeds: {args.seeds}")
    print(f"Total runs: {3 * 3 * args.seeds}")

    # Run analysis
    print("\n--- Running Analysis ---")
    results = run_full_analysis(seeds=seeds)

    print(f"\n--- Generating Plots ({len(results)} successful runs) ---")

    # Plot 1: WMAPE vs Horizon by Profile
    plot_wmape_vs_horizon_by_profile(
        results, str(output_dir / "wmape_horizon_by_profile.png")
    )

    # Plot 2: WMAPE vs Horizon by Noise
    plot_wmape_vs_horizon_by_noise(
        results, str(output_dir / "wmape_horizon_by_noise.png")
    )

    # Plot 3: Heatmap
    plot_heatmap_profile_noise(
        results, str(output_dir / "wmape_heatmap.png")
    )

    # Plot 4: Per-step WMAPE bars
    plot_step_wmape_bars(
        results, str(output_dir / "wmape_per_step_bars.png")
    )

    # Plot 5: Forecast trajectories
    plot_forecast_trajectories(
        results, str(output_dir / "forecast_trajectories.png")
    )

    # Summary table
    generate_summary_table(
        results, str(output_dir / "summary_table.png")
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")

    # Print quick summary
    print("\n--- Quick Summary ---")
    for profile in ["Personal", "SME", "Corporate"]:
        profile_results = [r for r in results if r.profile == profile]
        if profile_results:
            wmapes = [r.wmape_total for r in profile_results]
            passes = sum(1 for w in wmapes if w < 20)
            print(f"{profile}: WMAPE = {np.mean(wmapes):.1f}% +/- {np.std(wmapes):.1f}%, "
                  f"Pass Rate = {100 * passes / len(wmapes):.0f}%")


if __name__ == "__main__":
    main()
