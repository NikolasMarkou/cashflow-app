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
from typing import List, Dict, Tuple, Optional, Set
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
    flag_corruption_rate: float = 0.0  # Probability of wrong is_recurring_flag
    salary_raise_month: int = 0  # Month to apply salary raise (0 = no raise)
    # Fat-tailed distribution support (Phase 2.1)
    noise_distribution: str = "gaussian"  # "gaussian" | "student_t" | "laplace" | "mixture"
    df_param: float = 5.0  # Degrees of freedom for Student-t (lower = heavier tails)
    mixture_extreme_prob: float = 0.05  # Probability of extreme event in mixture model
    # Regime shift support (Phase 2.2)
    salary_change_amount: float = 500.0  # Amount of salary change (positive = raise, negative = cut)
    rent_change_month: int = 0  # Month to change rent (0 = no change)
    rent_change_amount: float = 0.0  # Amount of rent change
    category_extinction_month: int = 0  # Month when a category stops (0 = no extinction)
    category_extinction_type: str = ""  # Category to stop ("subscription", "loan", etc.)
    second_shift_month: int = 0  # Month for second regime shift (for multiple shifts)
    second_shift_amount: float = 0.0  # Amount for second shift


def sample_noise(std: float, config: NoiseConfig) -> float:
    """Sample noise from the configured distribution.

    Args:
        std: Standard deviation (or scale) parameter
        config: NoiseConfig with distribution settings

    Returns:
        A single noise sample
    """
    if std <= 0:
        return 0.0

    dist = config.noise_distribution

    if dist == "gaussian":
        return np.random.normal(0, std)

    elif dist == "student_t":
        # Student-t with df degrees of freedom, scaled to match std
        # Variance of t(df) = df / (df - 2) for df > 2
        df = config.df_param
        if df > 2:
            scale = std * np.sqrt((df - 2) / df)
        else:
            scale = std  # For df <= 2, variance is infinite, use std as scale
        return np.random.standard_t(df) * scale

    elif dist == "laplace":
        # Laplace (double exponential) - common in finance
        # Variance = 2 * scale^2, so scale = std / sqrt(2)
        scale = std / np.sqrt(2)
        return np.random.laplace(0, scale)

    elif dist == "mixture":
        # Mixture: 95% normal + 5% extreme events (3x std)
        if np.random.random() < config.mixture_extreme_prob:
            # Extreme event - 3x magnitude
            return np.random.normal(0, std * 3)
        else:
            return np.random.normal(0, std)

    else:
        # Fallback to Gaussian
        return np.random.normal(0, std)


# Define noise levels from clean to very noisy
# Added flag_corruption_rate and salary_raise to test improvements
NOISE_LEVELS_GAUSSIAN = [
    NoiseConfig("Baseline (No Noise)", 0, 0, 0, 1.0, 0, 0, 0.0, 0),
    NoiseConfig("Very Low Noise", 25, 10, 20, 1.0, 0.05, 200, 0.1, 0),  # 10% flag corruption
    NoiseConfig("Low Noise", 50, 20, 40, 1.2, 0.10, 400, 0.2, 12),  # 20% flag corruption + raise at month 12
    NoiseConfig("Moderate Noise", 100, 40, 60, 1.5, 0.15, 600, 0.3, 12),  # 30% flag corruption + raise
    NoiseConfig("High Noise", 200, 80, 100, 2.0, 0.20, 1000, 0.4, 12),  # 40% flag corruption + raise
]

# Phase 2.1: Fat-tailed distribution configurations
# Student-t distributions (heavy tails - common in financial data)
NOISE_LEVELS_STUDENT_T = [
    NoiseConfig("Student-t (df=3)", 100, 40, 60, 1.5, 0.15, 600, 0.2, 12,
                noise_distribution="student_t", df_param=3.0),  # Very heavy tails
    NoiseConfig("Student-t (df=5)", 100, 40, 60, 1.5, 0.15, 600, 0.2, 12,
                noise_distribution="student_t", df_param=5.0),  # Moderate heavy tails
    NoiseConfig("Student-t (df=10)", 100, 40, 60, 1.5, 0.15, 600, 0.2, 12,
                noise_distribution="student_t", df_param=10.0),  # Light heavy tails
]

# Laplace distribution (double exponential - sharp peak, heavy tails)
NOISE_LEVELS_LAPLACE = [
    NoiseConfig("Laplace Low", 50, 20, 40, 1.2, 0.10, 400, 0.2, 12,
                noise_distribution="laplace"),
    NoiseConfig("Laplace Moderate", 100, 40, 60, 1.5, 0.15, 600, 0.2, 12,
                noise_distribution="laplace"),
    NoiseConfig("Laplace High", 200, 80, 100, 2.0, 0.20, 1000, 0.3, 12,
                noise_distribution="laplace"),
]

# Mixture models (normal + rare extreme events)
NOISE_LEVELS_MIXTURE = [
    NoiseConfig("Mixture (5% extreme)", 100, 40, 60, 1.5, 0.15, 600, 0.2, 12,
                noise_distribution="mixture", mixture_extreme_prob=0.05),
    NoiseConfig("Mixture (10% extreme)", 100, 40, 60, 1.5, 0.15, 600, 0.2, 12,
                noise_distribution="mixture", mixture_extreme_prob=0.10),
    NoiseConfig("Mixture (15% extreme)", 100, 40, 60, 1.5, 0.15, 600, 0.2, 12,
                noise_distribution="mixture", mixture_extreme_prob=0.15),
]

# Phase 2.2: Regime shift configurations
NOISE_LEVELS_REGIME_SHIFT = [
    # Baseline with subscription and loan (no shifts)
    NoiseConfig("Regime: Baseline", 50, 20, 40, 1.2, 0.10, 400, 0.1, 0),

    # Positive shift: Promotion/salary increase at month 12
    NoiseConfig("Regime: Salary Raise", 50, 20, 40, 1.2, 0.10, 400, 0.1, 12,
                salary_change_amount=500),

    # Negative shift: Job loss / reduced hours at month 12
    NoiseConfig("Regime: Salary Cut", 50, 20, 40, 1.2, 0.10, 400, 0.1, 12,
                salary_change_amount=-800),

    # Multiple shifts: Raise at month 8, then cut at month 18
    NoiseConfig("Regime: Multiple Shifts", 50, 20, 40, 1.2, 0.10, 400, 0.1, 8,
                salary_change_amount=500, second_shift_month=18, second_shift_amount=-1000),

    # Recent shift: Shift at month 22 (only 2 months of post-shift data)
    NoiseConfig("Regime: Recent Shift", 50, 20, 40, 1.2, 0.10, 400, 0.1, 22,
                salary_change_amount=600),

    # Category extinction: Subscription cancelled at month 15
    NoiseConfig("Regime: Sub Cancelled", 50, 20, 40, 1.2, 0.10, 400, 0.1, 0,
                category_extinction_month=15, category_extinction_type="subscription"),

    # Category extinction: Loan paid off at month 12
    NoiseConfig("Regime: Loan Paid Off", 50, 20, 40, 1.2, 0.10, 400, 0.1, 0,
                category_extinction_month=12, category_extinction_type="loan"),

    # Rent increase at month 12
    NoiseConfig("Regime: Rent Increase", 50, 20, 40, 1.2, 0.10, 400, 0.1, 0,
                rent_change_month=12, rent_change_amount=200),

    # Combined: Salary raise + rent increase (lifestyle upgrade)
    NoiseConfig("Regime: Lifestyle Change", 50, 20, 40, 1.2, 0.10, 400, 0.1, 12,
                salary_change_amount=800, rent_change_month=12, rent_change_amount=300),
]

# Default: Gaussian only (backward compatible)
NOISE_LEVELS = NOISE_LEVELS_GAUSSIAN

# All distributions combined for comprehensive analysis
ALL_NOISE_LEVELS = (
    NOISE_LEVELS_GAUSSIAN +
    NOISE_LEVELS_STUDENT_T +
    NOISE_LEVELS_LAPLACE +
    NOISE_LEVELS_MIXTURE
)

# All regime shift scenarios for Phase 2.2 testing
ALL_REGIME_SHIFTS = NOISE_LEVELS_REGIME_SHIFT


# Phase 2.3: Transfer tolerance sweep configurations
@dataclass
class TransferConfig:
    """Configuration for transfer delay scenarios."""
    name: str
    domestic_delay_range: Tuple[int, int]  # Min/max delay in days for domestic transfers
    international_delay_range: Tuple[int, int]  # Min/max delay for international
    domestic_ratio: float  # Ratio of domestic transfers (0-1)


TRANSFER_CONFIGS = [
    TransferConfig("Same-Day Transfers", (0, 0), (0, 0), 1.0),
    TransferConfig("Domestic Only (0-2d)", (0, 2), (0, 2), 1.0),
    TransferConfig("Mixed (Domestic + International)", (0, 2), (3, 5), 0.7),
    TransferConfig("International Heavy (3-5d)", (3, 5), (3, 5), 0.3),
    TransferConfig("High Variance (0-7d)", (0, 7), (0, 7), 0.5),
]


def generate_transfer_data(
    transfer_config: TransferConfig,
    num_transfers: int = 24,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic data with configurable transfer delays.

    Args:
        transfer_config: Configuration for transfer delays
        num_transfers: Number of transfer pairs to generate (default: 24 for 2 years monthly)
        seed: Random seed

    Returns:
        DataFrame with transactions including transfer pairs with specified delays
    """
    np.random.seed(seed)
    transactions = []
    tx_id = 1

    # Base transactions (non-transfers)
    base_salary = 3000
    base_rent = 1200

    for month_idx in range(24):
        year = 2024 + month_idx // 12
        month = (month_idx % 12) + 1
        month_start = datetime(year, month, 1)

        # Salary
        transactions.append({
            "tx_id": f"TX{tx_id:06d}",
            "customer_id": "CUST001",
            "account_id": "MAIN_CHECKING",
            "tx_date": month_start,
            "amount": base_salary + np.random.normal(0, 50),
            "currency": "EUR",
            "direction": "CREDIT",
            "category": "SALARY",
            "description_raw": f"SALARY {year}-{month:02d}",
            "is_recurring_flag": True,
            "is_variable_amount": False,
            "is_transfer": False,
            "transfer_pair_id": None,
            "actual_delay": None,
        })
        tx_id += 1

        # Rent
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
            "is_transfer": False,
            "transfer_pair_id": None,
            "actual_delay": None,
        })
        tx_id += 1

        # Transfer pair with configurable delay
        is_domestic = np.random.random() < transfer_config.domestic_ratio
        if is_domestic:
            delay_range = transfer_config.domestic_delay_range
        else:
            delay_range = transfer_config.international_delay_range

        actual_delay = np.random.randint(delay_range[0], delay_range[1] + 1)
        transfer_amount = 500
        transfer_pair_id = f"PAIR{month_idx:03d}"

        # Outgoing transfer - use neutral category to test amount+date matching
        # (not TRANSFER_OUT which would be detected by category heuristics)
        out_date = datetime(year, month, 15)
        transactions.append({
            "tx_id": f"TX{tx_id:06d}",
            "customer_id": "CUST001",
            "account_id": "MAIN_CHECKING",
            "tx_date": out_date,
            "amount": -transfer_amount,
            "currency": "EUR",
            "direction": "DEBIT",
            "category": "MISCELLANEOUS",  # Neutral category - tests amount+date matching
            "description_raw": "ACCOUNT TRANSFER",
            "is_recurring_flag": False,
            "is_variable_amount": False,
            "is_transfer": True,
            "transfer_pair_id": transfer_pair_id,
            "actual_delay": actual_delay,
        })
        tx_id += 1

        # Incoming transfer (with delay)
        in_date = out_date + pd.Timedelta(days=actual_delay)
        transactions.append({
            "tx_id": f"TX{tx_id:06d}",
            "customer_id": "CUST001",
            "account_id": "SAVINGS",
            "tx_date": in_date,
            "amount": transfer_amount,
            "currency": "EUR",
            "direction": "CREDIT",
            "category": "MISCELLANEOUS",  # Neutral category - tests amount+date matching
            "description_raw": "ACCOUNT TRANSFER",
            "is_recurring_flag": False,
            "is_variable_amount": False,
            "is_transfer": True,
            "transfer_pair_id": transfer_pair_id,
            "actual_delay": actual_delay,
        })
        tx_id += 1

    df = pd.DataFrame(transactions)
    df["tx_date"] = pd.to_datetime(df["tx_date"])
    return df


def evaluate_transfer_detection(
    df: pd.DataFrame,
    tolerance_days: int
) -> Dict[str, float]:
    """Evaluate transfer detection accuracy at a given tolerance.

    Args:
        df: DataFrame with ground truth transfer markers
        tolerance_days: Tolerance setting for transfer detection

    Returns:
        Dict with precision, recall, f1, and other metrics
    """
    from cashflow.pipeline import clean_utf, detect_transfers

    # Get ground truth
    ground_truth_transfers = set(df[df["is_transfer"] == True]["tx_id"].tolist())
    total_true_transfers = len(ground_truth_transfers)

    # Clean and detect transfers
    df_clean = df.drop(columns=["is_transfer", "transfer_pair_id", "actual_delay"], errors="ignore")
    df_clean = clean_utf(df_clean)
    df_detected = detect_transfers(df_clean, date_tolerance_days=tolerance_days)

    # Get detected transfers
    detected_transfers = set(
        df_detected[df_detected["is_internal_transfer"] == True]["tx_id"].tolist()
    )
    total_detected = len(detected_transfers)

    # Calculate metrics
    true_positives = len(ground_truth_transfers & detected_transfers)
    false_positives = len(detected_transfers - ground_truth_transfers)
    false_negatives = len(ground_truth_transfers - detected_transfers)

    precision = true_positives / total_detected if total_detected > 0 else 0
    recall = true_positives / total_true_transfers if total_true_transfers > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tolerance_days": tolerance_days,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_true": total_true_transfers,
        "total_detected": total_detected,
    }


def run_transfer_tolerance_sweep(
    transfer_config: TransferConfig,
    tolerance_range: List[int] = None,
    seeds: List[int] = None
) -> pd.DataFrame:
    """Run transfer detection sweep across tolerance values.

    Args:
        transfer_config: Transfer delay configuration
        tolerance_range: List of tolerance values to test
        seeds: Random seeds to use

    Returns:
        DataFrame with results for each tolerance/seed combination
    """
    if tolerance_range is None:
        tolerance_range = [0, 1, 2, 3, 5, 7]
    if seeds is None:
        seeds = list(range(10))

    results = []

    for tolerance in tolerance_range:
        for seed in seeds:
            df = generate_transfer_data(transfer_config, seed=seed)
            metrics = evaluate_transfer_detection(df, tolerance)
            metrics["config_name"] = transfer_config.name
            metrics["seed"] = seed
            results.append(metrics)

    return pd.DataFrame(results)


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

    # Track month number for salary raise
    month_number = 0

    # Generate 24 months of data
    for year in [2024, 2025]:
        for month in range(1, 13):
            if year == 2025 and month > 12:
                break

            month_number += 1
            month_start = datetime(year, month, 1)

            # Apply salary change if configured (supports both raises and cuts)
            current_base_salary = base_salary
            if noise_config.salary_raise_month > 0 and month_number >= noise_config.salary_raise_month:
                current_base_salary = base_salary + noise_config.salary_change_amount
            # Apply second shift if configured (for multiple shifts scenario)
            if noise_config.second_shift_month > 0 and month_number >= noise_config.second_shift_month:
                current_base_salary = current_base_salary + noise_config.second_shift_amount

            # Salary (recurring income) - with noise from configured distribution
            salary_noise = sample_noise(noise_config.salary_std, noise_config)
            salary = current_base_salary + salary_noise

            # Corrupt flag with configured probability
            salary_flag = True
            if np.random.random() < noise_config.flag_corruption_rate:
                salary_flag = False  # Wrong flag!

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
                "is_recurring_flag": salary_flag,
                "is_variable_amount": False,
            })
            tx_id += 1

            # Rent (recurring expense) - with optional rent change
            current_rent = base_rent
            if noise_config.rent_change_month > 0 and month_number >= noise_config.rent_change_month:
                current_rent = base_rent + noise_config.rent_change_amount

            rent_flag = True
            if np.random.random() < noise_config.flag_corruption_rate:
                rent_flag = False

            transactions.append({
                "tx_id": f"TX{tx_id:06d}",
                "customer_id": "CUST001",
                "account_id": "MAIN_CHECKING",
                "tx_date": month_start,
                "amount": -current_rent,
                "currency": "EUR",
                "direction": "DEBIT",
                "category": "RENT_MORTGAGE",
                "description_raw": f"RENT {year}-{month:02d}",
                "is_recurring_flag": rent_flag,
                "is_variable_amount": False,
            })
            tx_id += 1

            # Subscription (recurring expense) - can be extinct
            subscription_active = True
            if noise_config.category_extinction_type == "subscription":
                if noise_config.category_extinction_month > 0 and month_number >= noise_config.category_extinction_month:
                    subscription_active = False

            if subscription_active:
                sub_flag = True
                if np.random.random() < noise_config.flag_corruption_rate:
                    sub_flag = False

                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN_CHECKING",
                    "tx_date": datetime(year, month, 10),
                    "amount": -50,  # €50/month subscription
                    "currency": "EUR",
                    "direction": "DEBIT",
                    "category": "SUBSCRIPTION",
                    "description_raw": "STREAMING SERVICE",
                    "is_recurring_flag": sub_flag,
                    "is_variable_amount": False,
                })
                tx_id += 1

            # Loan payment (recurring expense) - can be extinct (loan paid off)
            loan_active = True
            if noise_config.category_extinction_type == "loan":
                if noise_config.category_extinction_month > 0 and month_number >= noise_config.category_extinction_month:
                    loan_active = False

            if loan_active:
                loan_flag = True
                if np.random.random() < noise_config.flag_corruption_rate:
                    loan_flag = False

                transactions.append({
                    "tx_id": f"TX{tx_id:06d}",
                    "customer_id": "CUST001",
                    "account_id": "MAIN_CHECKING",
                    "tx_date": datetime(year, month, 15),
                    "amount": -300,  # €300/month loan payment
                    "currency": "EUR",
                    "direction": "DEBIT",
                    "category": "LOAN_PAYMENT",
                    "description_raw": "CAR LOAN",
                    "is_recurring_flag": loan_flag,
                    "is_variable_amount": False,
                })
                tx_id += 1

            # Utilities (seasonal variation + noise from configured distribution)
            winter_factor = 1.5 if month in [11, 12, 1, 2] else 1.0
            utility_noise = sample_noise(noise_config.expense_std, noise_config)
            utilities = -(base_utilities * winter_factor + utility_noise)

            utility_flag = True
            if np.random.random() < noise_config.flag_corruption_rate:
                utility_flag = False

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
                "is_recurring_flag": utility_flag,
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
    historical_values: List[float]
    forecast_values: List[float]
    forecast_lower: List[float]
    forecast_upper: List[float]


def analyze_noise_level(noise_config: NoiseConfig, seed: int = 42) -> NoiseResult:
    """Run analysis for a single noise level."""
    utf_df = generate_synthetic_data(noise_config, seed=seed)
    payload, historical_df = run_forecast(utf_df)

    # Calculate metrics
    forecast_values = [fr.forecast_total for fr in payload.forecast_results]
    forecast_lower = [fr.lower_ci for fr in payload.forecast_results]
    forecast_upper = [fr.upper_ci for fr in payload.forecast_results]
    ci_widths = [fr.upper_ci - fr.lower_ci for fr in payload.forecast_results]
    historical_values = historical_df["necf"].tolist()

    return NoiseResult(
        noise_config=noise_config,
        wmape=payload.wmape_winner,
        model_selected=payload.model_selected,
        meets_threshold=payload.meets_threshold,
        num_outliers=len(payload.outliers_detected),
        avg_ci_width=np.mean(ci_widths),
        historical_std=historical_df["necf"].std(),
        historical_values=historical_values,
        forecast_values=forecast_values,
        forecast_lower=forecast_lower,
        forecast_upper=forecast_upper,
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
    """Plot forecast trajectories for different noise levels with stacked subplots."""
    noise_levels = list(results.keys())
    n_levels = len(noise_levels)
    colors = COLORS["noise_gradient"][:n_levels]

    fig, axes = plt.subplots(n_levels, 1, figsize=(14, 3 * n_levels), sharex=True)

    # Handle single subplot case
    if n_levels == 1:
        axes = [axes]

    for i, (level, color, ax) in enumerate(zip(noise_levels, colors, axes)):
        if not results[level]:
            continue

        # Use first seed result
        result = results[level][0]
        historical = result.historical_values
        forecast = result.forecast_values
        forecast_lower = result.forecast_lower
        forecast_upper = result.forecast_upper

        n_hist = len(historical)
        n_fore = len(forecast)

        # Create x-axis: historical months (negative) + forecast months (positive)
        hist_x = list(range(-n_hist + 1, 1))  # ..., -2, -1, 0
        fore_x = list(range(1, n_fore + 1))   # 1, 2, ..., 12

        # Plot historical data
        ax.plot(hist_x, historical, color=COLORS["actual"], linewidth=2,
               marker="o", markersize=4, label="Historical")

        # Plot forecast with confidence interval
        ax.plot(fore_x, forecast, color=color, linewidth=2.5,
               marker="s", markersize=5, label=f"Forecast (WMAPE: {result.wmape:.1f}%)")
        ax.fill_between(fore_x, forecast_lower, forecast_upper,
                       color=color, alpha=0.2, label="95% CI")

        # Vertical line at forecast start
        ax.axvline(x=0.5, color="#6C757D", linestyle="--", linewidth=1.5, alpha=0.7)

        # Zero line
        ax.axhline(y=0, color="#6C757D", linewidth=0.5, alpha=0.5)

        # Styling
        ax.set_ylabel("EUR", fontsize=10, fontweight="bold")
        ax.set_title(f"{level}", fontsize=11, fontweight="bold", loc="left")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#FAFAFA")

    # X-axis label on bottom subplot only
    axes[-1].set_xlabel("Month (negative = historical, positive = forecast)", fontsize=11, fontweight="bold")

    # Main title
    fig.suptitle("Historical Data + Forecast Trajectories Under Different Noise Levels",
                fontsize=14, fontweight="bold", y=1.01)

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
    import argparse

    parser = argparse.ArgumentParser(description="Noise sensitivity analysis for cashflow forecasting")
    parser.add_argument(
        "--distribution", "-d",
        choices=["gaussian", "student_t", "laplace", "mixture", "regime_shift", "all"],
        default="gaussian",
        help="Distribution type to analyze (default: gaussian)"
    )
    parser.add_argument(
        "--seeds", "-s",
        type=int,
        default=30,
        help="Number of random seeds to use (default: 30)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: plots/noise_analysis)"
    )
    args = parser.parse_args()

    # Select noise levels based on distribution type
    if args.distribution == "gaussian":
        noise_levels = NOISE_LEVELS_GAUSSIAN
        suffix = ""
    elif args.distribution == "student_t":
        noise_levels = NOISE_LEVELS_STUDENT_T
        suffix = "_student_t"
    elif args.distribution == "laplace":
        noise_levels = NOISE_LEVELS_LAPLACE
        suffix = "_laplace"
    elif args.distribution == "mixture":
        noise_levels = NOISE_LEVELS_MIXTURE
        suffix = "_mixture"
    elif args.distribution == "regime_shift":
        noise_levels = NOISE_LEVELS_REGIME_SHIFT
        suffix = "_regime_shift"
    elif args.distribution == "all":
        noise_levels = ALL_NOISE_LEVELS
        suffix = "_all"
    else:
        noise_levels = NOISE_LEVELS_GAUSSIAN
        suffix = ""

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "plots" / "noise_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate seed list
    base_seeds = [42, 123, 456, 789, 2024, 1337, 9999, 5555, 7777, 3141,
                  1111, 2222, 3333, 4444, 6666, 8888, 1234, 5678, 9012, 3456,
                  1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 1010]
    seeds = base_seeds[:args.seeds]

    print("=" * 60)
    print("NOISE SENSITIVITY ANALYSIS")
    print(f"Distribution: {args.distribution.upper()}")
    print("=" * 60)

    print(f"\nAnalyzing {len(noise_levels)} noise levels with {len(seeds)} random seeds each...")

    # Override global NOISE_LEVELS for run_noise_analysis
    global NOISE_LEVELS
    original_levels = NOISE_LEVELS
    NOISE_LEVELS = noise_levels

    results = run_noise_analysis(seeds=seeds)

    # Restore original
    NOISE_LEVELS = original_levels

    print("\nGenerating plots...")

    # Plot 1: WMAPE vs Noise
    plot_wmape_vs_noise(results, str(output_dir / f"wmape_vs_noise{suffix}.png"))

    # Plot 2: Forecast comparison
    plot_forecast_comparison(results, str(output_dir / f"forecast_trajectories{suffix}.png"))

    # Plot 3: CI width expansion
    plot_ci_width_vs_noise(results, str(output_dir / f"ci_width_vs_noise{suffix}.png"))

    # Plot 4: Outlier detection rate
    plot_outlier_detection_rate(results, str(output_dir / f"outlier_detection{suffix}.png"))

    # Plot 5: Threshold pass rate
    plot_threshold_pass_rate(results, str(output_dir / f"threshold_pass_rate{suffix}.png"))

    # Generate summary table
    generate_summary_table(results, str(output_dir / f"summary_table{suffix}.png"))

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")

    # Print summary
    print("\n--- Quick Summary ---")
    for level in results.keys():
        if results[level]:
            wmapes = [r.wmape for r in results[level]]
            passes = sum(1 for r in results[level] if r.meets_threshold)
            pass_rate = 100 * passes / len(results[level])
            print(f"{level}: WMAPE = {np.mean(wmapes):.2f}% ± {np.std(wmapes):.2f}%, Pass Rate = {pass_rate:.0f}%")


if __name__ == "__main__":
    main()
