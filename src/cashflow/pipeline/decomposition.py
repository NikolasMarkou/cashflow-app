"""Cash flow decomposition - SDD Section 10.

Enhanced with trend-adjusted projection to fix the "Mean Fallacy" where
using historical mean fails to capture recent lifestyle changes
(salary raises, rent changes, etc.).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DeterministicProjection:
    """Holds deterministic base projection with trend information.

    Instead of a single scalar mean, this captures:
    - Base value (intercept)
    - Monthly trend (slope)
    - Confidence in the projection
    """
    base_value: float
    monthly_trend: float
    confidence: float
    method: str

    def project(self, months_ahead: int) -> float:
        """Project deterministic base for future month."""
        return self.base_value + (self.monthly_trend * months_ahead)

    def project_series(self, num_months: int) -> list[float]:
        """Project series of future deterministic values."""
        return [self.project(i) for i in range(1, num_months + 1)]


def decompose_cashflow(
    monthly_df: pd.DataFrame,
    transactions_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Decompose NECF into deterministic base and residual components.

    Per SDD Section 10.2:
    NECF = Deterministic Base + Residual

    Deterministic Base includes:
    - Salary
    - Contractual loans
    - Installment plans
    - Standing orders
    - Direct debits with known schedules

    Residual represents:
    - Discretionary spending
    - Variable utilities
    - One-off events
    - Behavioral volatility

    Args:
        monthly_df: Monthly NECF DataFrame
        transactions_df: Optional original transaction data for decomposition

    Returns:
        DataFrame with deterministic_base and residual columns added
    """
    df = monthly_df.copy()

    if len(df) == 0:
        df["deterministic_base"] = 0.0
        df["residual"] = 0.0
        return df

    if transactions_df is not None and len(transactions_df) > 0:
        # Full decomposition using transaction-level data
        df = _decompose_from_transactions(df, transactions_df)
    else:
        # Approximate decomposition using recurring pattern detection
        df = _decompose_approximate(df)

    # Validate integrity constraint: |NECF - (Deterministic + Residual)| < ε
    _validate_decomposition(df)

    logger.info("Cash flow decomposition completed")

    return df


# Minimum recurring value ratio to consider flags reliable
MIN_RECURRING_VALUE_RATIO = 0.15  # 15% of absolute value should be recurring
# Maximum coefficient of variation for recurring transactions to be considered reliable
MAX_RECURRING_CV = 0.50  # 50% CV threshold - higher means more variance = potentially corrupted


def _calculate_recurring_stability(tx_df: pd.DataFrame, recurring_mask: pd.Series) -> float:
    """Calculate stability score for transactions marked as recurring.

    Truly recurring transactions should produce consistent monthly totals.
    High month-to-month variance indicates corrupted flags (wrong transactions
    marked as recurring).

    Returns:
        Stability score (0-1, higher = more stable monthly totals)
    """
    recurring_tx = tx_df[recurring_mask].copy()

    if len(recurring_tx) < 3:
        return 1.0  # Not enough data to assess

    # Ensure month_key exists
    if "month_key" not in recurring_tx.columns:
        if "tx_date" in recurring_tx.columns:
            recurring_tx["month_key"] = pd.to_datetime(recurring_tx["tx_date"]).dt.strftime("%Y-%m")
        else:
            return 1.0  # Can't compute monthly stability

    # Calculate monthly totals (this is what becomes deterministic_base)
    monthly_totals = recurring_tx.groupby("month_key")["amount"].sum()

    if len(monthly_totals) < 3:
        return 1.0  # Not enough months to assess

    # Calculate coefficient of variation of monthly totals
    mean_total = monthly_totals.mean()
    if mean_total == 0:
        return 1.0

    cv = abs(monthly_totals.std() / mean_total)

    # Convert CV to stability score
    # CV < 0.3 = very stable (stability ~0.85)
    # CV = 0.5 = moderate (stability ~0.75)
    # CV > 1.0 = unstable (stability < 0.5)
    stability = max(0.0, 1.0 - cv / 2)

    return stability


def _select_recurring_mask(tx_df: pd.DataFrame) -> pd.Series:
    """Select the best recurring mask using smart fallback logic.

    Strategy:
    1. If is_recurring_flag has reasonable coverage (>15% of value) AND is stable, use it
    2. If original flags appear corrupted (high variance) AND discovered is better, use discovered
    3. If coverage is too low AND is_recurring_discovered exists with better coverage, use that
    4. Otherwise, use is_recurring_flag (even if low coverage)

    This handles:
    - Clean data with accurate flags → uses is_recurring_flag (e.g., PoC dataset)
    - Corrupted flags (wrong transactions marked) → falls back to is_recurring_discovered
    - Missing flags → falls back to is_recurring_discovered

    Args:
        tx_df: Transaction DataFrame

    Returns:
        Boolean Series indicating which transactions are recurring
    """
    # Get the original flag
    original_flag = tx_df.get("is_recurring_flag", pd.Series(False, index=tx_df.index))
    if original_flag.dtype == object:
        original_flag = original_flag.fillna(False).astype(bool)
    else:
        original_flag = original_flag.fillna(False)

    # Calculate coverage of original flag (by absolute value)
    total_abs_value = tx_df["amount"].abs().sum()
    if total_abs_value == 0:
        return original_flag

    original_recurring_value = tx_df.loc[original_flag, "amount"].abs().sum()
    original_coverage = original_recurring_value / total_abs_value

    # Check if discovered recurrence is available
    has_discovered = "is_recurring_discovered" in tx_df.columns

    if not has_discovered:
        # No alternative, use original flag
        logger.debug(f"Using is_recurring_flag (coverage: {original_coverage:.1%})")
        return original_flag

    # Get discovered flag
    discovered_flag = tx_df["is_recurring_discovered"].fillna(False)
    if discovered_flag.dtype == object:
        discovered_flag = discovered_flag.astype(bool)

    discovered_recurring_value = tx_df.loc[discovered_flag, "amount"].abs().sum()
    discovered_coverage = discovered_recurring_value / total_abs_value

    # Calculate stability scores for both
    original_stability = _calculate_recurring_stability(tx_df, original_flag)
    discovered_stability = _calculate_recurring_stability(tx_df, discovered_flag)

    logger.debug(
        f"Recurring mask comparison - Original: coverage={original_coverage:.1%}, "
        f"stability={original_stability:.2f}; Discovered: coverage={discovered_coverage:.1%}, "
        f"stability={discovered_stability:.2f}"
    )

    # Decision logic with stability check
    # Key insight: prefer discovered if it's significantly more stable (>0.1 improvement)
    stability_improvement = discovered_stability - original_stability

    if original_coverage >= MIN_RECURRING_VALUE_RATIO:
        # Check if discovered is significantly more stable
        if stability_improvement >= 0.1 and discovered_stability >= 0.9:
            # Discovered is much better, use it even if original is "acceptable"
            logger.info(
                f"Preferring is_recurring_discovered: significantly more stable "
                f"(original={original_stability:.2f}, discovered={discovered_stability:.2f}, "
                f"improvement={stability_improvement:.2f})"
            )
            return discovered_flag
        elif original_stability >= 0.5:  # Stability threshold
            logger.debug(
                f"Using is_recurring_flag (coverage: {original_coverage:.1%}, "
                f"stability: {original_stability:.2f})"
            )
            return original_flag
        elif discovered_stability > original_stability:
            # Original has coverage but appears corrupted, discovered is more stable
            logger.info(
                f"Falling back to is_recurring_discovered: original has coverage "
                f"({original_coverage:.1%}) but low stability ({original_stability:.2f}), "
                f"discovered more stable ({discovered_stability:.2f})"
            )
            return discovered_flag
        else:
            # Neither is great, use original
            logger.debug(
                f"Using is_recurring_flag (both have similar stability, "
                f"original: {original_stability:.2f}, discovered: {discovered_stability:.2f})"
            )
            return original_flag
    elif discovered_coverage > original_coverage * 1.5:
        # Original flags look corrupted, discovered has significantly better coverage
        logger.info(
            f"Falling back to is_recurring_discovered: original coverage {original_coverage:.1%} "
            f"below threshold, discovered coverage {discovered_coverage:.1%}"
        )
        return discovered_flag
    else:
        # Neither is great, but original is not much worse - trust original
        logger.debug(
            f"Using is_recurring_flag (coverage: {original_coverage:.1%}, "
            f"discovered not significantly better: {discovered_coverage:.1%})"
        )
        return original_flag


def _decompose_from_transactions(
    monthly_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose using transaction-level recurring flags with smart fallback.

    Uses is_recurring_flag when reliable, falls back to is_recurring_discovered
    when flags appear corrupted (e.g., very low recurring rate).

    Reliability heuristic:
    - If < 15% of absolute transaction value is marked recurring, flags may be corrupted
    - In that case, use is_recurring_discovered if available and has better coverage
    """
    df = monthly_df.copy()
    tx_df = transactions_df.copy()

    # Ensure month_key exists
    if "month_key" not in tx_df.columns and "tx_date" in tx_df.columns:
        tx_df["month_key"] = pd.to_datetime(tx_df["tx_date"]).dt.strftime("%Y-%m")

    # Determine grouping
    group_cols = ["month_key"]
    if "customer_id" in tx_df.columns and "customer_id" in df.columns:
        group_cols = ["customer_id", "month_key"]

    # Smart fallback: choose between is_recurring_flag and is_recurring_discovered
    recurring_mask = _select_recurring_mask(tx_df)

    # Aggregate deterministic (recurring transactions)
    recurring_agg = (
        tx_df[recurring_mask]
        .groupby(group_cols)["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "deterministic_base"})
    )

    # Aggregate residual (non-recurring transactions)
    residual_agg = (
        tx_df[~recurring_mask]
        .groupby(group_cols)["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "residual"})
    )

    # Merge with monthly data
    df = df.merge(recurring_agg, on=group_cols, how="left")
    df = df.merge(residual_agg, on=group_cols, how="left")

    # Fill NaN with 0
    df["deterministic_base"] = df["deterministic_base"].fillna(0.0)
    df["residual"] = df["residual"].fillna(0.0)

    return df


def _decompose_approximate(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Approximate decomposition when transaction details unavailable.

    Uses statistical methods to estimate the deterministic component:
    1. Compute median NECF as stable baseline
    2. Residual = NECF - median baseline

    This is a simplified approach; full decomposition requires
    transaction-level recurring flags.
    """
    df = df.copy()

    # Use rolling median as approximation of deterministic base
    # (Recurring flows are typically stable month-to-month)
    # Use center=False to prevent data leakage (no future data in calculation)
    if "customer_id" in df.columns:
        df["deterministic_base"] = df.groupby("customer_id")["necf"].transform(
            lambda x: x.rolling(window=12, min_periods=3, center=False).median()
        )
    else:
        df["deterministic_base"] = df["necf"].rolling(
            window=12, min_periods=3, center=False
        ).median()

    # Fill edge months with overall median
    overall_median = df["necf"].median()
    df["deterministic_base"] = df["deterministic_base"].fillna(overall_median)

    # Residual is the difference
    df["residual"] = df["necf"] - df["deterministic_base"]

    logger.warning(
        "Using approximate decomposition (no transaction data). "
        "For accurate decomposition, provide transaction-level is_recurring_flag."
    )

    return df


def _validate_decomposition(df: pd.DataFrame, epsilon: float = 0.01) -> None:
    """Validate decomposition integrity constraint.

    Per SDD Section 10.3:
    |NECF - (Deterministic Base + Residual)| < ε
    """
    if "necf" not in df.columns:
        return

    df_check = df.copy()
    df_check["_recomposed"] = df_check["deterministic_base"] + df_check["residual"]
    df_check["_error"] = (df_check["necf"] - df_check["_recomposed"]).abs()

    violations = df_check[df_check["_error"] > epsilon]

    if len(violations) > 0:
        logger.warning(
            f"Decomposition integrity violations: {len(violations)} months "
            f"exceed tolerance {epsilon}"
        )


def get_deterministic_flows(
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract deterministic (recurring) flows from transactions.

    Returns summary of recurring flow patterns for forecasting.
    """
    df = transactions_df.copy()

    # Use discovered recurrence if available, otherwise fall back to upstream flag
    recurring_col = "is_recurring_discovered" if "is_recurring_discovered" in df.columns else "is_recurring_flag"

    if recurring_col not in df.columns:
        return pd.DataFrame()

    recurring = df[df[recurring_col] == True].copy()

    if len(recurring) == 0:
        return pd.DataFrame()

    # Group by category to understand recurring patterns
    summary = (
        recurring.groupby("category")
        .agg(
            avg_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            std_amount=("amount", "std"),
            count=("tx_id", "count") if "tx_id" in recurring.columns else ("amount", "count"),
            months_active=("month_key", "nunique") if "month_key" in recurring.columns else ("amount", "count"),
        )
        .reset_index()
    )

    # Identify stable recurring flows (low variance)
    summary["coefficient_of_variation"] = summary["std_amount"].abs() / summary["avg_amount"].abs()
    summary["is_stable"] = summary["coefficient_of_variation"] < 0.1  # <10% variation

    return summary


def compute_deterministic_projection(
    historical_df: pd.DataFrame,
    recency_weight: float = 0.7,
    trend_window: int = 6,
) -> DeterministicProjection:
    """Compute trend-adjusted deterministic projection.

    This replaces the naive mean() with a projection that:
    1. Gives more weight to recent months
    2. Captures trends (salary raises, rent changes)
    3. Provides confidence based on stability

    Args:
        historical_df: DataFrame with month_key and deterministic_base
        recency_weight: Weight for recent vs older months (0-1)
        trend_window: Months to use for trend calculation

    Returns:
        DeterministicProjection with trend-adjusted values
    """
    if len(historical_df) == 0:
        return DeterministicProjection(
            base_value=0.0,
            monthly_trend=0.0,
            confidence=0.0,
            method="empty",
        )

    df = historical_df.copy()
    df = df.sort_values("month_key")
    values = df["deterministic_base"].values

    n = len(values)

    if n < 3:
        # Too little data, use simple mean
        return DeterministicProjection(
            base_value=float(np.mean(values)),
            monthly_trend=0.0,
            confidence=0.5,
            method="mean_fallback",
        )

    # Strategy 1: Detect level shifts (structural breaks)
    # Look for significant jumps in recent data
    shift_idx, shift_detected = _detect_level_shift(values)

    if shift_detected:
        # Use only post-shift data
        post_shift_values = values[shift_idx:]
        base_value = float(np.mean(post_shift_values))

        # Calculate trend from post-shift period
        if len(post_shift_values) >= 3:
            trend = _calculate_trend(post_shift_values)
        else:
            trend = 0.0

        return DeterministicProjection(
            base_value=base_value,
            monthly_trend=trend,
            confidence=0.8,
            method="level_shift_adjusted",
        )

    # Strategy 2: Exponentially weighted mean with trend
    # Recent months get more weight
    weights = np.array([recency_weight ** (n - i - 1) for i in range(n)])
    weights = weights / weights.sum()

    weighted_mean = float(np.sum(values * weights))

    # Calculate trend from recent window
    recent_values = values[-min(trend_window, n):]
    trend = _calculate_trend(recent_values)

    # Confidence based on stability
    cv = abs(np.std(recent_values) / np.mean(recent_values)) if np.mean(recent_values) != 0 else 1.0
    confidence = max(0.0, min(1.0, 1.0 - cv))

    # Adjust base value to be "as of last month" for projection
    # Base = last_value OR weighted_mean if last value is anomalous
    last_value = values[-1]
    if abs(last_value - weighted_mean) > 2 * np.std(values):
        # Last value is anomalous, use weighted mean
        base_value = weighted_mean
    else:
        base_value = last_value

    return DeterministicProjection(
        base_value=base_value,
        monthly_trend=trend,
        confidence=confidence,
        method="exponential_weighted_trend",
    )


def _detect_level_shift(
    values: np.ndarray,
    threshold: float = 2.0,
    min_post_shift_points: int = 3,
) -> Tuple[int, bool]:
    """Detect structural break (level shift) in time series.

    Uses Z-score approach to find sudden jumps. Returns the most recent
    significant shift that leaves enough data points for trend estimation.

    Enhanced to detect all significant shifts (not just recent half),
    which helps identify contract-related step changes correctly.

    Args:
        values: Time series values
        threshold: Number of standard deviations for shift detection
        min_post_shift_points: Minimum data points required after shift

    Returns:
        Tuple of (shift_index, shift_detected)
    """
    n = len(values)
    if n < 6:
        return 0, False

    diffs = np.diff(values)

    # Look for jumps larger than threshold * std
    std_diff = np.std(diffs)
    if std_diff == 0:
        return 0, False

    # Find significant jumps
    z_scores = np.abs(diffs) / std_diff
    shift_candidates = np.where(z_scores > threshold)[0]

    if len(shift_candidates) == 0:
        return 0, False

    # Find the most recent shift that leaves enough post-shift data
    # Iterate from most recent to oldest
    for candidate in reversed(shift_candidates):
        shift_idx = candidate + 1
        post_shift_count = n - shift_idx

        # Ensure enough data points after the shift for trend estimation
        if post_shift_count >= min_post_shift_points:
            return shift_idx, True

    return 0, False


def _calculate_trend(
    values: np.ndarray,
    min_points: int = 4,
    max_cv: float = 0.5,
) -> float:
    """Calculate linear trend (slope) from values with stability checks.

    Enhanced with:
    - Minimum data points requirement for reliable trend estimation
    - Stability check using coefficient of variation (CV)
    - Returns 0 (flat) if data is too volatile for trend estimation

    This prevents misinterpreting step changes as persistent trends.

    Args:
        values: Time series values
        min_points: Minimum data points required for trend (default 4)
        max_cv: Maximum coefficient of variation allowed (default 0.5 = 50%)

    Returns:
        Monthly trend (positive = increasing), or 0.0 if unstable
    """
    n = len(values)
    if n < 2:
        return 0.0

    # Require minimum data points for reliable trend
    if n < min_points:
        return 0.0

    # Stability check: coefficient of variation
    # If data is too volatile, trend estimation is unreliable
    mean_val = np.mean(values)
    if mean_val != 0:
        cv = abs(np.std(values) / mean_val)
        if cv > max_cv:
            # Data too volatile - return flat projection
            return 0.0

    # Simple linear regression
    x = np.arange(n)
    x_mean = x.mean()
    y_mean = values.mean()

    numerator = np.sum((x - x_mean) * (values - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0

    slope = numerator / denominator

    # Additional check: if slope is very small relative to mean, treat as flat
    # This prevents noise from creating spurious trends
    if mean_val != 0 and abs(slope / mean_val) < 0.01:
        return 0.0

    return float(slope)


def compute_known_future_delta(
    crf_df: pd.DataFrame,
    forecast_start: str,
    forecast_end: str,
) -> pd.DataFrame:
    """Compute KnownFutureFlow_Delta vector from CRF events.

    Per SDD Section 12.3, this represents known future changes:
    - Loan maturity (+amount freed)
    - Installment plan end (+amount freed)
    - Subscription termination (+amount freed)

    Args:
        crf_df: CRF DataFrame with recurrence_end_date
        forecast_start: Start of forecast period (YYYY-MM)
        forecast_end: End of forecast period (YYYY-MM)

    Returns:
        DataFrame with month_key and delta_value columns
    """
    if crf_df is None or len(crf_df) == 0:
        return pd.DataFrame(columns=["month_key", "delta_value", "counterparty_display_name"])

    df = crf_df.copy()

    # Filter to contracts ending within forecast period
    if "recurrence_end_date" not in df.columns:
        return pd.DataFrame(columns=["month_key", "delta_value", "counterparty_display_name"])

    df["end_month"] = pd.to_datetime(df["recurrence_end_date"]).dt.strftime("%Y-%m")

    # Filter to forecast period
    in_range = (df["end_month"] >= forecast_start) & (df["end_month"] <= forecast_end)
    ending_contracts = df[in_range].copy()

    if len(ending_contracts) == 0:
        return pd.DataFrame(columns=["month_key", "delta_value", "counterparty_display_name"])

    # Delta is the freed amount (positive = inflow improvement)
    # If contractual_amount is negative (outflow), ending it is positive delta
    ending_contracts["delta_value"] = -ending_contracts.get("contractual_amount", 0)

    result = ending_contracts[["end_month", "delta_value", "display_name"]].rename(
        columns={
            "end_month": "month_key",
            "display_name": "counterparty_display_name",
        }
    )

    # Aggregate by month in case multiple contracts end same month
    result = (
        result.groupby("month_key")
        .agg(
            delta_value=("delta_value", "sum"),
            counterparty_display_name=("counterparty_display_name", lambda x: ", ".join(x.dropna())),
        )
        .reset_index()
    )

    return result
