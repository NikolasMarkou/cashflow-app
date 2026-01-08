"""Cash flow decomposition - SDD Section 10."""

from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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


def _decompose_from_transactions(
    monthly_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose using transaction-level recurring flags.

    Uses is_recurring_flag to separate deterministic from residual.
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

    # Separate recurring (deterministic) and non-recurring (residual)
    recurring_mask = tx_df.get("is_recurring_flag", pd.Series(False, index=tx_df.index))

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
    if "customer_id" in df.columns:
        df["deterministic_base"] = df.groupby("customer_id")["necf"].transform(
            lambda x: x.rolling(window=12, min_periods=3, center=True).median()
        )
    else:
        df["deterministic_base"] = df["necf"].rolling(
            window=12, min_periods=3, center=True
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

    if "is_recurring_flag" not in df.columns:
        return pd.DataFrame()

    recurring = df[df["is_recurring_flag"] == True].copy()

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
