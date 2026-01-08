"""Monthly aggregation and NECF construction - SDD Section 9.4."""

from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def aggregate_monthly(
    df: pd.DataFrame,
    customer_id: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate transactions to monthly Net External Cash Flow (NECF).

    Per SDD Section 9.4:
    - NECF = sum of external transaction amounts per month
    - Aggregation keys: CustomerId, MonthKey (YYYY-MM)

    Args:
        df: DataFrame with external transactions (after transfer netting)
        customer_id: Optional customer ID filter

    Returns:
        DataFrame with monthly NECF records
    """
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "month_key",
                "necf",
                "credit_total",
                "debit_total",
                "transaction_count",
            ]
        )

    df = df.copy()

    # Ensure required columns
    if "month_key" not in df.columns and "tx_date" in df.columns:
        df["month_key"] = pd.to_datetime(df["tx_date"]).dt.strftime("%Y-%m")

    # Filter by customer if specified
    if customer_id and "customer_id" in df.columns:
        df = df[df["customer_id"] == customer_id]

    # Compute credit/debit splits
    df["credit"] = df["amount"].apply(lambda x: x if x > 0 else 0)
    df["debit"] = df["amount"].apply(lambda x: x if x < 0 else 0)

    # Determine grouping columns
    group_cols = ["month_key"]
    if "customer_id" in df.columns:
        group_cols = ["customer_id"] + group_cols

    # Aggregate by month
    monthly = (
        df.groupby(group_cols)
        .agg(
            necf=("amount", "sum"),
            credit_total=("credit", "sum"),
            debit_total=("debit", "sum"),
            transaction_count=("tx_id", "count") if "tx_id" in df.columns else ("amount", "count"),
        )
        .reset_index()
    )

    # Sort by month
    monthly = monthly.sort_values(by=group_cols).reset_index(drop=True)

    # Add rolling features
    monthly = _add_rolling_features(monthly, group_cols)

    logger.info(f"Monthly aggregation: {len(monthly)} month records created")

    return monthly


def _add_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Add rolling window features to monthly data.

    Per SDD Section 12.2, includes rolling averages for smoothing.
    """
    df = df.copy()

    # 3-month rolling average of NECF
    if "customer_id" in group_cols:
        df["necf_3m_avg"] = df.groupby("customer_id")["necf"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
    else:
        df["necf_3m_avg"] = df["necf"].rolling(window=3, min_periods=1).mean()

    return df


def validate_monthly_data(
    df: pd.DataFrame,
    expected_months: Optional[int] = None,
    min_months: int = 24,
) -> dict:
    """Validate monthly aggregated data quality.

    Args:
        df: Monthly NECF DataFrame
        expected_months: Expected number of months
        min_months: Minimum required months (default 24 per SDD)

    Returns:
        Validation report dictionary
    """
    report = {
        "valid": True,
        "month_count": 0,
        "missing_months": [],
        "warnings": [],
    }

    if len(df) == 0:
        report["valid"] = False
        report["warnings"].append("No monthly data")
        return report

    # Count months
    report["month_count"] = df["month_key"].nunique()

    # Check minimum months
    if report["month_count"] < min_months:
        report["warnings"].append(
            f"Insufficient history: {report['month_count']} months, "
            f"minimum {min_months} required"
        )
        # Note: Don't mark as invalid, just warn (low confidence)

    # Check for expected months
    if expected_months and report["month_count"] != expected_months:
        report["warnings"].append(
            f"Month count mismatch: expected {expected_months}, found {report['month_count']}"
        )

    # Check for missing months in sequence
    if "month_key" in df.columns:
        months = pd.to_datetime(df["month_key"]).sort_values()
        if len(months) > 1:
            expected_range = pd.date_range(
                start=months.min(), end=months.max(), freq="MS"
            )
            expected_keys = expected_range.strftime("%Y-%m").tolist()
            actual_keys = df["month_key"].tolist()
            missing = set(expected_keys) - set(actual_keys)
            if missing:
                report["missing_months"] = sorted(list(missing))
                report["warnings"].append(f"Missing months in sequence: {missing}")

    logger.info(f"Monthly validation: {report['month_count']} months, {len(report['warnings'])} warnings")

    return report


def fill_missing_months(
    df: pd.DataFrame,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Fill missing months in the time series.

    Args:
        df: Monthly NECF DataFrame
        fill_value: Value to use for missing months

    Returns:
        DataFrame with continuous monthly sequence
    """
    if len(df) == 0 or "month_key" not in df.columns:
        return df

    df = df.copy()

    # Convert month_key to datetime for range generation
    df["_month_dt"] = pd.to_datetime(df["month_key"])
    min_date = df["_month_dt"].min()
    max_date = df["_month_dt"].max()

    # Generate full date range
    full_range = pd.date_range(start=min_date, end=max_date, freq="MS")
    full_df = pd.DataFrame({"_month_dt": full_range})
    full_df["month_key"] = full_df["_month_dt"].dt.strftime("%Y-%m")

    # Determine customer column
    customer_col = "customer_id" if "customer_id" in df.columns else None
    if customer_col:
        # Fill for each customer
        customer_id = df[customer_col].iloc[0]
        full_df[customer_col] = customer_id

    # Merge and fill missing values
    merge_cols = ["month_key"] + ([customer_col] if customer_col else [])
    result = full_df.merge(df.drop(columns=["_month_dt"]), on=merge_cols, how="left")

    # Fill numeric columns with fill_value
    numeric_cols = ["necf", "credit_total", "debit_total", "transaction_count"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = result[col].fillna(fill_value)

    # Recalculate rolling features
    group_cols = merge_cols.copy()
    result = _add_rolling_features(result, group_cols)

    result = result.drop(columns=["_month_dt"])

    return result
