"""Monthly aggregation and NECF construction - SDD Section 9.4."""

from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional


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


