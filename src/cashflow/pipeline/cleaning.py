"""Data cleaning and validation - SDD Section 8."""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


def clean_utf(
    df: pd.DataFrame,
    drop_invalid_dates: bool = True,
    drop_invalid_amounts: bool = True,
    fill_missing_posting_date: bool = True,
) -> pd.DataFrame:
    """Clean and normalize UTF data.

    Per SDD Section 8.2:
    - Reject missing CustomerId, AccountId, TxDate, Amount
    - Normalize currencies
    - Enforce valid dates
    - Deduplicate using composite key

    Args:
        df: Raw UTF DataFrame
        drop_invalid_dates: Whether to drop rows with invalid dates
        drop_invalid_amounts: Whether to drop rows with invalid amounts
        fill_missing_posting_date: Whether to fill missing posting dates with tx_date

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    original_count = len(df)

    # 1. Normalize date columns
    for col in ["tx_date", "posting_date", "recurrence_start_date", "recurrence_end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 2. Drop rows with invalid transaction dates
    if drop_invalid_dates:
        before = len(df)
        df = df.dropna(subset=["tx_date"])
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with invalid tx_date")

    # 3. Fill missing posting_date with tx_date
    if fill_missing_posting_date and "posting_date" in df.columns:
        missing = df["posting_date"].isna().sum()
        if missing > 0:
            logger.info(f"Filling {missing} missing posting_date with tx_date")
            df["posting_date"] = df["posting_date"].fillna(df["tx_date"])

    # 4. Normalize amount to numeric
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        if drop_invalid_amounts:
            before = len(df)
            df = df.dropna(subset=["amount"])
            dropped = before - len(df)
            if dropped > 0:
                logger.warning(f"Dropped {dropped} rows with invalid amount")

    # 5. Normalize boolean fields
    for col in ["is_recurring_flag", "is_variable_amount"]:
        if col in df.columns:
            df[col] = _normalize_boolean(df[col])

    # 6. Normalize currency codes to uppercase
    if "currency" in df.columns:
        df["currency"] = df["currency"].astype(str).str.upper().str.strip()

    # 7. Drop rows with missing required fields
    required = ["customer_id", "account_id", "tx_id", "tx_date", "amount"]
    present_required = [c for c in required if c in df.columns]
    if present_required:
        before = len(df)
        df = df.dropna(subset=present_required)
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with missing required fields")

    # 8. Drop rows with empty category
    if "category" in df.columns:
        before = len(df)
        df = df[df["category"].astype(str).str.strip() != ""]
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with empty category")

    # 9. Deduplicate by composite key (account_id + tx_id)
    if "account_id" in df.columns and "tx_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["account_id", "tx_id"], keep="first")
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} duplicate transactions")

    # 10. Derive month_key from tx_date
    if "tx_date" in df.columns:
        df["month_key"] = df["tx_date"].dt.strftime("%Y-%m")

    # 11. Sort by date
    if "tx_date" in df.columns:
        df = df.sort_values(by=["tx_date", "tx_id"]).reset_index(drop=True)

    logger.info(f"Cleaned UTF: {len(df)} rows (from {original_count} original)")

    return df


def _normalize_boolean(series: pd.Series) -> pd.Series:
    """Normalize various boolean representations to Python bool."""
    # Convert to string and normalize
    str_series = series.astype(str).str.strip().str.lower()

    # Define truthy values
    truthy = {"true", "1", "yes", "y", "t"}

    return str_series.isin(truthy)


def validate_data_quality(df: pd.DataFrame) -> dict:
    """Generate data quality report for UTF data.

    Returns:
        Dictionary with quality metrics
    """
    report = {
        "total_rows": len(df),
        "date_range": None,
        "unique_customers": 0,
        "unique_accounts": 0,
        "missing_values": {},
        "data_quality_score": 0.0,
    }

    if len(df) == 0:
        return report

    # Date range
    if "tx_date" in df.columns and df["tx_date"].notna().any():
        report["date_range"] = {
            "min": df["tx_date"].min().strftime("%Y-%m-%d"),
            "max": df["tx_date"].max().strftime("%Y-%m-%d"),
        }

    # Unique counts
    if "customer_id" in df.columns:
        report["unique_customers"] = df["customer_id"].nunique()
    if "account_id" in df.columns:
        report["unique_accounts"] = df["account_id"].nunique()

    # Missing values
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            report["missing_values"][col] = missing

    # Quality score (simple completeness metric)
    total_cells = len(df) * len(df.columns)
    missing_cells = sum(report["missing_values"].values())
    report["data_quality_score"] = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0

    return report
