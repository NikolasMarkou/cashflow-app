"""Transfer detection and netting - SDD Section 9."""

from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Default tolerance for matching transfer dates
DEFAULT_DATE_TOLERANCE_DAYS = 2

# Categories that indicate transfers
TRANSFER_CATEGORIES = {
    "TRANSFER_IN",
    "TRANSFER_OUT",
    "INTERNAL_TRANSFER",
    "SAVINGS_CONTRIBUTION",
    "SAVINGS_WITHDRAWAL",
}


def detect_transfers(
    df: pd.DataFrame,
    date_tolerance_days: int = DEFAULT_DATE_TOLERANCE_DAYS,
) -> pd.DataFrame:
    """Detect internal transfers in transaction data.

    Per SDD Section 9.2, an internal transfer is defined as a mirrored
    transaction pair satisfying:
    - Same CustomerId
    - Same absolute Amount
    - Opposite Direction (DEBIT vs CREDIT)
    - Occurring within configurable time tolerance (default ±2 days)
    - Linked explicitly via TransferLinkID OR inferred heuristically

    Detection priority (SDD 9.2.2):
    1. Explicit TransferLinkID
    2. Amount + Date proximity + Account mismatch
    3. Category-based heuristics

    Args:
        df: UTF DataFrame with transaction data
        date_tolerance_days: Maximum days between transfer pairs

    Returns:
        DataFrame with 'is_internal_transfer' and 'transfer_match_id' columns added
    """
    df = df.copy()
    df["is_internal_transfer"] = False
    df["transfer_match_id"] = None
    df["transfer_detection_method"] = None

    if len(df) == 0:
        return df

    # Method 1: Explicit TransferLinkID matching
    df = _match_by_transfer_link_id(df)

    # Method 2: Amount + Date + Account matching
    unmatched = df[~df["is_internal_transfer"]]
    if len(unmatched) > 0:
        df = _match_by_amount_date(df, date_tolerance_days)

    # Method 3: Category-based heuristics
    df = _match_by_category(df)

    transfer_count = df["is_internal_transfer"].sum()
    logger.info(f"Detected {transfer_count} internal transfer transactions")

    return df


def _match_by_transfer_link_id(df: pd.DataFrame) -> pd.DataFrame:
    """Match transfers using explicit TransferLinkID."""
    if "transfer_link_id" not in df.columns:
        return df

    # Find rows with non-null TransferLinkID
    has_link = df["transfer_link_id"].notna() & (df["transfer_link_id"] != "")

    if not has_link.any():
        return df

    # Group by TransferLinkID and mark as transfers if there are pairs
    for link_id, group in df[has_link].groupby("transfer_link_id"):
        if len(group) < 2:
            continue

        # Match opposite-amount pairs within the group
        amounts = group["amount"].values
        indices = group.index.tolist()
        matched = set()

        for i in range(len(amounts)):
            if i in matched:
                continue
            for j in range(i + 1, len(amounts)):
                if j in matched:
                    continue
                if np.isclose(amounts[i], -amounts[j]):
                    matched.add(i)
                    matched.add(j)
                    break

        if matched:
            matched_indices = [indices[i] for i in matched]
            df.loc[matched_indices, "is_internal_transfer"] = True
            df.loc[matched_indices, "transfer_match_id"] = str(link_id)
            df.loc[matched_indices, "transfer_detection_method"] = "transfer_link_id"

    return df


def _match_by_amount_date(
    df: pd.DataFrame,
    date_tolerance_days: int,
) -> pd.DataFrame:
    """Match transfers by amount, date proximity, and account mismatch.

    For multi-account customers, finds matching debit/credit pairs.
    """
    if "customer_id" not in df.columns or "tx_date" not in df.columns:
        return df

    # Only process unmatched transactions
    unmatched_mask = ~df["is_internal_transfer"]

    # Group by customer
    for customer_id, customer_df in df[unmatched_mask].groupby("customer_id"):
        if len(customer_df) < 2:
            continue

        # Only consider multi-account scenarios
        if "account_id" in customer_df.columns:
            if customer_df["account_id"].nunique() < 2:
                continue

        # Find potential matches: opposite amounts within date tolerance
        match_id = 0
        matched_indices = set()

        for idx, row in customer_df.iterrows():
            if idx in matched_indices:
                continue

            amount = row["amount"]
            tx_date = row["tx_date"]
            account_id = row.get("account_id")

            # Find matching opposite transaction
            candidates = customer_df[
                (customer_df.index != idx)
                & (~customer_df.index.isin(matched_indices))
                & (np.isclose(customer_df["amount"], -amount))
            ]

            if "account_id" in customer_df.columns:
                # Must be from different account
                candidates = candidates[candidates["account_id"] != account_id]

            if len(candidates) == 0:
                continue

            # Check date proximity
            for cand_idx, cand_row in candidates.iterrows():
                date_diff = abs((tx_date - cand_row["tx_date"]).days)
                if date_diff <= date_tolerance_days:
                    # Found a match
                    match_key = f"{customer_id}_amt_{match_id}"
                    df.loc[[idx, cand_idx], "is_internal_transfer"] = True
                    df.loc[[idx, cand_idx], "transfer_match_id"] = match_key
                    df.loc[[idx, cand_idx], "transfer_detection_method"] = "amount_date_match"
                    matched_indices.add(idx)
                    matched_indices.add(cand_idx)
                    match_id += 1
                    break

    return df


def _match_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Mark transactions as transfers based on category heuristics.

    This catches remaining internal transfers that couldn't be paired
    but are clearly transfers based on category.
    """
    if "category" not in df.columns:
        return df

    # Only process unmatched transactions
    unmatched_mask = ~df["is_internal_transfer"]

    # Check category against transfer patterns
    category_mask = df["category"].str.upper().isin(TRANSFER_CATEGORIES)

    # Mark as internal transfers
    to_mark = unmatched_mask & category_mask
    df.loc[to_mark, "is_internal_transfer"] = True
    df.loc[to_mark, "transfer_detection_method"] = "category_heuristic"

    return df


def net_transfers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove internal transfers from transaction data.

    Per SDD Section 9.3: Once identified, both sides of an internal
    transfer are fully excluded from forecasting inputs.

    Args:
        df: DataFrame with is_internal_transfer column

    Returns:
        Tuple of (filtered DataFrame, netting summary dict)
    """
    if "is_internal_transfer" not in df.columns:
        logger.warning("is_internal_transfer column not found, no netting performed")
        return df, {"num_transfers_removed": 0, "total_volume_removed": 0.0}

    # Calculate summary before removal
    transfers = df[df["is_internal_transfer"]]
    summary = {
        "num_transfers_removed": len(transfers),
        "total_volume_removed": float(transfers["amount"].abs().sum()) if len(transfers) > 0 else 0.0,
    }

    # Filter out internal transfers
    external_df = df[~df["is_internal_transfer"]].copy()

    logger.info(
        f"Transfer netting: removed {summary['num_transfers_removed']} transactions, "
        f"volume {summary['total_volume_removed']:.2f}"
    )

    return external_df, summary
