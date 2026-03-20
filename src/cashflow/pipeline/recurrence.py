"""Recurrence detection - Layer 0.5 internal discovery.

This module implements automatic recurrence detection independent of
the upstream is_recurring_flag. It analyzes transaction patterns using
frequency analysis and autocorrelation to identify recurring flows.

This addresses the Single Point of Failure (SPOF) where noisy or missing
upstream flags would pollute the residual component.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional
from collections import defaultdict

# Minimum occurrences to consider a pattern recurring
MIN_OCCURRENCES = 3

# Maximum coefficient of variation for stable recurring patterns
MAX_CV_THRESHOLD = 0.15  # 15% variation

# Minimum months coverage (pattern should appear in at least this % of months)
MIN_COVERAGE_RATIO = 0.6  # 60% of months


def discover_recurring_patterns(
    transactions_df: pd.DataFrame,
    min_occurrences: int = MIN_OCCURRENCES,
    max_cv: float = MAX_CV_THRESHOLD,
    min_coverage: float = MIN_COVERAGE_RATIO,
) -> pd.DataFrame:
    """Discover recurring transaction patterns from raw transaction data.

    This implements Layer 0.5 "Recurrence Discovery" which runs before
    decomposition to identify recurring patterns regardless of the
    upstream is_recurring_flag.

    Detection criteria:
    1. Counterparty consistency - same counterparty/category
    2. Amount stability - coefficient of variation < threshold
    3. Monthly periodicity - appears regularly each month
    4. Frequency - minimum number of occurrences

    Args:
        transactions_df: Raw transaction DataFrame
        min_occurrences: Minimum transaction count to consider pattern
        max_cv: Maximum coefficient of variation for amount stability
        min_coverage: Minimum ratio of months pattern should appear

    Returns:
        DataFrame with discovered recurring patterns
    """
    df = transactions_df.copy()

    if len(df) == 0:
        return pd.DataFrame()

    # Ensure required columns
    if "tx_date" in df.columns:
        df["month_key"] = pd.to_datetime(df["tx_date"]).dt.strftime("%Y-%m")

    if "month_key" not in df.columns:
        logger.warning("No month_key or tx_date column, cannot detect recurrence")
        return pd.DataFrame()

    # Calculate total months span
    total_months = df["month_key"].nunique()
    min_months_coverage = int(total_months * min_coverage)

    discovered_patterns = []

    # Strategy 1: Category-based pattern detection
    category_patterns = _detect_by_category(df, min_occurrences, max_cv, min_months_coverage)
    discovered_patterns.extend(category_patterns)

    # Strategy 2: Counterparty-based pattern detection (if available)
    if "counterparty_key" in df.columns or "description_raw" in df.columns:
        counterparty_patterns = _detect_by_counterparty(df, min_occurrences, max_cv, min_months_coverage)
        discovered_patterns.extend(counterparty_patterns)

    # Strategy 3: Amount cluster detection (fixed amounts appearing monthly)
    amount_patterns = _detect_by_amount_cluster(df, min_occurrences, min_months_coverage)
    discovered_patterns.extend(amount_patterns)

    if discovered_patterns:
        result = pd.DataFrame(discovered_patterns)
        logger.info(f"Discovered {len(result)} recurring patterns via internal detection")
        return result

    return pd.DataFrame()


def _detect_by_category(
    df: pd.DataFrame,
    min_occurrences: int,
    max_cv: float,
    min_months_coverage: int,
) -> list[dict]:
    """Detect recurring patterns by category stability."""
    patterns = []

    if "category" not in df.columns:
        return patterns

    # Categories that are inherently variable and should NOT be auto-detected as recurring
    # Even if they occur regularly, their amounts are unpredictable
    VARIABLE_CATEGORIES = {
        "GROCERIES", "ENTERTAINMENT", "TRANSPORT", "DINING", "SHOPPING",
        "HEALTHCARE", "TRAVEL", "MARKETING", "SUPPLIES", "EQUIPMENT",
        "CAPITAL_EXPENSE", "PROFESSIONAL_SERVICES", "ONE_TIME_LARGE",
    }

    for category, group in df.groupby("category"):
        # Skip transfer categories (already handled)
        if str(category).upper() in {"TRANSFER_IN", "TRANSFER_OUT", "INTERNAL_TRANSFER"}:
            continue

        # Skip variable/discretionary categories - they are not truly recurring
        if str(category).upper() in VARIABLE_CATEGORIES:
            continue

        # Aggregate by month
        monthly = group.groupby("month_key")["amount"].agg(["sum", "count"]).reset_index()

        if len(monthly) < min_occurrences:
            continue

        if len(monthly) < min_months_coverage:
            continue

        # Check amount stability
        mean_amount = monthly["sum"].mean()
        std_amount = monthly["sum"].std()

        if mean_amount == 0:
            continue

        cv = abs(std_amount / mean_amount)

        if cv <= max_cv:
            patterns.append({
                "pattern_type": "category",
                "pattern_key": category,
                "avg_amount": mean_amount,
                "std_amount": std_amount,
                "coefficient_of_variation": cv,
                "months_active": len(monthly),
                "transaction_count": len(group),
                "confidence_score": 1.0 - cv,  # Higher is better
                "detection_method": "category_stability",
            })

    return patterns


def _detect_by_counterparty(
    df: pd.DataFrame,
    min_occurrences: int,
    max_cv: float,
    min_months_coverage: int,
) -> list[dict]:
    """Detect recurring patterns by counterparty consistency."""
    patterns = []

    # Use counterparty_key if available, otherwise try to extract from description
    key_col = None
    if "counterparty_key" in df.columns:
        key_col = "counterparty_key"
    elif "description_raw" in df.columns:
        # Extract potential counterparty from description
        df = df.copy()
        df["_counterparty_extracted"] = df["description_raw"].apply(_extract_counterparty)
        key_col = "_counterparty_extracted"

    if key_col is None:
        return patterns

    for counterparty, group in df.groupby(key_col):
        if pd.isna(counterparty) or counterparty == "":
            continue

        # Aggregate by month
        monthly = group.groupby("month_key")["amount"].agg(["sum", "mean", "count"]).reset_index()

        if len(monthly) < min_occurrences:
            continue

        if len(monthly) < min_months_coverage:
            continue

        # Check amount stability
        mean_amount = monthly["sum"].mean()
        std_amount = monthly["sum"].std()

        if mean_amount == 0:
            continue

        cv = abs(std_amount / mean_amount)

        if cv <= max_cv:
            patterns.append({
                "pattern_type": "counterparty",
                "pattern_key": counterparty,
                "avg_amount": mean_amount,
                "std_amount": std_amount,
                "coefficient_of_variation": cv,
                "months_active": len(monthly),
                "transaction_count": len(group),
                "confidence_score": 1.0 - cv,
                "detection_method": "counterparty_stability",
            })

    return patterns


def _detect_by_amount_cluster(
    df: pd.DataFrame,
    min_occurrences: int,
    min_months_coverage: int,
) -> list[dict]:
    """Detect recurring patterns by amount clustering.

    Finds transactions with nearly identical amounts appearing monthly.
    Useful for fixed payments like rent, subscriptions, loan installments.
    """
    patterns = []

    # Round amounts to reduce noise (to nearest 10)
    df = df.copy()
    df["_amount_rounded"] = (df["amount"] / 10).round() * 10

    # Group by rounded amount
    for amount, group in df.groupby("_amount_rounded"):
        if amount == 0:
            continue

        months_present = group["month_key"].nunique()

        if months_present < min_months_coverage:
            continue

        if len(group) < min_occurrences:
            continue

        # Check if appears roughly once per month
        avg_per_month = len(group) / months_present

        if 0.5 <= avg_per_month <= 2.0:  # Between 0.5 and 2 transactions per month
            patterns.append({
                "pattern_type": "fixed_amount",
                "pattern_key": f"amount_{amount:.0f}",
                "avg_amount": group["amount"].mean(),
                "std_amount": group["amount"].std(),
                "coefficient_of_variation": abs(group["amount"].std() / group["amount"].mean()) if group["amount"].mean() != 0 else 0,
                "months_active": months_present,
                "transaction_count": len(group),
                "confidence_score": min(1.0, months_present / (min_months_coverage * 1.5)),
                "detection_method": "amount_cluster",
            })

    return patterns


def _extract_counterparty(description: str) -> str:
    """Extract counterparty identifier from description."""
    if pd.isna(description):
        return ""

    desc = str(description).upper()

    # Common patterns to extract counterparty
    # Remove common transaction type prefixes
    prefixes = ["SALARY", "RENT", "UTILITIES", "GROCERIES", "TRANSPORT", "ENTERTAINMENT"]
    for prefix in prefixes:
        if desc.startswith(prefix):
            return prefix

    # Take first word as counterparty
    parts = desc.split()
    if parts:
        return parts[0]

    return ""


def apply_discovered_recurrence(
    transactions_df: pd.DataFrame,
    patterns_df: pd.DataFrame,
    confidence_threshold: float = 0.7,
) -> pd.DataFrame:
    """Apply discovered patterns to tag transactions as recurring.

    Updates the is_recurring_flag based on discovered patterns,
    using a union of upstream flag and internal detection.

    Args:
        transactions_df: Transaction DataFrame with existing is_recurring_flag
        patterns_df: Discovered patterns from discover_recurring_patterns
        confidence_threshold: Minimum confidence to apply pattern

    Returns:
        DataFrame with is_recurring_discovered column added
    """
    df = transactions_df.copy()
    df["is_recurring_discovered"] = False
    df["recurrence_detection_source"] = None

    if len(patterns_df) == 0:
        # Keep original flag
        df["is_recurring_discovered"] = df.get("is_recurring_flag", False)
        return df

    # Filter patterns by confidence
    high_confidence = patterns_df[patterns_df["confidence_score"] >= confidence_threshold]

    # Apply category-based patterns
    category_patterns = high_confidence[high_confidence["pattern_type"] == "category"]
    for _, pattern in category_patterns.iterrows():
        mask = df["category"] == pattern["pattern_key"]
        df.loc[mask, "is_recurring_discovered"] = True
        df.loc[mask & df["recurrence_detection_source"].isna(), "recurrence_detection_source"] = "category_discovery"

    # Apply amount-based patterns
    amount_patterns = high_confidence[high_confidence["pattern_type"] == "fixed_amount"]
    for _, pattern in amount_patterns.iterrows():
        target_amount = pattern["avg_amount"]
        tolerance = abs(target_amount) * 0.05  # 5% tolerance
        mask = (df["amount"] >= target_amount - tolerance) & (df["amount"] <= target_amount + tolerance)
        df.loc[mask, "is_recurring_discovered"] = True
        df.loc[mask & df["recurrence_detection_source"].isna(), "recurrence_detection_source"] = "amount_discovery"

    # Union with original flag (discovered OR upstream flag)
    if "is_recurring_flag" in df.columns:
        original_recurring = df["is_recurring_flag"].fillna(False)
        # Track source BEFORE union — items only in original get "upstream_flag"
        mask_upstream_only = original_recurring & ~df["is_recurring_discovered"]
        df.loc[mask_upstream_only, "recurrence_detection_source"] = "upstream_flag"
        # Now apply union
        df["is_recurring_discovered"] = df["is_recurring_discovered"] | original_recurring

    discovered_count = df["is_recurring_discovered"].sum()
    original_count = df.get("is_recurring_flag", pd.Series(False)).sum()
    new_discoveries = discovered_count - original_count

    logger.info(
        f"Recurrence discovery: {original_count} from upstream, "
        f"{new_discoveries} newly discovered, {discovered_count} total"
    )

    return df


def get_recurrence_summary(df: pd.DataFrame) -> dict:
    """Get summary of recurrence detection results."""
    if "is_recurring_discovered" not in df.columns:
        return {"error": "Recurrence discovery not applied"}

    original_count = df.get("is_recurring_flag", pd.Series(False)).sum()
    discovered_count = df["is_recurring_discovered"].sum()

    by_source = df[df["is_recurring_discovered"]].groupby("recurrence_detection_source").size().to_dict()

    return {
        "upstream_recurring_count": int(original_count),
        "total_recurring_after_discovery": int(discovered_count),
        "newly_discovered": int(discovered_count - original_count),
        "by_detection_source": by_source,
        "improvement_ratio": discovered_count / max(original_count, 1),
    }
