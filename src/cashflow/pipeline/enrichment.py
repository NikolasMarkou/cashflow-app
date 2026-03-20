"""UTF-CRF enrichment with precedence rules - SDD Section 5."""

from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

from cashflow.schemas.crf import ContractType, END_DATE_PRECEDENCE, AMOUNT_PRECEDENCE


def enrich_with_crf(
    utf_df: pd.DataFrame,
    crf_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Enrich UTF data with CRF contractual information.

    Per SDD Section 5.3, applies precedence rules for:
    - End dates: UTF override > Loan > Card Installment > Mandate > CRF generic
    - Amounts: Loan > Card > Mandate > Historical median

    Args:
        utf_df: Cleaned UTF DataFrame
        crf_df: Optional CRF DataFrame for enrichment

    Returns:
        Enriched UTF DataFrame with CRF data merged
    """
    df = utf_df.copy()

    if crf_df is None or len(crf_df) == 0:
        logger.info("No CRF data provided, skipping enrichment")
        # Add placeholder columns for consistency
        df["crf_end_date"] = pd.NaT
        df["crf_contractual_amount"] = np.nan
        df["crf_contract_type"] = None
        df["crf_display_name"] = None
        return df

    # Ensure CRF has required columns
    if "counterparty_key" not in crf_df.columns:
        logger.warning("CRF missing counterparty_key, skipping enrichment")
        return df

    # Join UTF with CRF on counterparty_key
    crf_subset = crf_df[
        [
            "counterparty_key",
            "display_name",
            "contract_type",
            "contractual_amount",
            "recurrence_end_date",
            "is_variable_amount",
        ]
    ].copy()

    crf_subset = crf_subset.rename(
        columns={
            "display_name": "crf_display_name",
            "contract_type": "crf_contract_type",
            "contractual_amount": "crf_contractual_amount",
            "recurrence_end_date": "crf_end_date",
            "is_variable_amount": "crf_is_variable",
        }
    )

    # Left join to preserve all UTF records
    df = df.merge(crf_subset, on="counterparty_key", how="left")

    # Apply end date precedence rules
    df = _apply_end_date_precedence(df)

    # Apply amount precedence rules
    df = _apply_amount_precedence(df)

    # Apply variability override (CRF overrides UTF)
    if "crf_is_variable" in df.columns:
        mask = df["crf_is_variable"].notna()
        df.loc[mask, "is_variable_amount"] = df.loc[mask, "crf_is_variable"]

    matched = df["crf_display_name"].notna().sum()
    logger.info(f"CRF enrichment: {matched}/{len(df)} transactions matched")

    return df


def _apply_end_date_precedence(df: pd.DataFrame) -> pd.DataFrame:
    """Apply end date precedence rules per SDD Section 5.3.

    Precedence: UTF override > Loan > Card Installment > Mandate > CRF generic
    """
    df = df.copy()

    # Initialize effective_end_date column
    df["effective_end_date"] = pd.NaT

    # Apply in reverse precedence order (lowest first, then override with higher)

    # 1. Start with CRF generic end date
    if "crf_end_date" in df.columns:
        df["effective_end_date"] = df["crf_end_date"]

    # 2. Override with contract-type specific dates (higher precedence)
    if "crf_contract_type" in df.columns and "crf_end_date" in df.columns:
        for contract_type in [
            ContractType.MANDATE.value,
            ContractType.CARD_INSTALLMENT.value,
            ContractType.LOAN.value,
        ]:
            mask = (df["crf_contract_type"] == contract_type) & df["crf_end_date"].notna()
            df.loc[mask, "effective_end_date"] = df.loc[mask, "crf_end_date"]

    # 3. UTF override has highest precedence
    if "recurrence_end_date" in df.columns:
        mask = df["recurrence_end_date"].notna()
        df.loc[mask, "effective_end_date"] = df.loc[mask, "recurrence_end_date"]

    return df


def _apply_amount_precedence(df: pd.DataFrame) -> pd.DataFrame:
    """Apply amount precedence rules per SDD Section 5.3.

    Precedence: Loan > Card > Mandate > Historical median
    """
    df = df.copy()

    # Initialize effective_amount column with actual transaction amount
    df["effective_amount"] = df["amount"]

    # For recurring transactions, we may want to use contractual amounts
    if "crf_contractual_amount" not in df.columns:
        return df

    # Only apply to recurring transactions
    recurring_mask = df.get("is_recurring_flag", pd.Series(False, index=df.index))

    if "crf_contract_type" in df.columns:
        # Apply in precedence order
        for contract_type in [
            ContractType.MANDATE.value,
            ContractType.CARD_INSTALLMENT.value,
            ContractType.LOAN.value,
        ]:
            mask = (
                recurring_mask
                & (df["crf_contract_type"] == contract_type)
                & df["crf_contractual_amount"].notna()
            )
            df.loc[mask, "effective_amount"] = df.loc[mask, "crf_contractual_amount"]

    return df


