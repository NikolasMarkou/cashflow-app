"""UTF and CRF data ingestion and validation - SDD Sections 4-5, 8."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import ValidationError

from cashflow.schemas.utf import UTFRecord, Direction
from cashflow.schemas.crf import CRFRecord


# Required columns for UTF per SDD Section 4.2
UTF_REQUIRED_COLUMNS = [
    "customer_id",
    "account_id",
    "tx_id",
    "tx_date",
    "amount",
    "currency",
    "direction",
    "category",
    "is_recurring_flag",
]

# Column name mapping from various formats to our schema
UTF_COLUMN_MAPPING = {
    # SDD format
    "CustomerId": "customer_id",
    "AccountId": "account_id",
    "TxID": "tx_id",
    "TxDate": "tx_date",
    "Amount": "amount",
    "Currency": "currency",
    "Direction": "direction",
    "Category": "category",
    "DescriptionRaw": "description_raw",
    "TransferLinkID": "transfer_link_id",
    "IsRecurringFlag": "is_recurring_flag",
    "IsVariableAmount": "is_variable_amount",
    "CounterpartyKey": "counterparty_key",
    "RecurrenceStartDate": "recurrence_start_date",
    "RecurrenceEndDate": "recurrence_end_date",
    # Client PoC format
    "TransactionID": "tx_id",
    "TransactionDate": "tx_date",
    "PostingDate": "posting_date",
    "AccountID": "account_id",
    "CurrencyCode": "currency",
    "CategoryCode": "category",
    "CounterpartyType": "counterparty_type",
}

CRF_COLUMN_MAPPING = {
    "CounterpartyKey": "counterparty_key",
    "CustomerId": "customer_id",
    "DisplayName": "display_name",
    "ContractType": "contract_type",
    "ContractualAmount": "contractual_amount",
    "RecurrenceEndDate": "recurrence_end_date",
    "IsVariableAmount": "is_variable_amount",
    "CategoryOverride": "category_override",
}


def _normalize_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Normalize column names using the provided mapping."""
    # First, rename any columns that match the mapping
    rename_dict = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # Convert remaining columns to snake_case
    def to_snake_case(name: str) -> str:
        import re

        # Handle camelCase and PascalCase
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    remaining = {c: to_snake_case(c) for c in df.columns if c not in rename_dict.values()}
    return df.rename(columns=remaining)


def load_utf(
    path: str | Path,
    customer_id: Optional[str] = None,
) -> pd.DataFrame:
    """Load UTF data from CSV file.

    Args:
        path: Path to the UTF CSV file
        customer_id: Optional customer ID to assign if not present in data

    Returns:
        DataFrame with normalized column names
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"UTF file not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_columns(df, UTF_COLUMN_MAPPING)

    # If customer_id column doesn't exist but was provided, add it
    if "customer_id" not in df.columns and customer_id:
        df["customer_id"] = customer_id
    elif "customer_id" not in df.columns:
        # Use account_id as customer_id for single-account scenarios
        if "account_id" in df.columns:
            df["customer_id"] = df["account_id"]

    # Infer direction from amount if not present
    if "direction" not in df.columns and "amount" in df.columns:
        df["direction"] = df["amount"].apply(
            lambda x: Direction.CREDIT.value if x >= 0 else Direction.DEBIT.value
        )

    # Handle is_variable_amount default
    if "is_variable_amount" not in df.columns:
        df["is_variable_amount"] = False

    return df


def load_crf(path: str | Path) -> pd.DataFrame:
    """Load CRF data from CSV file.

    Args:
        path: Path to the CRF CSV file

    Returns:
        DataFrame with normalized column names
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CRF file not found: {path}")

    df = pd.read_csv(path)
    return _normalize_columns(df, CRF_COLUMN_MAPPING)


def validate_utf(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Validate UTF records against the schema.

    Args:
        df: DataFrame with UTF data

    Returns:
        Tuple of (valid_records_df, list_of_errors)
    """
    valid_records = []
    errors = []

    for idx, row in df.iterrows():
        try:
            record = UTFRecord(
                customer_id=str(row.get("customer_id", "")),
                account_id=str(row.get("account_id", "")),
                tx_id=str(row.get("tx_id", "")),
                tx_date=pd.to_datetime(row.get("tx_date")).date(),
                amount=float(row.get("amount", 0)),
                currency=str(row.get("currency", "EUR")),
                direction=Direction(row.get("direction", "DEBIT")),
                category=str(row.get("category", "")),
                description_raw=row.get("description_raw"),
                transfer_link_id=row.get("transfer_link_id"),
                is_recurring_flag=bool(row.get("is_recurring_flag", False)),
                is_variable_amount=bool(row.get("is_variable_amount", False)),
                counterparty_key=row.get("counterparty_key"),
            )
            valid_records.append(record.model_dump())
        except (ValidationError, ValueError, TypeError) as e:
            errors.append({"row": idx, "error": str(e), "data": row.to_dict()})

    if not valid_records:
        return pd.DataFrame(), errors

    return pd.DataFrame(valid_records), errors


def validate_crf(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Validate CRF records against the schema.

    Args:
        df: DataFrame with CRF data

    Returns:
        Tuple of (valid_records_df, list_of_errors)
    """
    valid_records = []
    errors = []

    for idx, row in df.iterrows():
        try:
            record = CRFRecord(
                counterparty_key=str(row["counterparty_key"]),
                customer_id=str(row["customer_id"]),
                display_name=str(row.get("display_name", "")),
                contract_type=row.get("contract_type", "GENERIC"),
                contractual_amount=row.get("contractual_amount"),
                recurrence_end_date=(
                    pd.to_datetime(row["recurrence_end_date"]).date()
                    if pd.notna(row.get("recurrence_end_date"))
                    else None
                ),
                is_variable_amount=bool(row.get("is_variable_amount", False)),
                category_override=row.get("category_override"),
            )
            valid_records.append(record.model_dump())
        except (ValidationError, ValueError, TypeError) as e:
            errors.append({"row": idx, "error": str(e), "data": row.to_dict()})

    if not valid_records:
        return pd.DataFrame(), errors

    return pd.DataFrame(valid_records), errors
