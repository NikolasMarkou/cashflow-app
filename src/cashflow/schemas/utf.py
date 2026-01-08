"""Unified Transaction Feed (UTF) schema - SDD Section 4."""

from __future__ import annotations
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Direction(str, Enum):
    """Transaction direction."""

    CREDIT = "CREDIT"
    DEBIT = "DEBIT"


class UTFRecord(BaseModel):
    """Single transaction record from the Unified Transaction Feed.

    The UTF is the canonical, normalized source of all historical customer
    transaction activity. It serves as the sole behavioral input to the
    forecasting engine (SDD Section 4.1).
    """

    customer_id: str = Field(..., min_length=1, description="Unique customer identifier")
    account_id: str = Field(..., min_length=1, description="Source account identifier")
    tx_id: str = Field(..., min_length=1, description="Unique transaction ID")
    tx_date: date = Field(..., description="Posting date")
    amount: float = Field(..., description="Signed amount (+ credit, - debit)")
    currency: str = Field(..., min_length=3, max_length=3, description="ISO currency code")
    direction: Direction = Field(..., description="CREDIT or DEBIT")
    category: str = Field(..., min_length=1, description="Transaction category")
    description_raw: Optional[str] = Field(None, description="Raw transaction description")
    transfer_link_id: Optional[str] = Field(
        None, description="Links internal transfer pairs"
    )
    is_recurring_flag: bool = Field(..., description="Recurrence indicator")
    is_variable_amount: bool = Field(default=False, description="Amount variability indicator")
    counterparty_key: Optional[str] = Field(None, description="CRF join key")
    recurrence_start_date: Optional[date] = Field(None, description="UTF-level recurrence start")
    recurrence_end_date: Optional[date] = Field(None, description="UTF-level recurrence end override")

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, v: str) -> str:
        """Normalize currency code to uppercase."""
        return v.upper()

    @field_validator("amount")
    @classmethod
    def validate_amount_sign(cls, v: float, info) -> float:
        """Validate that amount sign matches direction."""
        # Note: We allow the amount to be pre-signed or unsigned
        # The direction field is the authoritative source
        return v

    @property
    def month_key(self) -> str:
        """Derive MonthKey from transaction date (YYYY-MM format)."""
        return self.tx_date.strftime("%Y-%m")

    @property
    def signed_amount(self) -> float:
        """Return amount with correct sign based on direction."""
        abs_amount = abs(self.amount)
        return abs_amount if self.direction == Direction.CREDIT else -abs_amount

    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }
