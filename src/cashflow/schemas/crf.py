"""Counterparty Reference Feed (CRF) schema - SDD Section 5."""

from __future__ import annotations
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ContractType(str, Enum):
    """Type of contractual relationship."""

    LOAN = "LOAN"
    CARD_INSTALLMENT = "CARD_INSTALLMENT"
    MANDATE = "MANDATE"
    SUBSCRIPTION = "SUBSCRIPTION"
    GENERIC = "GENERIC"


class CRFRecord(BaseModel):
    """Counterparty Reference Feed record.

    CRF provides authoritative contractual truth that cannot be inferred
    reliably from transactions alone (SDD Section 5.1).

    Responsibilities:
    - Define true recurrence end dates
    - Define contractual amounts
    - Override UTF ambiguity
    - Provide future event certainty
    """

    counterparty_key: str = Field(..., min_length=1, description="Unique counterparty identifier")
    customer_id: str = Field(..., min_length=1, description="Associated customer")
    display_name: str = Field(..., description="Human-readable counterparty name")
    contract_type: ContractType = Field(..., description="Type of contract")
    contractual_amount: Optional[float] = Field(
        None, description="Fixed contractual amount per period"
    )
    recurrence_end_date: Optional[date] = Field(
        None, description="Authoritative end date for recurring payments"
    )
    is_variable_amount: bool = Field(
        default=False, description="Whether amount varies (overrides UTF)"
    )
    category_override: Optional[str] = Field(
        None, description="Category to use instead of UTF category"
    )

    @property
    def has_known_end_date(self) -> bool:
        """Check if this contract has a known termination date."""
        return self.recurrence_end_date is not None

    @property
    def is_ending_soon(self) -> bool:
        """Check if contract ends within forecast horizon (12 months)."""
        if self.recurrence_end_date is None:
            return False
        from datetime import timedelta

        return self.recurrence_end_date <= date.today() + timedelta(days=365)

    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }


# Precedence rules from SDD Section 5.3
END_DATE_PRECEDENCE = [
    "utf_override",  # UTF RecurrenceEndDate if set
    "loan",  # Loan contract end date
    "card_installment",  # Card installment plan end
    "mandate",  # Mandate termination
    "crf_generic",  # Generic CRF end date
]

AMOUNT_PRECEDENCE = [
    "loan",  # Loan contractual amount
    "card_installment",  # Card installment amount
    "mandate",  # Mandate amount
    "historical_median",  # Fallback to historical median
]
