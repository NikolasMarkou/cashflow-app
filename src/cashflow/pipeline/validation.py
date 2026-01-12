"""Data quality validation contracts for production deployment.

Phase 3.2: Defines and enforces input data quality requirements.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date
import pandas as pd
import numpy as np


@dataclass
class ContractViolation:
    """Represents a single contract violation."""
    rule: str
    severity: str  # "error" | "warning"
    message: str
    details: Optional[Dict] = None


@dataclass
class ContractResult:
    """Result of contract enforcement."""
    passed: bool
    violations: List[ContractViolation] = field(default_factory=list)
    warnings: List[ContractViolation] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def add_violation(self, rule: str, message: str, details: Optional[Dict] = None) -> None:
        """Add an error violation."""
        self.violations.append(ContractViolation(rule, "error", message, details))
        self.passed = False

    def add_warning(self, rule: str, message: str, details: Optional[Dict] = None) -> None:
        """Add a warning (non-blocking)."""
        self.warnings.append(ContractViolation(rule, "warning", message, details))

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "violations": [
                {"rule": v.rule, "severity": v.severity, "message": v.message, "details": v.details}
                for v in self.violations
            ],
            "warnings": [
                {"rule": w.rule, "severity": w.severity, "message": w.message, "details": w.details}
                for w in self.warnings
            ],
            "stats": self.stats,
        }


@dataclass
class DataQualityContract:
    """Contract defining data quality requirements.

    Usage:
        contract = DataQualityContract()
        result = contract.enforce(utf_df)
        if not result.passed:
            raise ValueError(f"Data quality violations: {result.violations}")
    """
    # Minimum data requirements
    min_months_history: int = 24
    min_transactions_total: int = 100
    min_transactions_per_month: float = 3.0  # Average

    # Missing data thresholds
    max_missing_rate_required: float = 0.0  # Required fields must not be missing
    max_missing_rate_optional: float = 0.30  # 30% max for optional fields

    # Required fields (must be present and non-null)
    required_fields: List[str] = field(default_factory=lambda: [
        "tx_id", "customer_id", "account_id", "tx_date", "amount", "currency", "direction"
    ])

    # Optional fields (can have some missing values)
    optional_fields: List[str] = field(default_factory=lambda: [
        "description_raw", "counterparty_key", "category", "is_recurring_flag"
    ])

    # Date range constraints
    max_future_days: int = 0  # No future transactions allowed by default
    max_history_years: int = 10  # Max 10 years of history

    # Amount constraints
    max_single_amount: float = 1_000_000  # Max single transaction
    min_single_amount: float = -1_000_000  # Min single transaction (most negative)

    # Duplicate detection
    allow_duplicates: bool = False  # Reject duplicate tx_id

    # Currency constraints
    allowed_currencies: Optional[List[str]] = None  # None = all allowed

    # Direction constraints
    allowed_directions: List[str] = field(default_factory=lambda: ["CREDIT", "DEBIT"])

    def enforce(self, df: pd.DataFrame) -> ContractResult:
        """Enforce the contract on a DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            ContractResult with pass/fail status and details
        """
        result = ContractResult(passed=True)

        # Collect statistics
        result.stats = self._compute_stats(df)

        # Run all validation rules
        self._check_required_fields(df, result)
        self._check_row_count(df, result)
        self._check_date_range(df, result)
        self._check_monthly_coverage(df, result)
        self._check_missing_data(df, result)
        self._check_duplicates(df, result)
        self._check_amounts(df, result)
        self._check_currencies(df, result)
        self._check_directions(df, result)

        return result

    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics."""
        stats = {
            "total_rows": len(df),
            "unique_customers": df["customer_id"].nunique() if "customer_id" in df.columns else 0,
            "unique_accounts": df["account_id"].nunique() if "account_id" in df.columns else 0,
        }

        if "tx_date" in df.columns:
            dates = pd.to_datetime(df["tx_date"], errors="coerce")
            valid_dates = dates.dropna()
            if len(valid_dates) > 0:
                stats["date_range_start"] = str(valid_dates.min().date())
                stats["date_range_end"] = str(valid_dates.max().date())
                stats["months_span"] = (valid_dates.max().year - valid_dates.min().year) * 12 + \
                                       (valid_dates.max().month - valid_dates.min().month) + 1

        if "amount" in df.columns:
            amounts = pd.to_numeric(df["amount"], errors="coerce")
            stats["amount_sum"] = float(amounts.sum())
            stats["amount_mean"] = float(amounts.mean())
            stats["amount_std"] = float(amounts.std())

        return stats

    def _check_required_fields(self, df: pd.DataFrame, result: ContractResult) -> None:
        """Check that all required fields are present."""
        missing_columns = [col for col in self.required_fields if col not in df.columns]
        if missing_columns:
            result.add_violation(
                "required_fields",
                f"Missing required columns: {missing_columns}",
                {"missing_columns": missing_columns}
            )

    def _check_row_count(self, df: pd.DataFrame, result: ContractResult) -> None:
        """Check minimum transaction count."""
        if len(df) < self.min_transactions_total:
            result.add_violation(
                "min_transactions",
                f"Insufficient transactions: {len(df)} < {self.min_transactions_total} required",
                {"actual": len(df), "required": self.min_transactions_total}
            )

    def _check_date_range(self, df: pd.DataFrame, result: ContractResult) -> None:
        """Check date range constraints."""
        if "tx_date" not in df.columns:
            return

        dates = pd.to_datetime(df["tx_date"], errors="coerce")
        valid_dates = dates.dropna()

        if len(valid_dates) == 0:
            result.add_violation(
                "date_range",
                "No valid dates found in tx_date column",
            )
            return

        today = pd.Timestamp.now().normalize()
        min_date = valid_dates.min()
        max_date = valid_dates.max()

        # Check for future dates
        if self.max_future_days == 0:
            future_count = (valid_dates > today).sum()
            if future_count > 0:
                result.add_violation(
                    "future_dates",
                    f"Found {future_count} transactions with future dates",
                    {"future_count": int(future_count), "max_future_date": str(max_date.date())}
                )
        else:
            max_allowed = today + pd.Timedelta(days=self.max_future_days)
            future_count = (valid_dates > max_allowed).sum()
            if future_count > 0:
                result.add_warning(
                    "future_dates",
                    f"Found {future_count} transactions beyond {self.max_future_days} days in future",
                )

        # Check history depth
        history_years = (today - min_date).days / 365.25
        if history_years > self.max_history_years:
            result.add_warning(
                "history_depth",
                f"History spans {history_years:.1f} years, max recommended is {self.max_history_years}",
            )

    def _check_monthly_coverage(self, df: pd.DataFrame, result: ContractResult) -> None:
        """Check monthly data coverage."""
        if "tx_date" not in df.columns:
            return

        dates = pd.to_datetime(df["tx_date"], errors="coerce")
        valid_dates = dates.dropna()

        if len(valid_dates) == 0:
            return

        # Create month keys
        months = valid_dates.dt.to_period("M")
        month_counts = months.value_counts()
        num_months = len(month_counts)

        # Check minimum months
        if num_months < self.min_months_history:
            result.add_violation(
                "min_months",
                f"Insufficient history: {num_months} months < {self.min_months_history} required",
                {"actual_months": num_months, "required_months": self.min_months_history}
            )

        # Check average transactions per month
        avg_per_month = len(df) / max(num_months, 1)
        if avg_per_month < self.min_transactions_per_month:
            result.add_warning(
                "transactions_per_month",
                f"Low transaction density: {avg_per_month:.1f}/month < {self.min_transactions_per_month} recommended",
            )

        # Check for gaps (months with 0 transactions within the range)
        if num_months > 1:
            all_months = pd.period_range(months.min(), months.max(), freq="M")
            missing_months = set(all_months) - set(month_counts.index)
            if missing_months:
                result.add_warning(
                    "month_gaps",
                    f"Found {len(missing_months)} months with no transactions",
                    {"missing_months": [str(m) for m in sorted(missing_months)[:5]]}  # Show first 5
                )

    def _check_missing_data(self, df: pd.DataFrame, result: ContractResult) -> None:
        """Check missing data rates."""
        # Required fields
        for field in self.required_fields:
            if field in df.columns:
                missing_rate = df[field].isna().mean()
                if missing_rate > self.max_missing_rate_required:
                    result.add_violation(
                        "missing_required",
                        f"Required field '{field}' has {missing_rate:.1%} missing values",
                        {"field": field, "missing_rate": float(missing_rate)}
                    )

        # Optional fields
        for field in self.optional_fields:
            if field in df.columns:
                missing_rate = df[field].isna().mean()
                if missing_rate > self.max_missing_rate_optional:
                    result.add_warning(
                        "missing_optional",
                        f"Optional field '{field}' has {missing_rate:.1%} missing values (max {self.max_missing_rate_optional:.0%})",
                        {"field": field, "missing_rate": float(missing_rate)}
                    )

    def _check_duplicates(self, df: pd.DataFrame, result: ContractResult) -> None:
        """Check for duplicate transactions."""
        if self.allow_duplicates:
            return

        if "tx_id" not in df.columns:
            return

        duplicate_count = df["tx_id"].duplicated().sum()
        if duplicate_count > 0:
            result.add_violation(
                "duplicates",
                f"Found {duplicate_count} duplicate tx_id values",
                {"duplicate_count": int(duplicate_count)}
            )

    def _check_amounts(self, df: pd.DataFrame, result: ContractResult) -> None:
        """Check amount constraints."""
        if "amount" not in df.columns:
            return

        amounts = pd.to_numeric(df["amount"], errors="coerce")

        # Check for invalid amounts
        invalid_count = amounts.isna().sum()
        if invalid_count > 0:
            result.add_violation(
                "invalid_amounts",
                f"Found {invalid_count} non-numeric amount values",
            )

        # Check range
        max_amount = amounts.max()
        min_amount = amounts.min()

        if max_amount > self.max_single_amount:
            result.add_warning(
                "amount_range",
                f"Maximum amount {max_amount:,.2f} exceeds limit {self.max_single_amount:,.2f}",
            )

        if min_amount < self.min_single_amount:
            result.add_warning(
                "amount_range",
                f"Minimum amount {min_amount:,.2f} below limit {self.min_single_amount:,.2f}",
            )

    def _check_currencies(self, df: pd.DataFrame, result: ContractResult) -> None:
        """Check currency constraints."""
        if "currency" not in df.columns or self.allowed_currencies is None:
            return

        currencies = df["currency"].dropna().unique()
        invalid = [c for c in currencies if c not in self.allowed_currencies]
        if invalid:
            result.add_warning(
                "invalid_currency",
                f"Found unexpected currencies: {invalid}",
                {"invalid_currencies": invalid}
            )

    def _check_directions(self, df: pd.DataFrame, result: ContractResult) -> None:
        """Check direction constraints."""
        if "direction" not in df.columns:
            return

        directions = df["direction"].dropna().unique()
        invalid = [d for d in directions if d not in self.allowed_directions]
        if invalid:
            result.add_violation(
                "invalid_direction",
                f"Found invalid directions: {invalid}. Allowed: {self.allowed_directions}",
                {"invalid_directions": list(invalid)}
            )


# Preset contracts for common scenarios
STRICT_CONTRACT = DataQualityContract(
    min_months_history=24,
    min_transactions_total=200,
    min_transactions_per_month=5.0,
    max_missing_rate_optional=0.10,
    max_future_days=0,
    allow_duplicates=False,
)

LENIENT_CONTRACT = DataQualityContract(
    min_months_history=12,
    min_transactions_total=50,
    min_transactions_per_month=2.0,
    max_missing_rate_optional=0.50,
    max_future_days=7,
    allow_duplicates=True,
)

DEFAULT_CONTRACT = DataQualityContract()
