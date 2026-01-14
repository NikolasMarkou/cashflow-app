"""
Framework Configuration Module

Defines dataclasses for account types, randomness levels, and test configurations
as specified in docs/2026_01_13_framework.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class AccountType(Enum):
    """Account type enumeration."""
    PERSONAL = "personal"
    SME = "sme"
    CORPORATE = "corporate"


class RandomnessLevel(Enum):
    """Randomness level enumeration."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AccountConfig:
    """Configuration for a specific account type.

    Defines transaction patterns, volumes, and category distributions
    for Personal, SME, and Corporate accounts.
    """
    account_type: AccountType
    monthly_income: float
    income_sources: int
    recurring_expense_count: Tuple[int, int]  # (min, max)
    transaction_volume: Tuple[int, int]  # (min, max) per month
    categories: List[str]

    # Transaction composition percentages
    recurring_pct: float
    semi_predictable_pct: float
    variable_pct: float

    # Transfer patterns
    transfers_per_month: Tuple[int, int]  # (min, max)


# Pre-defined account configurations
ACCOUNT_CONFIGS: Dict[AccountType, AccountConfig] = {
    AccountType.PERSONAL: AccountConfig(
        account_type=AccountType.PERSONAL,
        monthly_income=3000.0,
        income_sources=1,
        recurring_expense_count=(5, 7),
        transaction_volume=(40, 60),
        categories=[
            "SALARY", "RENT", "UTILITIES", "GROCERIES",
            "TRANSPORT", "ENTERTAINMENT", "SUBSCRIPTIONS",
            "DINING", "SHOPPING", "HEALTHCARE"
        ],
        recurring_pct=0.60,
        semi_predictable_pct=0.0,
        variable_pct=0.40,
        transfers_per_month=(2, 4),
    ),
    AccountType.SME: AccountConfig(
        account_type=AccountType.SME,
        monthly_income=25000.0,
        income_sources=7,  # midpoint of 5-10
        recurring_expense_count=(10, 15),
        transaction_volume=(150, 250),
        categories=[
            "INVOICE_PAYMENT", "PAYROLL", "RENT", "UTILITIES",
            "SUPPLIES", "TAX", "INSURANCE", "PROFESSIONAL_SERVICES",
            "MARKETING", "EQUIPMENT", "TRAVEL", "LOAN_REPAYMENT"
        ],
        recurring_pct=0.45,
        semi_predictable_pct=0.35,
        variable_pct=0.20,
        transfers_per_month=(5, 10),
    ),
    AccountType.CORPORATE: AccountConfig(
        account_type=AccountType.CORPORATE,
        monthly_income=500000.0,
        income_sources=20,  # midpoint of 15-25
        recurring_expense_count=(20, 30),
        transaction_volume=(500, 800),
        categories=[
            "INVOICE_PAYMENT", "PAYROLL", "INTERCOMPANY", "TAX",
            "CAPITAL_EXPENSE", "RENT", "UTILITIES", "INSURANCE",
            "PROFESSIONAL_SERVICES", "IT_SERVICES", "MARKETING",
            "TRAVEL", "DEBT_SERVICE", "DIVIDENDS", "REGULATORY_FEES"
        ],
        recurring_pct=0.50,
        semi_predictable_pct=0.30,
        variable_pct=0.20,
        transfers_per_month=(20, 30),
    ),
}


@dataclass
class RandomnessConfig:
    """Configuration for a specific randomness level.

    Controls the degree of unpredictability in synthetic data generation.
    """
    level: RandomnessLevel

    # Residual-focused parameters
    residual_magnitude_multiplier: float  # 1.0 = baseline budget, 2.0 = double
    residual_volatility: float  # Std dev of per-transaction variation
    residual_transaction_multiplier: float  # 1.0 = baseline count, higher = more transactions

    # Predictability parameters
    predictable_expense_pct: float  # Percentage of recurring expenses that are fully predictable

    # Data quality testing
    flag_corruption_rate: float  # Probability of flipping is_recurring_flag


# Pre-defined randomness configurations
RANDOMNESS_CONFIGS: Dict[RandomnessLevel, RandomnessConfig] = {
    RandomnessLevel.NONE: RandomnessConfig(
        level=RandomnessLevel.NONE,
        residual_magnitude_multiplier=1.0,
        residual_volatility=0.0,
        residual_transaction_multiplier=1.0,
        predictable_expense_pct=0.90,  # 90% of recurring expenses are fully predictable
        flag_corruption_rate=0.0,
    ),
    RandomnessLevel.LOW: RandomnessConfig(
        level=RandomnessLevel.LOW,
        residual_magnitude_multiplier=0.9,  # Reduced 10%
        residual_volatility=0.135,  # Reduced 10% from 0.15
        residual_transaction_multiplier=0.9,  # Reduced 10%
        predictable_expense_pct=0.80,  # 80% predictable
        flag_corruption_rate=0.02,  # Reduced from 5% to 2%
    ),
    RandomnessLevel.MEDIUM: RandomnessConfig(
        level=RandomnessLevel.MEDIUM,
        residual_magnitude_multiplier=1.17,  # Reduced 10% from 1.3
        residual_volatility=0.27,  # Reduced 10% from 0.30
        residual_transaction_multiplier=1.08,  # Reduced 10% from 1.2
        predictable_expense_pct=0.70,  # 70% predictable
        flag_corruption_rate=0.05,  # Reduced from 15% to 5%
    ),
    RandomnessLevel.HIGH: RandomnessConfig(
        level=RandomnessLevel.HIGH,
        residual_magnitude_multiplier=1.44,  # Reduced 10% from 1.6
        residual_volatility=0.45,  # Reduced 10% from 0.50
        residual_transaction_multiplier=1.35,  # Reduced 10% from 1.5
        predictable_expense_pct=0.60,  # 60% predictable
        flag_corruption_rate=0.10,  # Reduced from 30% to 10%
    ),
}


@dataclass
class TestConfig:
    """Combined test configuration.

    Represents a single test configuration combining account type,
    randomness level, and execution parameters.
    """
    account_type: AccountType
    randomness_level: RandomnessLevel
    seed: int
    training_months: int = 28  # 28 months to allow 4-month test split while keeping 24 for TiRex
    forecast_horizon: int = 12

    @property
    def config_id(self) -> str:
        """Generate configuration identifier (e.g., 'P-LOW')."""
        prefix_map = {
            AccountType.PERSONAL: "P",
            AccountType.SME: "S",
            AccountType.CORPORATE: "C",
        }
        return f"{prefix_map[self.account_type]}-{self.randomness_level.value.upper()}"

    @property
    def account_config(self) -> AccountConfig:
        """Get account configuration."""
        return ACCOUNT_CONFIGS[self.account_type]

    @property
    def randomness_config(self) -> RandomnessConfig:
        """Get randomness configuration."""
        return RANDOMNESS_CONFIGS[self.randomness_level]


# Category classification for applying variation
CATEGORY_CLASSIFICATION = {
    # Fixed categories (very predictable)
    "SALARY": "fixed",
    "RENT": "fixed",
    "PAYROLL": "fixed",
    "LOAN_REPAYMENT": "fixed",
    "DEBT_SERVICE": "fixed",
    "INSURANCE": "fixed",

    # Semi-fixed categories (some variation)
    "UTILITIES": "semi_fixed",
    "SUBSCRIPTIONS": "semi_fixed",
    "TAX": "semi_fixed",
    "REGULATORY_FEES": "semi_fixed",
    "IT_SERVICES": "semi_fixed",

    # Variable categories (moderate variation)
    "GROCERIES": "variable",
    "SUPPLIES": "variable",
    "TRANSPORT": "variable",
    "HEALTHCARE": "variable",
    "INVOICE_PAYMENT": "variable",
    "INTERCOMPANY": "variable",

    # Discretionary categories (high variation)
    "ENTERTAINMENT": "discretionary",
    "DINING": "discretionary",
    "SHOPPING": "discretionary",
    "TRAVEL": "discretionary",
    "MARKETING": "discretionary",
    "EQUIPMENT": "discretionary",
    "CAPITAL_EXPENSE": "discretionary",
    "PROFESSIONAL_SERVICES": "discretionary",
    "DIVIDENDS": "discretionary",
}




def get_all_test_configs(seeds: int = 10) -> List[TestConfig]:
    """Generate all test configurations.

    Args:
        seeds: Number of seeds per configuration (default 10)

    Returns:
        List of TestConfig objects (12 configs * seeds)
    """
    configs = []
    for account_type in AccountType:
        for randomness_level in RandomnessLevel:
            for seed in range(1, seeds + 1):
                configs.append(TestConfig(
                    account_type=account_type,
                    randomness_level=randomness_level,
                    seed=seed,
                ))
    return configs
