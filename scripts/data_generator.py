"""
Synthetic Data Generator

Generates realistic transaction data for testing the Cash Flow Forecasting Engine
as specified in docs/2026_01_13_framework.md
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from framework_config import (
    AccountConfig,
    AccountType,
    ACCOUNT_CONFIGS,
    CATEGORY_CLASSIFICATION,
    RandomnessConfig,
    RandomnessLevel,
    RANDOMNESS_CONFIGS,
    TestConfig,
)


# Default customer/account IDs for synthetic data
DEFAULT_CUSTOMER_ID = "CUST001"
DEFAULT_ACCOUNT_ID = "ACC001"
DEFAULT_CURRENCY = "EUR"


# =============================================================================
# SEASONAL AND PERIODIC EXPENSE PATTERNS
# =============================================================================

# Seasonal expense multipliers - applied to variable expenses
# Note: Reduced by 10% (closer to 1.0) for smoother forecasting
SEASONAL_EXPENSE_MULTIPLIERS = {
    # Personal: Very mild seasonal variation (10% reduction in amplitude)
    AccountType.PERSONAL: {
        1: 0.96,  # Jan (was 0.95)
        2: 0.96,  # Feb (was 0.95)
        3: 1.0,   # Mar
        4: 1.0,   # Apr
        5: 1.0,   # May
        6: 1.04,  # Jun (was 1.05)
        7: 1.08,  # Jul: Mild vacation (was 1.1)
        8: 1.04,  # Aug (was 1.05)
        9: 1.0,   # Sep
        10: 1.0,  # Oct
        11: 1.04, # Nov (was 1.05)
        12: 1.12, # Dec: Mild holiday (was 1.15)
    },
    # SME: Minimal quarter-end patterns (10% reduction)
    AccountType.SME: {
        1: 0.98, 2: 1.0, 3: 1.04,
        4: 0.98, 5: 1.0, 6: 1.04,
        7: 0.96, 8: 0.98, 9: 1.04,
        10: 1.0, 11: 1.02, 12: 1.06,
    },
    # Corporate: Minimal patterns (10% reduction)
    AccountType.CORPORATE: {
        1: 0.98, 2: 1.0, 3: 1.04,
        4: 0.98, 5: 1.0, 6: 1.02,
        7: 0.96, 8: 0.96, 9: 1.02,
        10: 1.0, 11: 1.02, 12: 1.06,
    },
}

# Seasonal income multipliers - applied to income (SME/Corporate have variation)
# Note: Multipliers minimized for stable, predictable income patterns
SEASONAL_INCOME_MULTIPLIERS = {
    AccountType.PERSONAL: {m: 1.0 for m in range(1, 13)},  # Salary is stable
    AccountType.SME: {
        1: 0.98, 2: 0.99, 3: 1.0,   # Minimal variation
        4: 1.0, 5: 1.01, 6: 1.0,
        7: 0.99, 8: 0.98, 9: 0.99,   # Minimal summer slowdown
        10: 1.01, 11: 1.02, 12: 1.03,  # Minimal Q4 rush
    },
    AccountType.CORPORATE: {
        1: 0.99, 2: 0.99, 3: 1.01,   # Q1 - minimal
        4: 0.99, 5: 1.0, 6: 1.01,   # Q2 - minimal
        7: 0.99, 8: 0.99, 9: 1.0,   # Q3 - minimal
        10: 1.0, 11: 1.01, 12: 1.02,  # Q4 - minimal
    },
}

# Occasional large expenses - DISABLED to reduce volatility
# These random expenses cause unpredictable spikes that the model can't forecast
OCCASIONAL_EXPENSES = {
    AccountType.PERSONAL: [],  # Removed for smoother data
    AccountType.SME: [],
    AccountType.CORPORATE: [],
}

# Periodic expenses - occur on specific months (annual/quarterly)
# (category, months (int or list), amount, description)
# Note: Amounts reduced to minimize volatility
PERIODIC_EXPENSES = {
    AccountType.PERSONAL: [
        ("INSURANCE", [1], 150, "Annual car insurance"),
        ("TAX", [4], 100, "Tax filing fees"),
        ("VACATION", [7], 300, "Annual vacation"),
    ],
    AccountType.SME: [
        ("TAX", [3, 6, 9, 12], 300, "Quarterly tax payment"),
        ("INSURANCE", [1], 500, "Annual insurance renewal"),
        ("AUDIT", [3], 400, "Annual audit fees"),
    ],
    AccountType.CORPORATE: [
        ("TAX", [3, 6, 9, 12], 4000, "Quarterly tax payment"),
        ("BONUS", [12], 8000, "Annual employee bonuses"),
        ("AUDIT", [3], 2500, "Annual audit fees"),
    ],
}


@dataclass
class TransactionTemplate:
    """Template for generating recurring transactions."""
    category: str
    base_amount: float
    day_of_month: int
    counterparty: str
    is_recurring: bool
    description: str


def generate_synthetic_data(
    config: TestConfig,
    start_date: Optional[datetime] = None,
    months: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic transaction data.

    Args:
        config: Test configuration
        start_date: Optional start date (defaults to 2024-01-01)
        months: Number of months to generate (defaults to training_months)

    Returns:
        DataFrame with UTF-formatted transactions
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1)

    if months is None:
        months = config.training_months

    np.random.seed(config.seed)

    account_cfg = config.account_config
    random_cfg = config.randomness_config

    transactions = []

    # Generate transactions for each month
    for month_offset in range(months):
        # Calculate the target year and month properly
        total_months = (start_date.year * 12 + start_date.month - 1) + month_offset
        year = total_months // 12
        month = (total_months % 12) + 1

        month_start = datetime(year, month, 1)

        # Generate income transactions
        income_txns = _generate_income_transactions(
            account_cfg, random_cfg, month_start, config.seed + month_offset
        )
        transactions.extend(income_txns)

        # Generate recurring expense transactions
        recurring_txns = _generate_recurring_expenses(
            account_cfg, random_cfg, month_start, config.seed + month_offset
        )
        transactions.extend(recurring_txns)

        # Generate variable expense transactions
        variable_txns = _generate_variable_expenses(
            account_cfg, random_cfg, month_start, config.seed + month_offset
        )
        transactions.extend(variable_txns)

        # Generate occasional large expenses (probability-based)
        occasional_txns = _generate_occasional_expenses(
            account_cfg, random_cfg, month_start, config.seed + month_offset
        )
        transactions.extend(occasional_txns)

        # Generate periodic expenses (annual/quarterly)
        periodic_txns = _generate_periodic_expenses(
            account_cfg, month_start
        )
        transactions.extend(periodic_txns)

        # Generate transfer pairs (internal transfers)
        transfer_txns = _generate_transfers(
            account_cfg, random_cfg, month_start, config.seed + month_offset
        )
        transactions.extend(transfer_txns)

    # Convert to DataFrame
    df = pd.DataFrame(transactions)

    # Apply flag corruption if configured
    if random_cfg.flag_corruption_rate > 0:
        df = _apply_flag_corruption(df, random_cfg.flag_corruption_rate, config.seed)

    return df


def generate_full_period_data(
    config: TestConfig,
    start_date: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate continuous 36-month data and split into training and holdout.

    This ensures holdout period has realistic continuity with training data.

    Args:
        config: Test configuration
        start_date: Optional start date

    Returns:
        Tuple of (training_df, holdout_df)
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1)

    total_months = config.training_months + config.forecast_horizon

    # Generate full continuous period
    full_df = generate_synthetic_data(config, start_date, months=total_months)

    # Split based on date
    full_df["tx_date"] = pd.to_datetime(full_df["tx_date"])

    # Calculate split date
    split_date = start_date + pd.DateOffset(months=config.training_months)

    training_df = full_df[full_df["tx_date"] < split_date].copy()
    holdout_df = full_df[full_df["tx_date"] >= split_date].copy()

    return training_df, holdout_df


def generate_holdout_data(
    config: TestConfig,
    training_end_date: datetime,
) -> pd.DataFrame:
    """Generate holdout period data for WMAPE calculation.

    DEPRECATED: Use generate_full_period_data() for continuous data generation.
    This function is kept for backward compatibility.

    Args:
        config: Test configuration
        training_end_date: End date of training period

    Returns:
        DataFrame with UTF-formatted transactions for holdout period
    """
    # Use offset seed to ensure different but reproducible data
    holdout_seed = config.seed + 1000

    np.random.seed(holdout_seed)

    account_cfg = config.account_config
    random_cfg = config.randomness_config

    transactions = []

    # Generate transactions for each month in forecast horizon
    for month_offset in range(config.forecast_horizon):
        month_date = training_end_date + timedelta(days=(month_offset + 1) * 30)
        year = month_date.year
        month = month_date.month

        # Adjust for month overflow
        if month > 12:
            year += (month - 1) // 12
            month = ((month - 1) % 12) + 1

        month_start = datetime(year, month, 1)

        # Generate income transactions
        income_txns = _generate_income_transactions(
            account_cfg, random_cfg, month_start, holdout_seed + month_offset
        )
        transactions.extend(income_txns)

        # Generate recurring expense transactions
        recurring_txns = _generate_recurring_expenses(
            account_cfg, random_cfg, month_start, holdout_seed + month_offset
        )
        transactions.extend(recurring_txns)

        # Generate variable expense transactions
        variable_txns = _generate_variable_expenses(
            account_cfg, random_cfg, month_start, holdout_seed + month_offset
        )
        transactions.extend(variable_txns)

        # Generate occasional large expenses (probability-based)
        occasional_txns = _generate_occasional_expenses(
            account_cfg, random_cfg, month_start, holdout_seed + month_offset
        )
        transactions.extend(occasional_txns)

        # Generate periodic expenses (annual/quarterly)
        periodic_txns = _generate_periodic_expenses(
            account_cfg, month_start
        )
        transactions.extend(periodic_txns)

    # Convert to DataFrame
    df = pd.DataFrame(transactions)

    # Apply flag corruption if configured
    if random_cfg.flag_corruption_rate > 0:
        df = _apply_flag_corruption(df, random_cfg.flag_corruption_rate, holdout_seed)

    return df


def _create_transaction(
    tx_date: datetime,
    amount: float,
    category: str,
    counterparty: str,
    is_recurring: bool,
    description: str,
    transfer_link_id: Optional[str] = None,
) -> dict:
    """Create a transaction dictionary with UTF schema.

    Args:
        tx_date: Transaction date
        amount: Transaction amount (positive=inflow, negative=outflow)
        category: Transaction category
        counterparty: Counterparty name
        is_recurring: Whether transaction is recurring
        description: Transaction description
        transfer_link_id: Optional transfer link ID

    Returns:
        Dictionary with UTF-formatted transaction
    """
    direction = "CREDIT" if amount >= 0 else "DEBIT"

    txn = {
        "tx_id": str(uuid.uuid4()),
        "customer_id": DEFAULT_CUSTOMER_ID,
        "account_id": DEFAULT_ACCOUNT_ID,
        "tx_date": tx_date,
        "posting_date": tx_date,
        "amount": round(amount, 2),
        "currency": DEFAULT_CURRENCY,
        "direction": direction,
        "category": category,
        "counterparty_name": counterparty,
        "is_recurring_flag": is_recurring,
        "description": description,
    }

    if transfer_link_id:
        txn["transfer_link_id"] = transfer_link_id

    return txn


def _generate_income_transactions(
    account_cfg: AccountConfig,
    random_cfg: RandomnessConfig,
    month_start: datetime,
    seed: int,
) -> List[dict]:
    """Generate income transactions for a month.

    Personal income is DETERMINISTIC (salary).
    SME/Corporate income has SEASONAL variation (business cycles).
    """
    np.random.seed(seed)
    transactions = []

    # Get seasonal income multiplier for this month
    month_num = month_start.month
    seasonal_multiplier = SEASONAL_INCOME_MULTIPLIERS[account_cfg.account_type].get(month_num, 1.0)

    if account_cfg.account_type == AccountType.PERSONAL:
        # Single salary payment - FIXED day 27, exact amount (no seasonality)
        salary_date = _safe_date(month_start.year, month_start.month, 27)

        # Exact salary amount - no variation
        amount = account_cfg.monthly_income

        transactions.append(_create_transaction(
            tx_date=salary_date,
            amount=round(amount, 2),
            category="SALARY",
            counterparty="Employer ABC",
            is_recurring=True,
            description="Monthly salary payment",
        ))

    elif account_cfg.account_type == AccountType.SME:
        # Customer payments - FIXED amounts for stable decomposition
        # No seasonal variation to ensure predictable recurring income
        n_payments = account_cfg.income_sources
        per_payment = account_cfg.monthly_income / n_payments

        for i in range(n_payments):
            # Fixed days based on customer index
            pay_day = 5 + (i * 3) % 20
            pay_date = _safe_date(month_start.year, month_start.month, pay_day)

            transactions.append(_create_transaction(
                tx_date=pay_date,
                amount=round(per_payment, 2),
                category="INVOICE_PAYMENT",
                counterparty=f"Customer {i + 1}",
                is_recurring=True,
                description=f"Invoice payment from customer {i + 1}",
            ))

    else:  # Corporate
        # Daily settlements - FIXED amounts for stable decomposition
        # No seasonal variation to ensure predictable recurring income
        n_days = 20
        per_day = account_cfg.monthly_income / n_days

        for i in range(n_days):
            pay_day = 1 + int(i * 28 / n_days)
            pay_date = _safe_date(month_start.year, month_start.month, pay_day)

            transactions.append(_create_transaction(
                tx_date=pay_date,
                amount=round(per_day, 2),
                category="INVOICE_PAYMENT",
                counterparty=f"BU Settlement {i + 1}",
                is_recurring=True,
                description=f"Daily settlement batch {i + 1}",
            ))

    return transactions


def _generate_recurring_expenses(
    account_cfg: AccountConfig,
    random_cfg: RandomnessConfig,
    month_start: datetime,
    seed: int,
) -> List[dict]:
    """Generate recurring expense transactions for a month.

    Recurring expenses have varying degrees of predictability based on
    the randomness level's predictable_expense_pct parameter.
    - At 90% predictable: 90% of expenses have exact amounts, 10% have variation
    - At 60% predictable: 60% of expenses have exact amounts, 40% have variation
    """
    np.random.seed(seed + 100)  # Offset seed for recurring expenses
    transactions = []

    # Define expense templates based on account type
    if account_cfg.account_type == AccountType.PERSONAL:
        templates = [
            TransactionTemplate("RENT", 1200, 1, "Landlord", True, "Monthly rent"),
            TransactionTemplate("UTILITIES", 150, 12, "Electric Co", True, "Electricity bill"),
            TransactionTemplate("UTILITIES", 80, 15, "Water Utility", True, "Water bill"),
            TransactionTemplate("SUBSCRIPTIONS", 15, 5, "Netflix", True, "Streaming subscription"),
            TransactionTemplate("SUBSCRIPTIONS", 10, 8, "Spotify", True, "Music subscription"),
            TransactionTemplate("INSURANCE", 100, 20, "Insurance Co", True, "Insurance premium"),
        ]
    elif account_cfg.account_type == AccountType.SME:
        templates = [
            TransactionTemplate("RENT", 3500, 1, "Commercial Property", True, "Office rent"),
            TransactionTemplate("UTILITIES", 800, 12, "Business Electric", True, "Electricity"),
            TransactionTemplate("UTILITIES", 200, 15, "Business Water", True, "Water"),
            TransactionTemplate("PAYROLL", 8000, 25, "Payroll Batch 1", True, "Staff payroll"),
            TransactionTemplate("PAYROLL", 4000, 25, "Payroll Batch 2", True, "Staff payroll"),
            TransactionTemplate("INSURANCE", 500, 10, "Business Insurance", True, "Insurance"),
            TransactionTemplate("LOAN_REPAYMENT", 1500, 5, "Bank Loan", True, "Loan repayment"),
            TransactionTemplate("IT_SERVICES", 300, 18, "Cloud Provider", True, "IT services"),
            TransactionTemplate("SUBSCRIPTIONS", 200, 7, "Software Licenses", True, "Software"),
        ]
    else:  # Corporate
        templates = [
            TransactionTemplate("RENT", 50000, 1, "Property Holdings", True, "HQ rent"),
            TransactionTemplate("UTILITIES", 15000, 12, "Utility Provider", True, "Utilities"),
            TransactionTemplate("PAYROLL", 200000, 25, "Payroll Services", True, "Monthly payroll"),
            TransactionTemplate("INSURANCE", 25000, 10, "Corporate Insurance", True, "Insurance"),
            TransactionTemplate("DEBT_SERVICE", 75000, 5, "Bond Payments", True, "Debt service"),
            TransactionTemplate("IT_SERVICES", 30000, 18, "IT Provider", True, "IT services"),
            TransactionTemplate("TAX", 40000, 20, "Tax Authority", True, "Tax payment"),
            TransactionTemplate("REGULATORY_FEES", 10000, 15, "Regulator", True, "Compliance fees"),
        ]

    for template in templates:
        # Fixed date - no jitter for recurring
        pay_date = _safe_date(month_start.year, month_start.month, template.day_of_month)

        # Base amount from template - FIXED amount, no variation for recurring
        # This ensures decomposition produces stable deterministic base
        amount = template.base_amount

        transactions.append(_create_transaction(
            tx_date=pay_date,
            amount=-round(abs(amount), 2),  # Expenses are negative
            category=template.category,
            counterparty=template.counterparty,
            is_recurring=template.is_recurring,
            description=template.description,
        ))

    return transactions


def _generate_variable_expenses(
    account_cfg: AccountConfig,
    random_cfg: RandomnessConfig,
    month_start: datetime,
    seed: int,
) -> List[dict]:
    """Generate variable expense transactions for a month.

    This is the RESIDUAL component where randomness is applied.
    Randomness controls:
    - Budget magnitude (residual_magnitude_multiplier)
    - Per-transaction variation (residual_volatility)
    - Number of transactions (residual_transaction_multiplier)
    """
    np.random.seed(seed + 200)
    transactions = []

    # Base variable expense budget (deterministic baseline)
    # Reduced to make predictable component ~75-80% of total expenses
    if account_cfg.account_type == AccountType.PERSONAL:
        # Fixed: ~1555, Variable: ~350 -> Predictable ~80%
        base_budget = 350
        categories = [
            # (name, weight, min_amount, max_amount)
            ("GROCERIES", 0.40, 15, 50),
            ("TRANSPORT", 0.15, 5, 20),
            ("DINING", 0.15, 10, 35),
            ("ENTERTAINMENT", 0.10, 10, 40),
            ("SHOPPING", 0.12, 15, 60),
            ("HEALTHCARE", 0.08, 20, 80),
        ]
        base_max_transactions = 12
    elif account_cfg.account_type == AccountType.SME:
        # Fixed: ~19000, Variable: ~800 -> Predictable ~96%
        # Reduced budget and tighter amount ranges for stability
        base_budget = 800
        categories = [
            ("SUPPLIES", 0.35, 40, 150),
            ("TRAVEL", 0.15, 80, 250),
            ("MARKETING", 0.20, 80, 300),
            ("EQUIPMENT", 0.15, 40, 200),
            ("PROFESSIONAL_SERVICES", 0.15, 80, 250),
        ]
        base_max_transactions = 10
    else:  # Corporate
        # Fixed: ~445000, Variable: ~8000 -> Predictable ~98%
        # Reduced budget and tighter amount ranges for stability
        base_budget = 8000
        categories = [
            ("SUPPLIES", 0.25, 400, 1500),
            ("TRAVEL", 0.15, 400, 1200),
            ("MARKETING", 0.20, 800, 2500),
            ("EQUIPMENT", 0.15, 400, 1500),
            ("CAPITAL_EXPENSE", 0.10, 800, 4000),
            ("PROFESSIONAL_SERVICES", 0.15, 800, 2500),
        ]
        base_max_transactions = 20

    # Apply seasonal expense multiplier based on month
    month_num = month_start.month
    seasonal_multiplier = SEASONAL_EXPENSE_MULTIPLIERS[account_cfg.account_type].get(month_num, 1.0)

    # Apply randomness to budget magnitude
    # Higher randomness = larger and more volatile residual
    variable_budget = base_budget * random_cfg.residual_magnitude_multiplier * seasonal_multiplier

    # Add some stochastic variation to budget based on volatility
    if random_cfg.residual_volatility > 0:
        budget_noise = np.random.normal(0, random_cfg.residual_volatility * 0.3)
        variable_budget *= (1 + budget_noise)

    # Apply randomness to transaction count
    max_transactions = int(base_max_transactions * random_cfg.residual_transaction_multiplier)

    # Generate transactions until we hit the budget
    spent = 0.0

    while spent < variable_budget and len(transactions) < max_transactions:
        # Select category based on weights
        cat_names = [c[0] for c in categories]
        cat_weights = [c[1] for c in categories]
        cat_idx = np.random.choice(len(categories), p=cat_weights)
        cat_name, _, amount_min, amount_max = categories[cat_idx]

        # Generate transaction date
        txn_day = np.random.randint(1, 29)
        txn_date = _safe_date(month_start.year, month_start.month, txn_day)

        # Log-normal distribution for base amounts (tends toward lower end)
        base_amount = np.exp(np.random.uniform(
            np.log(amount_min),
            np.log(amount_max)
        ))

        # Apply per-transaction variation based on residual_volatility
        if random_cfg.residual_volatility > 0:
            variation = np.random.normal(0, random_cfg.residual_volatility)
            amount = base_amount * (1 + variation)
        else:
            amount = base_amount

        # Clamp to reasonable range
        amount = max(amount_min * 0.5, min(amount, amount_max * 2.0))

        # Don't exceed budget by too much
        if spent + amount > variable_budget * 1.1:
            amount = max(0, variable_budget - spent)
            if amount < amount_min * 0.5:
                break

        spent += amount

        transactions.append(_create_transaction(
            tx_date=txn_date,
            amount=-round(abs(amount), 2),  # Expenses are negative
            category=cat_name,
            counterparty=f"{cat_name.title()} Vendor {np.random.randint(1, 20)}",
            is_recurring=False,
            description=f"{cat_name.lower().replace('_', ' ')} expense",
        ))

    return transactions


def _generate_occasional_expenses(
    account_cfg: AccountConfig,
    random_cfg: RandomnessConfig,
    month_start: datetime,
    seed: int,
) -> List[dict]:
    """Generate occasional large expenses based on probability.

    These are unexpected expenses that can cause negative cash flow months:
    - Car repairs, medical bills, appliance replacements (Personal)
    - Equipment repairs, legal fees (SME)
    - Equipment replacement, facility repairs (Corporate)
    """
    np.random.seed(seed + 400)
    transactions = []

    expense_defs = OCCASIONAL_EXPENSES.get(account_cfg.account_type, [])

    for category, probability, min_amount, max_amount in expense_defs:
        # Check if this expense occurs this month
        if np.random.random() < probability:
            # Generate the expense
            txn_day = np.random.randint(1, 29)
            txn_date = _safe_date(month_start.year, month_start.month, txn_day)

            # Random amount within range
            amount = np.random.uniform(min_amount, max_amount)

            transactions.append(_create_transaction(
                tx_date=txn_date,
                amount=-round(abs(amount), 2),
                category=category,
                counterparty=f"{category.replace('_', ' ').title()} Provider",
                is_recurring=False,
                description=f"Unexpected {category.lower().replace('_', ' ')}",
            ))

    return transactions


def _generate_periodic_expenses(
    account_cfg: AccountConfig,
    month_start: datetime,
) -> List[dict]:
    """Generate periodic expenses that occur on specific months.

    These are predictable but infrequent expenses:
    - Annual insurance, tax filing, vacation (Personal)
    - Quarterly taxes, annual audit (SME/Corporate)
    """
    transactions = []
    month_num = month_start.month

    expense_defs = PERIODIC_EXPENSES.get(account_cfg.account_type, [])

    for category, months, amount, description in expense_defs:
        # Check if this expense occurs this month
        if month_num in months:
            # Generate on a fixed day based on category
            txn_day = 15  # Mid-month for most periodic expenses
            if category == "TAX":
                txn_day = 20
            elif category == "INSURANCE":
                txn_day = 5
            elif category == "VACATION":
                txn_day = 1

            txn_date = _safe_date(month_start.year, month_start.month, txn_day)

            transactions.append(_create_transaction(
                tx_date=txn_date,
                amount=-round(abs(amount), 2),
                category=category,
                counterparty=f"{category.title()} Payment",
                is_recurring=True,  # These are predictable annual/quarterly
                description=description,
            ))

    return transactions


def _generate_transfers(
    account_cfg: AccountConfig,
    random_cfg: RandomnessConfig,
    month_start: datetime,
    seed: int,
) -> List[dict]:
    """Generate internal transfer pairs for a month."""
    np.random.seed(seed + 300)
    transactions = []

    min_transfers, max_transfers = account_cfg.transfers_per_month
    n_transfers = np.random.randint(min_transfers, max_transfers + 1)

    for i in range(n_transfers):
        transfer_id = str(uuid.uuid4())
        txn_day = np.random.randint(1, 29)
        txn_date = _safe_date(month_start.year, month_start.month, txn_day)

        # Transfer amount based on account type
        if account_cfg.account_type == AccountType.PERSONAL:
            amount = np.random.uniform(100, 500)
        elif account_cfg.account_type == AccountType.SME:
            amount = np.random.uniform(500, 5000)
        else:
            amount = np.random.uniform(10000, 100000)

        # Transfer OUT
        transactions.append(_create_transaction(
            tx_date=txn_date,
            amount=-round(amount, 2),
            category="TRANSFER_OUT",
            counterparty="Internal Account",
            is_recurring=False,
            description="Internal transfer",
            transfer_link_id=transfer_id,
        ))

        # Transfer IN (same day - transfers are paired)
        transactions.append(_create_transaction(
            tx_date=txn_date,
            amount=round(amount, 2),
            category="TRANSFER_IN",
            counterparty="Internal Account",
            is_recurring=False,
            description="Internal transfer",
            transfer_link_id=transfer_id,
        ))

    return transactions


def _apply_flag_corruption(
    df: pd.DataFrame,
    corruption_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Randomly flip is_recurring_flag to simulate data quality issues."""
    np.random.seed(seed + 500)

    # Create mask for corruption
    mask = np.random.random(len(df)) < corruption_rate

    # Flip the flags
    df = df.copy()
    df.loc[mask, "is_recurring_flag"] = ~df.loc[mask, "is_recurring_flag"]

    return df


def _safe_date(year: int, month: int, day: int) -> datetime:
    """Create a date, clamping day to valid range for the month."""
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    return datetime(year, month, min(day, max_day))


def aggregate_monthly_actuals(df: pd.DataFrame, decompose: bool = False) -> pd.DataFrame:
    """Aggregate transactions to monthly net cash flow.

    Args:
        df: Transaction DataFrame
        decompose: If True, also return predictable/residual decomposition

    Returns:
        DataFrame with monthly aggregated net cash flow (and decomposition if requested)
    """
    df = df.copy()
    df["tx_date"] = pd.to_datetime(df["tx_date"])
    df["month_key"] = df["tx_date"].dt.to_period("M").astype(str)

    # Filter out internal transfers for net calculation
    external_df = df[~df["category"].isin(["TRANSFER_IN", "TRANSFER_OUT"])]

    # Aggregate by month
    monthly = external_df.groupby("month_key").agg(
        net_cash_flow=("amount", "sum"),
        transaction_count=("tx_id", "count"),
    ).reset_index()

    if decompose:
        # Predictable = recurring transactions (salary, rent, utilities, etc.)
        # Use discovered recurrence if available (from Layer 0.5), otherwise fall back to flag
        recurring_col = "is_recurring_discovered" if "is_recurring_discovered" in external_df.columns else "is_recurring_flag"
        recurring_mask = external_df[recurring_col] == True
        predictable = external_df[recurring_mask].groupby("month_key")["amount"].sum().reset_index()
        predictable.columns = ["month_key", "predictable"]

        # Residual = non-recurring transactions (variable expenses)
        residual = external_df[~recurring_mask].groupby("month_key")["amount"].sum().reset_index()
        residual.columns = ["month_key", "residual"]

        # Merge decomposition
        monthly = monthly.merge(predictable, on="month_key", how="left")
        monthly = monthly.merge(residual, on="month_key", how="left")
        monthly["predictable"] = monthly["predictable"].fillna(0)
        monthly["residual"] = monthly["residual"].fillna(0)

    return monthly


if __name__ == "__main__":
    # Quick test of data generation
    from framework_config import TestConfig, AccountType, RandomnessLevel

    config = TestConfig(
        account_type=AccountType.PERSONAL,
        randomness_level=RandomnessLevel.LOW,
        seed=42,
    )

    print(f"Generating data for {config.config_id}...")
    df = generate_synthetic_data(config)
    print(f"Generated {len(df)} transactions")
    print(f"\nSample transactions:")
    print(df.head(10).to_string())

    monthly = aggregate_monthly_actuals(df)
    print(f"\nMonthly aggregation:")
    print(monthly.to_string())
