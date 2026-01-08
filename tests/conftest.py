"""Shared pytest fixtures for Cash Flow Forecasting Engine tests."""

import pandas as pd
import numpy as np
import pytest
from datetime import date, timedelta
from pathlib import Path


# Fixture directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_utf_data() -> pd.DataFrame:
    """Generate synthetic UTF dataset for testing.

    Creates 24 months of transaction data with:
    - Recurring income (salary)
    - Recurring expenses (rent, utilities)
    - Variable expenses (groceries, entertainment)
    - Internal transfers (to be netted)
    - Outliers (controlled anomalies)
    """
    np.random.seed(42)

    records = []
    tx_id = 1

    # Generate 24 months of data
    start_date = date(2024, 1, 1)

    for month_offset in range(24):
        month_date = start_date + timedelta(days=month_offset * 30)
        year = month_date.year
        month = ((start_date.month - 1 + month_offset) % 12) + 1
        if month == 1 and month_offset > 0:
            year = start_date.year + ((start_date.month + month_offset - 1) // 12)

        month_key = f"{year}-{month:02d}"

        # Salary (recurring, 25th of month)
        records.append({
            "tx_id": f"T{tx_id:06d}",
            "customer_id": "CUST001",
            "account_id": "ACC001",
            "tx_date": date(year, month, 25),
            "amount": 3000.0 + np.random.normal(0, 50),
            "currency": "EUR",
            "direction": "CREDIT",
            "category": "SALARY",
            "description_raw": f"SALARY {month_key}",
            "is_recurring_flag": True,
            "is_variable_amount": False,
        })
        tx_id += 1

        # Rent (recurring, 1st of month)
        records.append({
            "tx_id": f"T{tx_id:06d}",
            "customer_id": "CUST001",
            "account_id": "ACC001",
            "tx_date": date(year, month, 1),
            "amount": -1200.0,
            "currency": "EUR",
            "direction": "DEBIT",
            "category": "RENT_MORTGAGE",
            "description_raw": f"RENT {month_key}",
            "is_recurring_flag": True,
            "is_variable_amount": False,
        })
        tx_id += 1

        # Utilities (recurring with seasonal variation)
        utility_base = -120.0
        if month in [1, 2, 11, 12]:  # Winter months
            utility_base *= 1.5
        records.append({
            "tx_id": f"T{tx_id:06d}",
            "customer_id": "CUST001",
            "account_id": "ACC001",
            "tx_date": date(year, month, 5),
            "amount": utility_base + np.random.normal(0, 10),
            "currency": "EUR",
            "direction": "DEBIT",
            "category": "UTILITIES",
            "description_raw": f"UTILITIES {month_key}",
            "is_recurring_flag": True,
            "is_variable_amount": True,
        })
        tx_id += 1

        # Groceries (variable, ~4 per month)
        for _ in range(4):
            records.append({
                "tx_id": f"T{tx_id:06d}",
                "customer_id": "CUST001",
                "account_id": "ACC001",
                "tx_date": date(year, month, np.random.randint(1, 28)),
                "amount": -100.0 + np.random.normal(0, 20),
                "currency": "EUR",
                "direction": "DEBIT",
                "category": "GROCERIES",
                "description_raw": f"GROCERY STORE {np.random.randint(1000, 9999)}",
                "is_recurring_flag": False,
                "is_variable_amount": True,
            })
            tx_id += 1

        # Internal transfer to savings (to be netted)
        records.append({
            "tx_id": f"T{tx_id:06d}",
            "customer_id": "CUST001",
            "account_id": "ACC001",
            "tx_date": date(year, month, 26),
            "amount": -500.0,
            "currency": "EUR",
            "direction": "DEBIT",
            "category": "SAVINGS_CONTRIBUTION",
            "description_raw": "TRANSFER TO SAVINGS",
            "is_recurring_flag": True,
            "is_variable_amount": False,
        })
        tx_id += 1

    # Add controlled outliers
    # Large tax refund in August 2024
    records.append({
        "tx_id": f"T{tx_id:06d}",
        "customer_id": "CUST001",
        "account_id": "ACC001",
        "tx_date": date(2024, 8, 15),
        "amount": 5000.0,
        "currency": "EUR",
        "direction": "CREDIT",
        "category": "ONE_TIME_LARGE",
        "description_raw": "TAX REFUND",
        "is_recurring_flag": False,
        "is_variable_amount": False,
    })
    tx_id += 1

    # Vacation expenses in July 2024 and July 2025
    for year in [2024, 2025]:
        records.append({
            "tx_id": f"T{tx_id:06d}",
            "customer_id": "CUST001",
            "account_id": "ACC001",
            "tx_date": date(year, 7, 10),
            "amount": -1800.0,
            "currency": "EUR",
            "direction": "DEBIT",
            "category": "TRAVEL",
            "description_raw": "VACATION EXPENSE",
            "is_recurring_flag": False,
            "is_variable_amount": False,
        })
        tx_id += 1

    return pd.DataFrame(records)


@pytest.fixture
def sample_crf_data() -> pd.DataFrame:
    """Generate sample CRF data for testing."""
    return pd.DataFrame([
        {
            "counterparty_key": "CPK_SALARY",
            "customer_id": "CUST001",
            "display_name": "Employer Corp",
            "contract_type": "GENERIC",
            "contractual_amount": 3000.0,
            "recurrence_end_date": None,
            "is_variable_amount": False,
        },
        {
            "counterparty_key": "CPK_RENT",
            "customer_id": "CUST001",
            "display_name": "Landlord",
            "contract_type": "MANDATE",
            "contractual_amount": -1200.0,
            "recurrence_end_date": None,
            "is_variable_amount": False,
        },
        {
            "counterparty_key": "CPK_LOAN",
            "customer_id": "CUST001",
            "display_name": "Personal Loan",
            "contract_type": "LOAN",
            "contractual_amount": -200.0,
            "recurrence_end_date": "2026-06-01",  # Ends during forecast
            "is_variable_amount": False,
        },
    ])


@pytest.fixture
def sample_monthly_data() -> pd.DataFrame:
    """Generate pre-aggregated monthly NECF data."""
    months = pd.date_range("2024-01", "2025-12", freq="MS")
    np.random.seed(42)

    data = []
    for month in months:
        # Base NECF with some seasonality
        base = 500 + 100 * np.sin(2 * np.pi * month.month / 12)
        necf = base + np.random.normal(0, 50)

        data.append({
            "customer_id": "CUST001",
            "month_key": month.strftime("%Y-%m"),
            "necf": necf,
            "credit_total": 3000.0,
            "debit_total": -(3000.0 - necf),
            "transaction_count": 10,
        })

    return pd.DataFrame(data)


@pytest.fixture
def sample_decomposed_data(sample_monthly_data) -> pd.DataFrame:
    """Generate decomposed NECF data with deterministic and residual."""
    df = sample_monthly_data.copy()

    # Fixed deterministic base (salary - rent - utilities)
    df["deterministic_base"] = 3000 - 1200 - 150  # = 1650

    # Residual is what remains
    df["residual"] = df["necf"] - df["deterministic_base"]

    return df


@pytest.fixture
def sample_time_series() -> pd.Series:
    """Generate a simple time series for model testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01", "2025-12", freq="MS")

    # Trend + seasonality + noise
    trend = np.linspace(400, 500, len(dates))
    seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 20, len(dates))

    values = trend + seasonal + noise

    series = pd.Series(values, index=dates.to_period("M"))
    series.name = "residual"

    return series
