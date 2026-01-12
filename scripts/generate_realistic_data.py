"""Enhanced synthetic data generator for production-like validation.

Phase 3.1a: Creates synthetic transaction data with realistic characteristics:
- 100+ unique counterparties
- Realistic category distributions (not uniform)
- Multi-account scenarios
- Missing data patterns (random and systematic)
- Transaction frequency distributions from banking benchmarks
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CustomerProfile:
    """Profile defining a customer's transaction patterns."""
    name: str
    monthly_income_range: Tuple[float, float]  # Min/max monthly income
    income_sources: int  # Number of income sources (1=single job, 2+=multiple)
    recurring_expense_count: int  # Number of recurring expenses
    avg_transactions_per_month: int  # Average transaction count
    has_savings_account: bool
    has_credit_card: bool
    loan_probability: float  # Probability of having a loan
    missing_data_rate: float  # Rate of missing optional fields


# Customer profiles representing different segments
CUSTOMER_PROFILES = [
    CustomerProfile("Young Professional", (2000, 4000), 1, 5, 30, True, True, 0.3, 0.05),
    CustomerProfile("Family", (4000, 8000), 2, 10, 60, True, True, 0.6, 0.03),
    CustomerProfile("Retiree", (1500, 3000), 1, 4, 20, True, False, 0.1, 0.08),
    CustomerProfile("Student", (500, 1500), 1, 3, 25, False, False, 0.2, 0.10),
    CustomerProfile("High Earner", (8000, 15000), 1, 8, 50, True, True, 0.4, 0.02),
]


# Realistic counterparty database (100+ unique)
COUNTERPARTIES = {
    "SALARY": [
        "ACME CORPORATION", "TECH SOLUTIONS INC", "HEALTHCARE PARTNERS",
        "GLOBAL RETAIL GROUP", "FINANCIAL SERVICES CO", "EDUCATION BOARD",
        "CONSULTING FIRM LLP", "MANUFACTURING PLUS", "STARTUP VENTURES",
        "GOVERNMENT AGENCY", "NONPROFIT FOUNDATION", "UNIVERSITY PAYROLL",
    ],
    "RENT_MORTGAGE": [
        "CITY PROPERTIES LTD", "HOME MORTGAGE BANK", "RENTAL MANAGEMENT CO",
        "HOUSING ASSOCIATION", "LANDLORD SERVICES", "PROPERTY TRUST",
    ],
    "UTILITIES": [
        "ENERGY PROVIDER A", "WATER UTILITY CO", "GAS SUPPLY INC",
        "ELECTRIC COMPANY", "MUNICIPAL UTILITIES", "GREEN ENERGY LTD",
    ],
    "GROCERIES": [
        "SUPERMARKET CHAIN A", "ORGANIC FOODS MARKET", "DISCOUNT GROCERY",
        "WHOLESALE CLUB", "NEIGHBORHOOD MARKET", "FRESH PRODUCE STORE",
        "ONLINE GROCERY", "SPECIALTY FOODS", "LOCAL BUTCHER",
        "BAKERY DELIGHTS", "FARMERS MARKET", "CONVENIENCE STORE",
    ],
    "SUBSCRIPTION": [
        "STREAMING SERVICE A", "STREAMING SERVICE B", "MUSIC UNLIMITED",
        "NEWS SUBSCRIPTION", "FITNESS APP", "CLOUD STORAGE",
        "GAMING SERVICE", "PRODUCTIVITY SUITE", "VPN SERVICE",
    ],
    "INSURANCE": [
        "LIFE INSURANCE CO", "AUTO INSURANCE INC", "HEALTH INSURANCE",
        "HOME INSURANCE LTD", "TRAVEL INSURANCE", "PET INSURANCE",
    ],
    "LOAN_PAYMENT": [
        "CAR FINANCE CO", "PERSONAL LOANS INC", "STUDENT LOAN AGENCY",
        "MORTGAGE BANK A", "CREDIT UNION LOANS",
    ],
    "RESTAURANT": [
        "FAST FOOD CHAIN A", "ITALIAN RESTAURANT", "CHINESE TAKEAWAY",
        "COFFEE SHOP CHAIN", "LOCAL DINER", "SUSHI PLACE",
        "BURGER JOINT", "PIZZA DELIVERY", "INDIAN CUISINE",
        "THAI RESTAURANT", "MEXICAN GRILL", "SANDWICH SHOP",
    ],
    "TRANSPORT": [
        "PUBLIC TRANSIT", "RIDE SHARE APP", "TAXI SERVICE",
        "FUEL STATION A", "FUEL STATION B", "PARKING SERVICES",
        "TOLL ROADS", "CAR RENTAL", "BIKE SHARE",
    ],
    "SHOPPING": [
        "ELECTRONICS STORE", "CLOTHING RETAILER A", "CLOTHING RETAILER B",
        "HOME GOODS STORE", "SPORTS EQUIPMENT", "BOOKSTORE",
        "PHARMACY CHAIN", "PET STORE", "HARDWARE STORE",
        "DEPARTMENT STORE", "ONLINE MARKETPLACE", "DISCOUNT RETAILER",
    ],
    "HEALTHCARE": [
        "MEDICAL CLINIC", "DENTAL OFFICE", "PHARMACY",
        "HOSPITAL SERVICES", "OPTICIAN", "PHYSIOTHERAPY",
    ],
    "ENTERTAINMENT": [
        "CINEMA CHAIN", "CONCERT VENUE", "SPORTS TICKETS",
        "MUSEUM ENTRY", "THEME PARK", "BOWLING ALLEY",
    ],
    "MISCELLANEOUS": [
        "ATM WITHDRAWAL", "CASH DEPOSIT", "BANK FEE",
        "FOREIGN EXCHANGE", "WIRE TRANSFER", "CHECK DEPOSIT",
    ],
}


# Category probability distribution (realistic, not uniform)
CATEGORY_PROBABILITIES = {
    "GROCERIES": 0.25,      # Most frequent
    "RESTAURANT": 0.15,
    "TRANSPORT": 0.12,
    "SHOPPING": 0.10,
    "MISCELLANEOUS": 0.08,
    "UTILITIES": 0.06,
    "SUBSCRIPTION": 0.05,
    "HEALTHCARE": 0.05,
    "ENTERTAINMENT": 0.04,
    "INSURANCE": 0.04,
    "RENT_MORTGAGE": 0.03,  # Monthly, so low frequency
    "LOAN_PAYMENT": 0.02,
    "SALARY": 0.01,         # 1-2 per month
}


@dataclass
class RealisticDataConfig:
    """Configuration for realistic data generation."""
    num_customers: int = 1
    months_history: int = 24
    profile: Optional[CustomerProfile] = None
    seed: int = 42
    # Missing data patterns
    missing_description_rate: float = 0.15
    missing_counterparty_rate: float = 0.10
    systematic_missing_months: List[int] = field(default_factory=list)  # Months with gaps


def generate_realistic_data(config: RealisticDataConfig) -> pd.DataFrame:
    """Generate realistic synthetic transaction data.

    Args:
        config: Configuration for data generation

    Returns:
        DataFrame with realistic transactions
    """
    np.random.seed(config.seed)

    # Select profile if not specified
    profile = config.profile or np.random.choice(CUSTOMER_PROFILES)

    transactions = []
    tx_id = 1

    for customer_idx in range(config.num_customers):
        customer_id = f"CUST{customer_idx + 1:05d}"

        # Determine accounts for this customer
        accounts = [f"{customer_id}_CHECKING"]
        if profile.has_savings_account:
            accounts.append(f"{customer_id}_SAVINGS")

        # Generate base income
        monthly_income = np.random.uniform(*profile.monthly_income_range)

        # Generate recurring expenses (fixed amounts)
        recurring_expenses = []
        for i in range(profile.recurring_expense_count):
            category = np.random.choice(
                ["RENT_MORTGAGE", "UTILITIES", "SUBSCRIPTION", "INSURANCE", "LOAN_PAYMENT"],
                p=[0.2, 0.3, 0.25, 0.15, 0.10]
            )
            if category == "RENT_MORTGAGE":
                amount = np.random.uniform(800, 2000)
            elif category == "UTILITIES":
                amount = np.random.uniform(50, 200)
            elif category == "SUBSCRIPTION":
                amount = np.random.uniform(5, 50)
            elif category == "INSURANCE":
                amount = np.random.uniform(50, 300)
            else:  # LOAN_PAYMENT
                amount = np.random.uniform(100, 500)

            recurring_expenses.append({
                "category": category,
                "amount": amount,
                "day": np.random.randint(1, 28),
                "counterparty": np.random.choice(COUNTERPARTIES.get(category, ["UNKNOWN"])),
            })

        # Generate months of data
        for month_idx in range(config.months_history):
            year = 2024 + month_idx // 12
            month = (month_idx % 12) + 1

            # Check for systematic missing month
            if month_idx + 1 in config.systematic_missing_months:
                continue

            # Generate salary (income)
            for income_idx in range(profile.income_sources):
                salary = monthly_income / profile.income_sources
                salary += np.random.normal(0, salary * 0.02)  # 2% noise

                counterparty = np.random.choice(COUNTERPARTIES["SALARY"])

                # Random missing description
                description = f"SALARY PAYMENT {year}-{month:02d}"
                if np.random.random() < config.missing_description_rate:
                    description = None

                transactions.append({
                    "tx_id": f"TX{tx_id:08d}",
                    "customer_id": customer_id,
                    "account_id": accounts[0],  # Main checking
                    "tx_date": datetime(year, month, np.random.randint(1, 5)),
                    "amount": salary,
                    "currency": "EUR",
                    "direction": "CREDIT",
                    "category": "SALARY",
                    "description_raw": description,
                    "counterparty_key": counterparty if np.random.random() > config.missing_counterparty_rate else None,
                    "is_recurring_flag": True,
                    "is_variable_amount": False,
                })
                tx_id += 1

            # Generate recurring expenses
            for expense in recurring_expenses:
                # Add seasonal variation for utilities
                amount = expense["amount"]
                if expense["category"] == "UTILITIES" and month in [11, 12, 1, 2]:
                    amount *= 1.4  # Winter increase

                description = f"{expense['category']} {year}-{month:02d}"
                if np.random.random() < config.missing_description_rate:
                    description = None

                transactions.append({
                    "tx_id": f"TX{tx_id:08d}",
                    "customer_id": customer_id,
                    "account_id": accounts[0],
                    "tx_date": datetime(year, month, expense["day"]),
                    "amount": -amount,
                    "currency": "EUR",
                    "direction": "DEBIT",
                    "category": expense["category"],
                    "description_raw": description,
                    "counterparty_key": expense["counterparty"] if np.random.random() > config.missing_counterparty_rate else None,
                    "is_recurring_flag": True,
                    "is_variable_amount": expense["category"] == "UTILITIES",
                })
                tx_id += 1

            # Generate variable expenses based on profile
            num_variable = np.random.poisson(profile.avg_transactions_per_month - profile.recurring_expense_count)

            categories = list(CATEGORY_PROBABILITIES.keys())
            probs = list(CATEGORY_PROBABILITIES.values())
            # Normalize probabilities
            probs = np.array(probs) / sum(probs)

            for _ in range(num_variable):
                category = np.random.choice(categories, p=probs)

                # Skip income categories (handled above)
                if category in ["SALARY", "RENT_MORTGAGE", "LOAN_PAYMENT", "INSURANCE"]:
                    category = "SHOPPING"

                # Generate amount based on category
                if category == "GROCERIES":
                    amount = np.random.lognormal(3.5, 0.5)  # Mean ~40, skewed
                elif category == "RESTAURANT":
                    amount = np.random.lognormal(2.7, 0.6)  # Mean ~20, skewed
                elif category == "TRANSPORT":
                    amount = np.random.lognormal(2.0, 0.8)  # Mean ~10
                elif category == "SHOPPING":
                    amount = np.random.lognormal(3.2, 0.9)  # Mean ~35, high variance
                elif category == "HEALTHCARE":
                    amount = np.random.lognormal(3.5, 0.7)  # Mean ~45
                elif category == "ENTERTAINMENT":
                    amount = np.random.lognormal(3.0, 0.6)  # Mean ~25
                elif category == "UTILITIES":
                    amount = np.random.lognormal(3.8, 0.4)  # Mean ~50
                elif category == "SUBSCRIPTION":
                    amount = np.random.lognormal(2.5, 0.5)  # Mean ~15
                else:
                    amount = np.random.lognormal(3.0, 0.7)  # Default

                counterparties = COUNTERPARTIES.get(category, COUNTERPARTIES["MISCELLANEOUS"])
                counterparty = np.random.choice(counterparties)

                day = np.random.randint(1, 28)
                description = f"{counterparty} PURCHASE"
                if np.random.random() < config.missing_description_rate:
                    description = None

                transactions.append({
                    "tx_id": f"TX{tx_id:08d}",
                    "customer_id": customer_id,
                    "account_id": accounts[0],
                    "tx_date": datetime(year, month, day),
                    "amount": -amount,
                    "currency": "EUR",
                    "direction": "DEBIT",
                    "category": category,
                    "description_raw": description,
                    "counterparty_key": counterparty if np.random.random() > config.missing_counterparty_rate else None,
                    "is_recurring_flag": False,
                    "is_variable_amount": True,
                })
                tx_id += 1

            # Generate savings transfer if applicable
            if profile.has_savings_account and len(accounts) > 1:
                savings_amount = monthly_income * np.random.uniform(0.05, 0.15)

                # Outgoing from checking
                transactions.append({
                    "tx_id": f"TX{tx_id:08d}",
                    "customer_id": customer_id,
                    "account_id": accounts[0],
                    "tx_date": datetime(year, month, 15),
                    "amount": -savings_amount,
                    "currency": "EUR",
                    "direction": "DEBIT",
                    "category": "TRANSFER_OUT",
                    "description_raw": "SAVINGS TRANSFER",
                    "counterparty_key": None,
                    "is_recurring_flag": True,
                    "is_variable_amount": True,
                })
                tx_id += 1

                # Incoming to savings
                transactions.append({
                    "tx_id": f"TX{tx_id:08d}",
                    "customer_id": customer_id,
                    "account_id": accounts[1],
                    "tx_date": datetime(year, month, 15),
                    "amount": savings_amount,
                    "currency": "EUR",
                    "direction": "CREDIT",
                    "category": "TRANSFER_IN",
                    "description_raw": "SAVINGS TRANSFER",
                    "counterparty_key": None,
                    "is_recurring_flag": True,
                    "is_variable_amount": True,
                })
                tx_id += 1

            # Occasional outliers (tax refund, large purchase, etc.)
            if np.random.random() < 0.04:  # ~1 per year
                if np.random.random() < 0.5:
                    # Tax refund (credit)
                    amount = np.random.uniform(500, 3000)
                    transactions.append({
                        "tx_id": f"TX{tx_id:08d}",
                        "customer_id": customer_id,
                        "account_id": accounts[0],
                        "tx_date": datetime(year, month, np.random.randint(10, 25)),
                        "amount": amount,
                        "currency": "EUR",
                        "direction": "CREDIT",
                        "category": "TAX_REFUND",
                        "description_raw": "TAX REFUND",
                        "counterparty_key": "TAX AUTHORITY",
                        "is_recurring_flag": False,
                        "is_variable_amount": False,
                    })
                else:
                    # Large purchase (debit)
                    amount = np.random.uniform(500, 2000)
                    transactions.append({
                        "tx_id": f"TX{tx_id:08d}",
                        "customer_id": customer_id,
                        "account_id": accounts[0],
                        "tx_date": datetime(year, month, np.random.randint(10, 25)),
                        "amount": -amount,
                        "currency": "EUR",
                        "direction": "DEBIT",
                        "category": "SHOPPING",
                        "description_raw": "LARGE PURCHASE",
                        "counterparty_key": np.random.choice(COUNTERPARTIES["SHOPPING"]),
                        "is_recurring_flag": False,
                        "is_variable_amount": False,
                    })
                tx_id += 1

    df = pd.DataFrame(transactions)
    df["tx_date"] = pd.to_datetime(df["tx_date"])
    return df


def print_data_stats(df: pd.DataFrame) -> None:
    """Print statistics about generated data."""
    print("\n--- Data Statistics ---")
    print(f"Total transactions: {len(df)}")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    print(f"Unique accounts: {df['account_id'].nunique()}")
    print(f"Unique counterparties: {df['counterparty_key'].nunique()}")
    print(f"Date range: {df['tx_date'].min()} to {df['tx_date'].max()}")

    print("\nCategory distribution:")
    cat_dist = df['category'].value_counts(normalize=True)
    for cat, pct in cat_dist.head(10).items():
        print(f"  {cat}: {pct:.1%}")

    print("\nMissing data:")
    for col in ["description_raw", "counterparty_key"]:
        missing_pct = df[col].isna().mean()
        print(f"  {col}: {missing_pct:.1%} missing")

    print("\nMonthly transaction counts:")
    monthly = df.groupby(df['tx_date'].dt.to_period('M')).size()
    print(f"  Mean: {monthly.mean():.1f}, Std: {monthly.std():.1f}")
    print(f"  Min: {monthly.min()}, Max: {monthly.max()}")


def main():
    """Generate and save realistic synthetic data."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate realistic synthetic transaction data")
    parser.add_argument(
        "--customers", "-n",
        type=int,
        default=1,
        help="Number of customers to generate (default: 1)"
    )
    parser.add_argument(
        "--months", "-m",
        type=int,
        default=24,
        help="Months of history (default: 24)"
    )
    parser.add_argument(
        "--profile", "-p",
        choices=["young_professional", "family", "retiree", "student", "high_earner"],
        default=None,
        help="Customer profile to use"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--missing-gaps",
        type=str,
        default="",
        help="Comma-separated list of months with missing data (e.g., '5,12,18')"
    )
    args = parser.parse_args()

    # Parse profile
    profile = None
    if args.profile:
        profile_map = {
            "young_professional": CUSTOMER_PROFILES[0],
            "family": CUSTOMER_PROFILES[1],
            "retiree": CUSTOMER_PROFILES[2],
            "student": CUSTOMER_PROFILES[3],
            "high_earner": CUSTOMER_PROFILES[4],
        }
        profile = profile_map.get(args.profile)

    # Parse missing gaps
    missing_months = []
    if args.missing_gaps:
        missing_months = [int(m.strip()) for m in args.missing_gaps.split(",")]

    config = RealisticDataConfig(
        num_customers=args.customers,
        months_history=args.months,
        profile=profile,
        seed=args.seed,
        systematic_missing_months=missing_months,
    )

    print("=" * 60)
    print("REALISTIC DATA GENERATOR")
    print("=" * 60)
    print(f"\nGenerating data for {args.customers} customer(s), {args.months} months...")
    if profile:
        print(f"Profile: {profile.name}")
    if missing_months:
        print(f"Missing months: {missing_months}")

    df = generate_realistic_data(config)
    print_data_stats(df)

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    else:
        # Default output
        output_dir = Path(__file__).parent.parent / "data" / "synthetic"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "realistic_utf.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
