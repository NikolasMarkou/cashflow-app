import pandas as pd
import numpy as np
from datetime import date, timedelta
import random
import os

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

OUTPUT_PATH = "data/PoC_UTF_Dataset.csv"

START_YEAR = 2024
END_YEAR = 2025

ACCOUNT_ID = "MAIN_CHECKING"
CURRENCY = "EUR"

# Ensure deterministic output
random.seed(42)

# ----------------------------------------------------------
# CATEGORY DEFINITIONS
# ----------------------------------------------------------
# Drift values simulate inflation on variable categories.

CATEGORIES = {
    "SALARY": {
        "base": 3000.00,
        "drift_pct": 0.002,
        "day": 25,
        "recurring": True,
        "type": "CREDIT"
    },
    "RENT_MORTGAGE": {
        "base": -1200.00,
        "drift_pct": 0.0,
        "day": 1,
        "recurring": True,
        "type": "DEBIT"
    },
    "UTILITIES": {
        "base": -120.00,
        "drift_pct": 0.005,
        "recurring": True,
        "seasonal": True,           # winter ×1.5
        "day": 5,                   # billed early in month
        "type": "DEBIT"
    },
    "GROCERIES": {
        "base": -450.00,
        "drift_pct": 0.003,
        "frequency": 4,             # 4 transactions per month
        "recurring": False,
        "type": "DEBIT"
    },
    "ENTERTAINMENT": {
        "base": -250.00,
        "drift_pct": 0.001,
        "frequency": 6,
        "recurring": False,
        "type": "DEBIT"
    },
    "TRANSPORT": {
        "base": -150.00,
        "drift_pct": 0.001,
        "frequency": 3,
        "recurring": False,
        "type": "DEBIT"
    },
    # This acts as an internal transfer that 03_aggregate_monthly.py must filter out
    "SAVINGS_CONTRIBUTION": {
        "base": -500.00,
        "drift_pct": 0.0,
        "day": 26,
        "recurring": True,
        "type": "DEBIT"
    }
}

# ----------------------------------------------------------
# HELPER: COUNTERPARTY INFERENCE
# ----------------------------------------------------------

def infer_counterparty(category):
    """Infers CounterpartyType based on CategoryCode."""
    if category in ["SALARY"]:
        return "EMPLOYER"
    if category in ["RENT_MORTGAGE", "SAVINGS_CONTRIBUTION"]:
        return "BANK"
    if category in ["UTILITIES"]:
        return "UTILITY_PROVIDER"
    if category in ["GROCERIES", "ENTERTAINMENT", "TRANSPORT", "TRAVEL"]:
        return "MERCHANT"
    if category in ["ONE_TIME_LARGE"]:
        return "EXTERNAL"
    return "UNKNOWN"


# ----------------------------------------------------------
# DATE LOOP UTILITY
# ----------------------------------------------------------

def month_date_range(year, month):
    """Yield all dates in a month."""
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    delta = end - start
    return [start + timedelta(days=i) for i in range(delta.days)]


# ----------------------------------------------------------
# MAIN GENERATION ROUTINE
# ----------------------------------------------------------

records = []
tx_id = 1

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):

        dates = month_date_range(year, month)
        month_key = f"{year}-{month:02d}"

        # ----------------------------
        # FIXED / RECURRING ITEMS
        # ----------------------------
        for cat, cfg in CATEGORIES.items():

            if not cfg.get("recurring"):
                continue

            # day-of-month check
            day = cfg.get("day")
            recurring_date = None

            for d in dates:
                if d.day == day:
                    recurring_date = d
                    break
            if recurring_date is None:
                continue  # month too short for chosen day

            amount = cfg["base"]

            # drift across years
            drift_months = (year - START_YEAR) * 12 + (month - 1)
            amount *= (1 + cfg["drift_pct"] * drift_months)

            # winter uplift for utilities
            if cfg.get("seasonal") and month in [1, 2, 11, 12]:
                amount *= 1.5
            
            # --- PATCH: Add CounterpartyType ---
            counterparty = infer_counterparty(cat)
            
            records.append({
                "TransactionID": f"T{tx_id:06d}",
                "TransactionDate": recurring_date,
                "PostingDate": recurring_date + timedelta(days=1),
                "MonthKey": month_key,
                "AccountID": ACCOUNT_ID,
                "Amount": round(amount, 2),
                "CurrencyCode": CURRENCY,
                "CategoryCode": cat,
                "DescriptionRaw": f"{cat} {month_key}",
                "CounterpartyType": counterparty, # PATCHED FIELD
                "IsRecurringFlag": True
            })
            tx_id += 1

        # ----------------------------
        # VARIABLE ITEMS
        # ----------------------------
        for cat, cfg in CATEGORIES.items():

            if cfg.get("recurring"):
                continue

            freq = cfg.get("frequency", 0)
            if freq == 0:
                continue

            # one amount split into multiple transactions
            drift_months = (year - START_YEAR) * 12 + (month - 1)
            base = cfg["base"] * (1 + cfg["drift_pct"] * drift_months)
            base_per_tx = base / freq

            # generate random days
            tx_dates = random.sample(dates, freq)

            for d in tx_dates:

                # ±10% noise
                noisy = base_per_tx * (1 + random.uniform(-0.1, 0.1))
                
                # --- PATCH: Add CounterpartyType ---
                counterparty = infer_counterparty(cat)

                records.append({
                    "TransactionID": f"T{tx_id:06d}",
                    "TransactionDate": d,
                    "PostingDate": d + timedelta(days=random.choice([0, 1])),
                    "MonthKey": month_key,
                    "AccountID": ACCOUNT_ID,
                    "Amount": round(noisy, 2),
                    "CurrencyCode": CURRENCY,
                    "CategoryCode": cat,
                    "DescriptionRaw": f"{cat} PURCHASE {random.randint(1000, 9999)}",
                    "CounterpartyType": counterparty, # PATCHED FIELD
                    "IsRecurringFlag": False
                })
                tx_id += 1

# ----------------------------------------------------------
# CONTROLLED OUTLIERS
# ----------------------------------------------------------

# Large positive inflow (one-off event)
outlier_dt = date(2024, 8, 15)
records.append({
    "TransactionID": f"T{tx_id:06d}",
    "TransactionDate": outlier_dt,
    "PostingDate": outlier_dt,
    "MonthKey": "2024-08",
    "AccountID": ACCOUNT_ID,
    "Amount": 5000.00,
    "CurrencyCode": CURRENCY,
    "CategoryCode": "ONE_TIME_LARGE",
    "DescriptionRaw": "TAX REFUND",
    "CounterpartyType": infer_counterparty("ONE_TIME_LARGE"), # PATCHED FIELD
    "IsRecurringFlag": False
})
tx_id += 1

# Annual vacation expense (negative outlier)
for yr in [2024, 2025]:
    d = date(yr, 7, 10)
    cat = "TRAVEL"
    records.append({
        "TransactionID": f"T{tx_id:06d}",
        "TransactionDate": d,
        "PostingDate": d + timedelta(days=1),
        "MonthKey": f"{yr}-07",
        "AccountID": ACCOUNT_ID,
        "Amount": -1800.00,
        "CurrencyCode": CURRENCY,
        "CategoryCode": cat,
        "DescriptionRaw": "VACATION EXPENSE",
        "CounterpartyType": infer_counterparty(cat), # PATCHED FIELD
        "IsRecurringFlag": False
    })
    tx_id += 1

# ----------------------------------------------------------
# SAVE OUTPUT
# ----------------------------------------------------------

df = pd.DataFrame(records)
df = df.sort_values("TransactionDate").reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ 01_generate_utf_GPT.py: Generated {len(df)} UTF records.")
print(f"File saved to: {OUTPUT_PATH}")