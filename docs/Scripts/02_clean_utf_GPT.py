#!/usr/bin/env python3
"""
02_clean_utf.py

Cleans the synthetic UTF dataset and produces a normalized "clean" CSV:

Input : ./data/PoC_UTF_Dataset.csv
Output: ./data/PoC_UTF_Clean.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "PoC_UTF_Dataset.csv"
OUTPUT_FILE = DATA_DIR / "PoC_UTF_Clean.csv"

# Required schema (logical contract for downstream steps)
REQUIRED_COLUMNS = [
    "TransactionID",
    "TransactionDate",
    "PostingDate",
    "AccountID",
    "Amount",
    "CurrencyCode",
    "CategoryCode",
    "DescriptionRaw",
    "CounterpartyType",
    "IsRecurringFlag",
    # MonthKey will be recomputed from TransactionDate
]


def main():
    # ----------------------------------------------------------------
    # 1. Load raw UTF
    # ----------------------------------------------------------------
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading raw UTF from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    original_rows = len(df)
    print(f"[INFO] Loaded {original_rows} raw rows")

    # ----------------------------------------------------------------
    # 2. Check for missing columns
    # ----------------------------------------------------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns in input: {missing}", file=sys.stderr)
        sys.exit(1)

    # Work only with required + any extra columns (if you want to preserve them)
    # Here we restrict to required first, then add MonthKey later.
    df = df[REQUIRED_COLUMNS].copy()

    # ----------------------------------------------------------------
    # 3. Normalize data types
    # ----------------------------------------------------------------
    # 3.1 Dates
    for col in ["TransactionDate", "PostingDate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop rows where TransactionDate is invalid – that’s non-negotiable
    before = len(df)
    df = df.dropna(subset=["TransactionDate"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows with invalid TransactionDate")

    # If PostingDate is NaT, we can backfill it with TransactionDate
    missing_posting = df["PostingDate"].isna().sum()
    if missing_posting > 0:
        print(f"[INFO] Filling {missing_posting} missing PostingDate with TransactionDate")
        df["PostingDate"] = df["PostingDate"].fillna(df["TransactionDate"])

    # 3.2 Amount
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Amount"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows with invalid Amount")

    # 3.3 IsRecurringFlag -> boolean
    # Accept "True"/"TRUE"/"1"/1; everything else -> False
    df["IsRecurringFlag"] = (
        df["IsRecurringFlag"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )

    # ----------------------------------------------------------------
    # 4. Derive MonthKey from TransactionDate
    # ----------------------------------------------------------------
    # Enforce YYYY-MM format from the actual TransactionDate
    df["MonthKey"] = df["TransactionDate"].dt.strftime("%Y-%m")

    # ----------------------------------------------------------------
    # 5. Basic sanity filters (optional, but useful)
    # ----------------------------------------------------------------
    # Drop rows with empty CategoryCode or CurrencyCode – garbage in, garbage out
    before = len(df)
    df = df.dropna(subset=["CategoryCode", "CurrencyCode"])
    df = df[df["CategoryCode"].astype(str).str.strip() != ""]
    df = df[df["CurrencyCode"].astype(str).str.strip() != ""]
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows with empty CategoryCode/CurrencyCode")

    # ----------------------------------------------------------------
    # 6. Sort and write out
    # ----------------------------------------------------------------
    df = df.sort_values(by=["TransactionDate", "TransactionID"]).reset_index(drop=True)

    # Enforce final column order (add MonthKey at the end)
    final_cols = REQUIRED_COLUMNS + ["MonthKey"]
    df = df[final_cols]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"[INFO] Clean UTF written to: {OUTPUT_FILE}")
    print(f"[INFO] Final row count: {len(df)} (from {original_rows} raw rows)")


if __name__ == "__main__":
    main()
