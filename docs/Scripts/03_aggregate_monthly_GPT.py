#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys

INPUT_FILE = "./data/PoC_UTF_Clean.csv"
OUTPUT_FILE = "./data/PoC_Monthly_Features.csv"

EXPECTED_MONTH_COUNT = 24   # Adjust if your synthetic data span changes


def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def main():

    print("[INFO] Loading cleaned UTF dataset...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        fail(f"Unable to load {INPUT_FILE}: {e}")

    # Ensure required fields
    required_cols = [
        "TransactionID", "TransactionDate", "MonthKey",
        "Amount", "CategoryCode"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        fail(f"Missing required columns in cleaned UTF: {missing}")

    # Enforce types
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    if df["TransactionDate"].isna().any() or df["Amount"].isna().any():
        fail("Invalid values after dtype enforcement (dates or amounts).")

    # Recalculate MonthKey from TransactionDate (defensive)
    df["MonthKey"] = df["TransactionDate"].dt.strftime("%Y-%m")

    # Debit / Credit split
    df["Debit"] = df["Amount"].apply(lambda x: x if x < 0 else 0)
    df["Credit"] = df["Amount"].apply(lambda x: x if x > 0 else 0)

    # Exclude internal transfers from NetFlowExternal
    df_external = df[df["CategoryCode"] != "TRANSFER_OUT"].copy()

    # Monthly aggregation
    print("[INFO] Aggregating monthly summaries...")
    monthly = df_external.groupby("MonthKey").agg(
        MonthlyDebitTotal=("Debit", "sum"),
        MonthlyCreditTotal=("Credit", "sum"),
        TransactionCount=("TransactionID", "count")
    ).reset_index()

    # Compute NetFlowExternal
    monthly["NetFlowExternal"] = (
        monthly["MonthlyCreditTotal"] + monthly["MonthlyDebitTotal"]
    )

    # Sort by MonthKey for rolling windows
    monthly = monthly.sort_values("MonthKey").reset_index(drop=True)

    # Rolling features
    monthly["NetFlow_3M_Avg"] = (
        monthly["NetFlowExternal"].rolling(window=3).mean()
    )

    # Validation: expected month count
    month_count = monthly["MonthKey"].nunique()
    print(f"[INFO] Months detected: {month_count}")

    if month_count != EXPECTED_MONTH_COUNT:
        print("[WARNING] Month count does NOT match expectation.")
        print(f"Expected: {EXPECTED_MONTH_COUNT}, Found: {month_count}")
        print("This may indicate missing or partial months in UTF data.")
        # Do not fail automatically — PoC may intentionally vary.

    # Save output
    monthly.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Monthly features saved to: {OUTPUT_FILE}")

    # Preview
    print("\n--- Monthly Snapshot (Head) ---")
    print(monthly.head(5))

    print("\n--- Monthly Snapshot (Tail) ---")
    print(monthly.tail(5))


if __name__ == "__main__":
    main()
