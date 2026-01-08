import pandas as pd
import numpy as np
import os

INPUT_FILE = "./data/PoC_Monthly_Features.csv"
OUTPUT_FILE = "./data/PoC_Outliers_Treated.csv"

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # --------------------------------------------------------
    # 1. LOAD AND SORT
    # --------------------------------------------------------
    df = pd.read_csv(INPUT_FILE)

    # Ensure correct ordering (YYYY-MM lexicographically works)
    df = df.sort_values("MonthKey").reset_index(drop=True)

    # --------------------------------------------------------
    # 2. MODIFIED Z-SCORE CALCULATION (Robust Outlier Detection)
    # --------------------------------------------------------
    netflow = df["NetFlowExternal"]

    median = netflow.median()
    mad = (netflow - median).abs().median()

    # Prevent division-by-zero – if MAD = 0, no outliers can be identified
    if mad == 0:
        df["MZ_Score"] = 0
        df["IsOutlier"] = False
    else:
        df["MZ_Score"] = 0.6745 * (netflow - median) / mad
        df["IsOutlier"] = df["MZ_Score"].abs() > 3.5

    # --------------------------------------------------------
    # 3. OUTLIER TREATMENT (Median Imputation)
    # --------------------------------------------------------
    df["NetFlow_Clean"] = np.where(df["IsOutlier"], median, df["NetFlowExternal"])

    # --------------------------------------------------------
    # 4. TREATMENT TAGS FOR LLM EXPLAINABILITY
    # --------------------------------------------------------
    df["TreatmentTag"] = np.where(
        df["IsOutlier"], 
        "ABNORMAL_EXTERNAL_FLOW",
        "NORMAL"
    )

    # --------------------------------------------------------
    # 5. SAVE RESULT
    # --------------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------------
    # 6. LOG SUMMARY
    # --------------------------------------------------------
    print("Outlier Detection Completed")
    print("---------------------------")
    print(f"Total months: {len(df)}")
    print(f"Outliers detected: {df['IsOutlier'].sum()}")
    if df['IsOutlier'].sum() > 0:
        print(df[df["IsOutlier"]][["MonthKey", "NetFlowExternal", "MZ_Score"]])


if __name__ == "__main__":
    main()
