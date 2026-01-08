# Final PoC Documentation Report: SDD Compliance Update

This report serves as the final documentation for the Proof of Concept (PoC) pipeline, confirming its successful execution and compliance with the **Software Design Document (SDD) v0.03 Draft**. All critical requirements and Acceptance Test Plan (ATP) test cases have been validated as **PASS**.

---

## 1. Introduction: Purpose of the PoC (Non-Technical)

The primary goal of this Proof of Concept was to confirm, in a live environment, that our new analytical system can **accurately predict a customer's future bank account balance**---specifically, their **Net Cash Flow**---over the next 12 months.

In simple terms, we need to prove the system can answer the customer's question: *"Based on my history, will I have enough money next month, and what will my balance look like for the rest of the year?"*

The core challenge for any financial forecasting system is dealing with **surprises**---large, one-off events like a substantial tax refund, an emergency expense, or a major vacation payment. If these anomalies are included in the training data, they can ruin the forecast.

This PoC validated two crucial capabilities:

1. **Robust Data Cleanup:** The system can automatically **identify and neutralize** these unusual transactions so they don't skew the prediction (SDD Chapter 9).

2. **Highly Accurate Forecasting:** The system can choose the best mathematical model (out of the two tested) to generate a **trustworthy, highly accurate 12-month projection** (SDD Chapter 11).

The successful validation below confirms the technical foundations for delivering powerful, proactive, and explainable financial insights.

---

## 2. Alignment with SDD Architectural Layers

The 5-step PoC pipeline successfully implemented the logic outlined in the respective SDD chapters, ensuring the data integrity and separation of concerns required for a modular architecture.

| SDD Chapter | Layer | PoC Script | Validation Result |
|-------------|-------|------------|-------------------|
| **Ch. 7** | Data Cleaning & Validation | 02_clean_utf_GPT.py | Correctly enforced schema and standardized formats (TC-02). |
| **Ch. 8** | Monthly Aggregation | 03_aggregate_monthly_GPT.py | Successfully calculated **NetFlowExternal** (excluding transfers) and rolling features (TC-03). |
| **Ch. 9** | Outlier Detection & Treatment | 04_outliers_GPT.py | Implemented **Modified Z-Score** detection and **Median Imputation**. |
| **Ch. 11** | Forecasting Engine | 05_forecast_GPT.py | Executed ARIMA vs. SARIMA comparison and WMAPE calculation. |
| **Ch. 4** | LLM Interaction Layer | Final JSON Output | Generated the required structured payload for LLM narrative assembly (TC-06). |

---

## 3. Detailed Test Case Results

The following section closes out the three most critical Acceptance Test Plan (ATP) test cases based on the final pipeline execution.

### TC-04 --- Outlier Detection & Treatment Validation

**Requirement (SDD Chapter 9):** Use the **Modified Z-Score** (MZ-Score) method with a threshold of abs(MZ) > 3.5 to identify and treat abnormal monthly flows using **median imputation**.

| MonthKey | NetFlowExternal (Original) | MZ_Score | IsOutlier | NetFlow_Clean (Imputed) | TreatmentTag | Status |
|----------|----------------------------|----------|-----------|-------------------------|--------------|--------|
| **2024-07** | **-1,414.27** EUR | -23.13 | **True** | 346.01 EUR | ABNORMAL_EXTERNAL_FLOW | **PASS** |
| **2024-08** | **+5,364.32** EUR | 75.31 | **True** | 346.01 EUR | ABNORMAL_EXTERNAL_FLOW | **PASS** |
| **2025-07** | **-1,435.58** EUR | -23.44 | **True** | 346.01 EUR | ABNORMAL_EXTERNAL_FLOW | **PASS** |

**Conclusion:** The pipeline successfully implemented the Dual Value Model (SDD Chapter 9) by logging the actual flow while replacing the value with the robust median in the **NetFlow_Clean** series used for forecasting.

### TC-05 --- Forecasting & Winner Selection

**Requirement (SDD Chapter 11):** Compare ARIMA and SARIMA models on the **NetFlow_Clean** series and select the model with the lower **WMAPE**. The final WMAPE must be **< 20.0%**.

| Model | WMAPE | Result | SDD Status |
|-------|-------|--------|------------|
| **SARIMA** (1, 1, 1)x(1, 1, 0, 12) | **6.171%** | **WINNER** | ✅ **PASS** |
| **ARIMA** (1, 1, 1) | 6.209% | LOSER | ✅ **PASS** |

**Conclusion:** The **SARIMA** model was correctly chosen as the winner. The winning WMAPE of **6.171%** is highly accurate and significantly better than the SDD threshold of 20.0%, fully satisfying the core accuracy requirement of **SDD H.10 Acceptance Decision**.

### TC-06 --- Explainability Payload Validation

**Requirement (SDD Appendix A & B):** The final output must be a structured JSON payload (PoC_Forecast_Summary.json) that adheres to the defined schema and provides all necessary metadata for the LLM Rendering Engine.

**Conclusion:** The PoC_Forecast_Summary.json file was successfully generated and contains the necessary metrics, outlier list, and forecast data, confirming compliance with the requirements for the **LLM Interaction Layer** (SDD Chapter 4).

---

## 4. Final Forecast Snapshot

The winning **SARIMA** model provides a robust 12-month forecast. The model parameters show strong evidence of monthly seasonality, captured by the ar.S.L12 value of **0.866**.

### 12-Month Forecast Mean (2026)

| MonthKey | ForecastMean (EUR) | ForecastLowerCI (95%) | ForecastUpperCI (95%) |
|----------|--------------------|-----------------------|-----------------------|
| **2026-01** | **300.07** | 266.71 | 333.44 |
| **2026-02** | **290.40** | 257.04 | 323.77 |
| **2026-03** | **469.80** | 436.43 | 503.17 |
| **2026-04** | **469.04** | 435.67 | 502.41 |
| **2026-05** | **448.70** | 415.33 | 482.07 |
| **2026-06** | **414.11** | 380.74 | 447.48 |
| **2026-07** | **351.15** | 317.78 | 384.51 |
| **2026-08** | **468.78** | 435.41 | 502.15 |

---

## 5. Production-Grade PoC Execution Guide

This guide details the steps required to execute the PoC pipeline and replicate the validation results, ensuring a clean, deterministic, and production-ready environment setup.

### 5.1 Prerequisites

1. **Environment:** Python 3.9+ installed.

2. **Dependencies:** All required packages (e.g., pandas, numpy, statsmodels, dateutil) are listed and installed (ideally within a virtual environment).

3. **Directory Structure:** A working directory (poc_di_insights) with a Scripts sub-directory and a data sub-directory.

```
/poc_di_insights
├── /Scripts
│   ├── 01_generate_utf_GPT.py
│   ├── 02_clean_utf_GPT.py
│   ├── 03_aggregate_monthly_GPT.py
│   ├── 04_outliers_GPT.py
│   └── 05_forecast_GPT.py
└── /data
    └── (Empty, files will be generated here)
```

### 5.2 Execution Steps

All commands must be executed sequentially from within the **Scripts** directory.

| Step | Command | Input File | Output File | SDD Layer Validation |
|------|---------|------------|-------------|----------------------|
| **1. Generate UTF** | `python 01_generate_utf_GPT.py` | (None) | data/PoC_UTF_Dataset.csv | Initial data creation (TC-01). |
| **2. Clean UTF** | `python 02_clean_utf_GPT.py` | data/PoC_UTF_Dataset.csv | data/PoC_UTF_Clean.csv | Data Cleaning & Validation (TC-02). |
| **3. Aggregate** | `python 03_aggregate_monthly_GPT.py` | data/PoC_UTF_Clean.csv | data/PoC_Monthly_Features.csv | Monthly Aggregation (TC-03). |
| **4. Outlier Detection** | `python 04_outliers_GPT.py` | data/PoC_Monthly_Features.csv | data/PoC_Outliers_Treated.csv | Outlier Detection & Treatment (TC-04). |
| **5. Forecast** | `python 05_forecast_GPT.py` | data/PoC_Outliers_Treated.csv | data/PoC_Forecast_Results.csv, data/PoC_Forecast_Summary.json | Forecasting Engine, Model Selection, LLM Payload (TC-05, TC-06). |

### 5.3 Post-Execution Validation

After Step 5 is complete, verify the following files and outputs:

| File | Content to Verify | SDD Reference |
|------|-------------------|---------------|
| PoC_Outliers_Treated.csv | Contains the NetFlowExternal (raw) column and the NetFlow_Clean (imputed) column. Check that the months **2024-07, 2024-08, and 2025-07** show different values for these two columns. | Chapter 9 (Dual Value Model) |
| PoC_Forecast_Results.csv | Final output containing history and the 12-month forecast (ForecastMean, ForecastLowerCI, ForecastUpperCI). | Chapter 11 |
| PoC_Forecast_Summary.json | Must be a valid JSON structure containing: winner_model: **"SARIMA"** and wmape_sarima: **6.171** (or very close) | Appendix A (JSON Schema), H.10 (Acceptance) |