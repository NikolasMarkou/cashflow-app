# Cash Flow Forecasting Testing Framework v1.0

**Document Version:** 1.0
**Date:** 2026-01-13
**Status:** Authoritative Specification

---

## 1. Executive Summary

This document defines the standardized testing framework for evaluating the Cash Flow Forecasting Engine across different customer segments and data quality conditions. The framework provides reproducible, statistically rigorous evaluation with clear reporting.

**Key Parameters:**
- **Forecast Horizon:** 12 months forward prediction
- **Historical Data:** 24 months synthetic transactions
- **Random Seeds:** 10 runs per configuration
- **Acceptance Threshold:** WMAPE < 20%

---

## 2. Account Type Definitions

### 2.1 Personal Account

Represents individual consumer banking with predictable income and variable discretionary spending.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Monthly Income | EUR 3,000 | Single salary source |
| Income Sources | 1 | Primary employer only |
| Recurring Expenses | 5-7 | Rent, utilities, subscriptions |
| Transaction Volume | 40-60/month | Mix of recurring and discretionary |
| Typical Categories | SALARY, RENT, UTILITIES, GROCERIES, TRANSPORT, ENTERTAINMENT |

**Transaction Composition:**
- 60% Recurring (salary, rent, utilities, subscriptions)
- 40% Variable (groceries, dining, entertainment, shopping)

### 2.2 SME Account (Small-Medium Enterprise)

Represents business accounts with multiple revenue streams and operational expenses.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Monthly Revenue | EUR 25,000 | Multiple customer payments |
| Revenue Sources | 5-10 | B2B invoices, retail sales |
| Recurring Expenses | 10-15 | Payroll, rent, utilities, services |
| Transaction Volume | 150-250/month | High operational activity |
| Typical Categories | INVOICE_PAYMENT, PAYROLL, RENT, UTILITIES, SUPPLIES, TAX |

**Transaction Composition:**
- 45% Recurring (payroll, rent, loan repayments, subscriptions)
- 35% Semi-predictable (supplier payments, inventory)
- 20% Variable (ad-hoc expenses, one-time purchases)

### 2.3 Corporate Account

Represents large enterprise accounts with diverse, high-volume transaction patterns.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Monthly Revenue | EUR 500,000 | Diversified revenue streams |
| Revenue Sources | 15-25 | Multiple business units |
| Recurring Expenses | 20-30 | Complex operational structure |
| Transaction Volume | 500-800/month | High frequency, large values |
| Typical Categories | INVOICE_PAYMENT, PAYROLL, INTERCOMPANY, TAX, CAPITAL_EXPENSE |

**Transaction Composition:**
- 50% Recurring (payroll, intercompany transfers, debt service)
- 30% Contractual (known but variable timing/amounts)
- 20% Variable (project expenses, opportunistic purchases)

---

## 3. Randomness Levels

The randomness parameter controls transaction unpredictability through multiple mechanisms.

### 3.1 Level Definitions

| Level | Code | Income CV | Expense CV | Flag Corruption | Timing Jitter | Description |
|-------|------|-----------|------------|-----------------|---------------|-------------|
| None | `none` | 0% | 0% | 0% | 0 days | Perfectly predictable baseline |
| Low | `low` | 2% | 5% | 5% | ±1 day | Minor natural variation |
| Medium | `medium` | 5% | 15% | 15% | ±3 days | Realistic business conditions |
| High | `high` | 10% | 30% | 30% | ±5 days | Stressed/volatile conditions |

**CV = Coefficient of Variation (std / mean)**

### 3.2 Randomness Mechanisms

**3.2.1 Amount Variation**
- Income: Gaussian noise applied to base amounts
- Expenses: Category-specific variation (groceries more variable than rent)

**3.2.2 Flag Corruption**
- Simulates upstream data quality issues
- Randomly flips `is_recurring_flag` on transactions
- Tests robustness of recurrence detection (Layer 0.5)

**3.2.3 Timing Jitter**
- Shifts transaction dates within tolerance window
- Affects transfer detection matching
- Simulates payment processing delays

**3.2.4 Category-Specific Variation**
| Category Type | None | Low | Medium | High |
|---------------|------|-----|--------|------|
| Fixed (Rent, Salary) | 0% | 1% | 3% | 5% |
| Semi-fixed (Utilities) | 0% | 5% | 15% | 25% |
| Variable (Groceries) | 0% | 10% | 25% | 40% |
| Discretionary (Entertainment) | 0% | 15% | 35% | 50% |

---

## 4. Synthetic Data Generation

### 4.1 Data Structure

Each synthetic dataset contains 24 months of historical transactions following the UTF (Unified Transaction Format) schema:

```
Required Fields:
- transaction_id: Unique identifier (UUID)
- transaction_date: Date of transaction
- amount: Transaction value (positive=inflow, negative=outflow)
- category: Transaction category code
- counterparty_name: Transaction counterparty
- is_recurring_flag: Boolean recurring indicator
- description: Transaction description
```

### 4.2 Generation Rules

**4.2.1 Income Transactions**
- Personal: Monthly salary on day 25-28
- SME: Weekly/biweekly customer payments
- Corporate: Daily batch settlements

**4.2.2 Recurring Expenses**
- Rent: Day 1-5 of month
- Utilities: Day 10-15 of month
- Subscriptions: Various fixed days
- Payroll (SME/Corporate): Day 25-28

**4.2.3 Variable Expenses**
- Poisson-distributed transaction counts
- Log-normal amount distributions
- Category-weighted probabilities

### 4.3 Transfer Pairs

Internal transfers are generated to test transfer netting:
- Personal: 2-4 savings transfers/month
- SME: 5-10 interaccount transfers/month
- Corporate: 20-30 intercompany transfers/month

---

## 5. WMAPE Measurement Protocol

### 5.1 Definition

Weighted Mean Absolute Percentage Error (WMAPE) as per SDD Section 13.4:

```
WMAPE = 100 * SUM(|actual - forecast|) / SUM(|actual|)
```

### 5.2 Measurement Points

WMAPE is calculated at multiple horizons:

| Horizon | Months | Use Case |
|---------|--------|----------|
| Short-term | 1-3 | Operational planning |
| Medium-term | 1-6 | Quarterly forecasting |
| Long-term | 1-12 | Annual budgeting |

### 5.3 Per-Period Calculation

For granular analysis, WMAPE is also computed per individual month:

```
WMAPE_month_h = 100 * |actual_h - forecast_h| / |actual_h|
```

This enables identification of forecast degradation patterns over the horizon.

---

## 6. Test Execution Protocol

### 6.1 Configuration Matrix

Total configurations: **3 account types x 4 randomness levels = 12 configurations**

| Config ID | Account Type | Randomness | Seeds |
|-----------|--------------|------------|-------|
| P-NONE | Personal | None | 10 |
| P-LOW | Personal | Low | 10 |
| P-MED | Personal | Medium | 10 |
| P-HIGH | Personal | High | 10 |
| S-NONE | SME | None | 10 |
| S-LOW | SME | Low | 10 |
| S-MED | SME | Medium | 10 |
| S-HIGH | SME | High | 10 |
| C-NONE | Corporate | None | 10 |
| C-LOW | Corporate | Low | 10 |
| C-MED | Corporate | Medium | 10 |
| C-HIGH | Corporate | High | 10 |

**Total test runs: 120** (12 configurations x 10 seeds)

### 6.2 Execution Steps

For each configuration:

1. **Generate** 24 months synthetic data with specified parameters
2. **Split** data: months 1-24 for training, generate holdout months 25-36
3. **Run** ForecastEngine on months 1-24
4. **Predict** 12-month forecast (months 25-36)
5. **Compare** forecast vs synthetic holdout actuals
6. **Calculate** WMAPE at horizons 3, 6, 12 and per-period
7. **Record** all metrics with seed identifier

### 6.3 Holdout Generation

The holdout period (months 25-36) uses the **same randomness level** as the training period. This ensures:
- Consistent pattern continuation for realistic WMAPE measurement
- Fair comparison across randomness levels (high randomness training vs high randomness actuals)
- The holdout represents "what would have happened" under the same conditions

**Important:** The same seed controls both training and holdout generation, ensuring reproducibility while maintaining realistic variation.

---

## 7. Output Specification

### 7.1 CSV Output: `results/wmape_results.csv`

| Column | Type | Description |
|--------|------|-------------|
| account_type | str | personal, sme, corporate |
| randomness | str | none, low, medium, high |
| seed | int | Random seed (1-10) |
| wmape_3m | float | WMAPE for months 1-3 |
| wmape_6m | float | WMAPE for months 1-6 |
| wmape_12m | float | WMAPE for months 1-12 |
| wmape_m1 | float | WMAPE for month 1 only |
| wmape_m2 | float | WMAPE for month 2 only |
| ... | ... | ... |
| wmape_m12 | float | WMAPE for month 12 only |
| model_selected | str | Winning model name |
| passes_threshold | bool | WMAPE_12m < 20% |
| ci_width_avg | float | Average CI width |
| runtime_seconds | float | Execution time |

### 7.2 CSV Output: `results/wmape_summary.csv`

Aggregated statistics per configuration:

| Column | Type | Description |
|--------|------|-------------|
| account_type | str | Account type |
| randomness | str | Randomness level |
| wmape_3m_mean | float | Mean WMAPE 3-month |
| wmape_3m_std | float | Std dev WMAPE 3-month |
| wmape_6m_mean | float | Mean WMAPE 6-month |
| wmape_6m_std | float | Std dev WMAPE 6-month |
| wmape_12m_mean | float | Mean WMAPE 12-month |
| wmape_12m_std | float | Std dev WMAPE 12-month |
| pass_rate | float | % runs with WMAPE < 20% |
| model_ets_pct | float | % runs selecting ETS |
| model_sarima_pct | float | % runs selecting SARIMA |

### 7.3 Plot Outputs

All plots saved to `plots/` directory:

| Filename | Description |
|----------|-------------|
| `wmape_by_account_type.png` | Bar chart: WMAPE by account type (grouped by randomness) |
| `wmape_by_randomness.png` | Bar chart: WMAPE by randomness level (grouped by account type) |
| `wmape_heatmap.png` | Heatmap: Account type vs Randomness (12-month WMAPE) |
| `wmape_horizon_degradation.png` | Line chart: WMAPE by forecast month (1-12) per config |
| `pass_rate_matrix.png` | Heatmap: Pass rate (%) by account type and randomness |
| `forecast_trajectories.png` | Grid: Sample forecasts for each configuration |

---

## 8. Acceptance Criteria

### 8.1 Model Acceptance

Individual forecast run passes if: **WMAPE_12m < 20%**

### 8.2 Configuration Acceptance

Configuration is acceptable if: **Pass rate >= 80%** (8+ of 10 seeds pass)

### 8.3 Expected Performance Targets

| Account Type | None | Low | Medium | High |
|--------------|------|-----|--------|------|
| Personal | >95% | >90% | >70% | >50% |
| SME | >90% | >85% | >65% | >45% |
| Corporate | >95% | >90% | >75% | >55% |

*Values represent expected pass rates based on account predictability characteristics.*

---

## 9. Implementation Notes

### 9.1 Script Structure

Single consolidated script: `scripts/run_framework_tests.py`

```
scripts/
  run_framework_tests.py    # Main test execution
  framework_config.py       # Configuration dataclasses
  data_generator.py         # Synthetic data generation

results/
  wmape_results.csv         # Detailed results
  wmape_summary.csv         # Aggregated summary

plots/
  wmape_by_account_type.png
  wmape_by_randomness.png
  wmape_heatmap.png
  wmape_horizon_degradation.png
  pass_rate_matrix.png
  forecast_trajectories.png
```

### 9.2 Reproducibility

- All random operations seeded with explicit seed values
- Seeds numbered 1-10 for each configuration
- Full results traceable to specific seed

### 9.3 Execution Command

```bash
# Run full test suite (120 runs)
python scripts/run_framework_tests.py

# Run specific account type
python scripts/run_framework_tests.py --account-type personal

# Run specific randomness level
python scripts/run_framework_tests.py --randomness low

# Quick validation (3 seeds instead of 10)
python scripts/run_framework_tests.py --quick
```

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-13 | Engineering | Initial framework specification |

---

*End of Document*
