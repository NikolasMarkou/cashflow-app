# Cash Flow Forecasting Engine - Handover Document

## 1. Executive Summary

**Project:** Cash Flow Forecasting Predictive Engine
**Version:** v0.5.0
**Specification:** SDD v0.05

### Key Results

| Metric | Value |
|--------|-------|
| Overall Pass Rate | 89.2% (107/120 test runs) |
| Average WMAPE | 10.2% |
| Acceptance Threshold | WMAPE < 20% |
| Models Available | ETS, SARIMA, SARIMAX, TiRex |

The forecasting engine achieves the required WMAPE < 20% threshold across most test configurations, with particularly strong performance on SME and Corporate account types.

---

## 2. Quick Start Guide

### Installation

```bash
# Clone and install with all dependencies
git clone <repository-url>
cd cashflow-app
pip install -e ".[dev,viz,web]"
```

### Running the Web Interface

```bash
# Start the web server
cashflow-web

# Open in browser
# http://localhost:8000
```

The web interface allows CSV file upload, parameter configuration, and interactive visualization of forecasts.

### Running CLI Forecasts

```bash
# Basic forecast
cashflow forecast --utf data/utf.csv --output ./output

# With CRF data
cashflow forecast --utf data/utf.csv --crf data/crf.csv --output ./output

# Validate data format
cashflow validate --utf data/transactions.csv
```

### Running the Test Framework

```bash
# Full test suite (120 runs, ~15 minutes)
python scripts/run_framework_tests.py

# Quick validation (36 runs, ~5 minutes)
python scripts/run_framework_tests.py --quick

# Results exported to results/ directory
# Plots generated in plots/ directory
```

---

## 3. Test Results Summary

Results from `results/wmape_summary.csv` (10 seeds per configuration):

| Account Type | Randomness | WMAPE 3M | WMAPE 6M | WMAPE 12M | Pass Rate | Model |
|--------------|------------|----------|----------|-----------|-----------|-------|
| Personal | None | 10.64% | 7.12% | 6.92% | 90% | TiRex |
| Personal | Low | 11.32% | 9.23% | 9.32% | 100% | TiRex |
| Personal | Medium | 12.01% | 11.86% | 14.66% | 80% | TiRex |
| Personal | High | 21.85% | 18.58% | 18.48% | 40% | TiRex |
| SME | None | 2.63% | 2.77% | 4.02% | 100% | TiRex |
| SME | Low | 2.70% | 2.75% | 3.98% | 100% | TiRex |
| SME | Medium | 4.61% | 4.67% | 6.60% | 100% | TiRex |
| SME | High | 5.54% | 5.91% | 6.98% | 70% | TiRex |
| Corporate | None | 5.53% | 6.71% | 10.72% | 100% | TiRex |
| Corporate | Low | 5.05% | 6.29% | 10.85% | 100% | TiRex |
| Corporate | Medium | 6.48% | 8.21% | 13.53% | 100% | TiRex |
| Corporate | High | 7.67% | 9.91% | 16.48% | 90% | TiRex |

### Performance Highlights

- **Best Performance:** SME accounts with None/Low randomness (2.6-4.0% WMAPE)
- **Most Consistent:** Corporate accounts maintain <17% WMAPE even at High randomness
- **Most Challenging:** Personal accounts at High randomness (40% pass rate)
- **Model Selection:** TiRex selected 100% of the time, outperforming ETS and SARIMA

---

## 4. Data Generation Framework

The test framework generates synthetic transaction data to evaluate forecasting performance under controlled conditions. This section provides detailed documentation of the generation methodology.

### 4.1 Account Types

Three account types represent different customer segments with distinct transaction patterns:

#### Personal Account

| Parameter | Value |
|-----------|-------|
| Monthly Income | EUR 3,000 |
| Income Sources | 1 (salary) |
| Recurring Expenses | 5-7 per month |
| Transaction Volume | 40-60 per month |
| Recurring % | 60% |
| Variable % | 40% |

**Categories:** SALARY, RENT, UTILITIES, GROCERIES, TRANSPORT, ENTERTAINMENT, SUBSCRIPTIONS, DINING, SHOPPING, HEALTHCARE

**Characteristics:**
- Single fixed salary on day 27
- Fixed recurring expenses (rent EUR 1,200, utilities, subscriptions)
- Variable discretionary spending (groceries, dining, entertainment)
- 2-4 savings transfers per month

#### SME Account (Small-Medium Enterprise)

| Parameter | Value |
|-----------|-------|
| Monthly Revenue | EUR 25,000 |
| Revenue Sources | 5-10 customers |
| Recurring Expenses | 10-15 per month |
| Transaction Volume | 150-250 per month |
| Recurring % | 45% |
| Semi-predictable % | 35% |
| Variable % | 20% |

**Categories:** INVOICE_PAYMENT, PAYROLL, RENT, UTILITIES, SUPPLIES, TAX, INSURANCE, PROFESSIONAL_SERVICES, MARKETING, EQUIPMENT, TRAVEL, LOAN_REPAYMENT

**Characteristics:**
- 7 customer payments distributed throughout month
- Fixed operational expenses (payroll EUR 12,000, rent EUR 3,500)
- Quarterly tax obligations
- 5-10 interaccount transfers per month

#### Corporate Account

| Parameter | Value |
|-----------|-------|
| Monthly Revenue | EUR 500,000 |
| Revenue Sources | 15-25 business units |
| Recurring Expenses | 20-30 per month |
| Transaction Volume | 500-800 per month |
| Recurring % | 50% |
| Contractual % | 30% |
| Variable % | 20% |

**Categories:** INVOICE_PAYMENT, PAYROLL, INTERCOMPANY, TAX, CAPITAL_EXPENSE, RENT, UTILITIES, INSURANCE, PROFESSIONAL_SERVICES, IT_SERVICES, MARKETING, TRAVEL, DEBT_SERVICE, DIVIDENDS, REGULATORY_FEES

**Characteristics:**
- 20 daily settlement batches (EUR 25,000 each)
- Large fixed obligations (payroll EUR 200,000, debt service EUR 75,000)
- Quarterly and annual patterns (bonuses, audits)
- 20-30 intercompany transfers per month

### 4.2 Randomness Levels

Four randomness levels control data volatility and quality:

#### None (Baseline)

| Parameter | Value |
|-----------|-------|
| Residual Magnitude Multiplier | 1.0 |
| Residual Volatility | 0.0% |
| Transaction Multiplier | 1.0 |
| Predictable Expense % | 90% |
| Flag Corruption Rate | 0% |

**Purpose:** Clean baseline with deterministic patterns. Validates the engine works correctly under ideal conditions.

#### Low

| Parameter | Value |
|-----------|-------|
| Residual Magnitude Multiplier | 0.9 |
| Residual Volatility | 13.5% |
| Transaction Multiplier | 0.9 |
| Predictable Expense % | 80% |
| Flag Corruption Rate | 2% |

**Purpose:** Minor natural variation. Simulates typical business conditions with small amount fluctuations.

#### Medium

| Parameter | Value |
|-----------|-------|
| Residual Magnitude Multiplier | 1.17 |
| Residual Volatility | 27% |
| Transaction Multiplier | 1.08 |
| Predictable Expense % | 70% |
| Flag Corruption Rate | 5% |

**Purpose:** Realistic business conditions. Includes budget variations, seasonal patterns, and moderate data quality issues.

#### High

| Parameter | Value |
|-----------|-------|
| Residual Magnitude Multiplier | 1.44 |
| Residual Volatility | 45% |
| Transaction Multiplier | 1.35 |
| Predictable Expense % | 60% |
| Flag Corruption Rate | 10% |

**Purpose:** Stressed/volatile conditions. Tests model robustness under significant unpredictability and data quality degradation.

### 4.3 Parameter Definitions

**residual_magnitude_multiplier**
- Controls the total budget allocated to variable (non-recurring) expenses
- Value 1.0 = baseline budget, 1.44 = 44% more variable expenses
- Higher values increase the unpredictable portion of cash flow

**residual_volatility**
- Standard deviation applied to individual transaction amounts
- Value 0.0 = exact amounts, 0.45 = high per-transaction variation
- Applied using Gaussian noise to base amounts

**residual_transaction_multiplier**
- Controls the number of variable transactions generated
- Value 0.9 = 10% fewer transactions, 1.35 = 35% more transactions
- Affects granularity of variable expense patterns

**predictable_expense_pct**
- Percentage of recurring expenses with fixed (exact) amounts
- Value 0.90 = 90% exact, 10% with variation
- Lower values make decomposition more challenging

**flag_corruption_rate**
- Probability of randomly flipping is_recurring_flag values
- Value 0.0 = perfect flags, 0.10 = 10% corrupted
- Tests robustness of recurrence detection (Layer 0.5)

### 4.4 Category Classification

Transactions are classified by variability:

| Type | Categories | Variation |
|------|-----------|-----------|
| Fixed | SALARY, RENT, PAYROLL, LOAN_REPAYMENT, DEBT_SERVICE, INSURANCE | Minimal |
| Semi-fixed | UTILITIES, SUBSCRIPTIONS, TAX, REGULATORY_FEES, IT_SERVICES | Low |
| Variable | GROCERIES, SUPPLIES, TRANSPORT, HEALTHCARE, INVOICE_PAYMENT | Moderate |
| Discretionary | ENTERTAINMENT, DINING, SHOPPING, TRAVEL, MARKETING, EQUIPMENT | High |

### 4.5 Data Generation Process

1. **Generate 40 months continuously** (28 training + 12 holdout)
2. **Income transactions:** Fixed amounts, distributed dates by account type
3. **Recurring expenses:** Fixed amounts on specific days
4. **Variable expenses:** Log-normal distribution within budget, random days
5. **Periodic expenses:** Quarterly/annual patterns (tax, bonuses, audits)
6. **Internal transfers:** Paired IN/OUT with matching transfer_link_id
7. **Apply flag corruption:** Random flip of is_recurring_flag at configured rate
8. **Split by date:** Training period (months 1-28), Holdout period (months 29-40)

### 4.6 Seasonal Multipliers

Subtle seasonal patterns applied to variable expenses:

| Month | Personal | SME | Corporate |
|-------|----------|-----|-----------|
| January | 0.96 | 0.98 | 0.98 |
| February | 0.96 | 1.00 | 1.00 |
| March | 1.00 | 1.04 | 1.04 |
| June | 1.04 | 1.04 | 1.02 |
| July | 1.08 | 0.96 | 0.96 |
| December | 1.12 | 1.06 | 1.06 |

Income remains deterministic (no seasonal multipliers) to ensure stable decomposition testing.

---

## 5. Scripts and Commands

### 5.1 Test Framework

Location: `scripts/run_framework_tests.py`

```bash
# Full test suite (3 account types x 4 randomness x 10 seeds = 120 runs)
python scripts/run_framework_tests.py

# Quick validation (3 seeds per config = 36 runs)
python scripts/run_framework_tests.py --quick

# Filter by account type
python scripts/run_framework_tests.py --account-type personal
python scripts/run_framework_tests.py --account-type sme
python scripts/run_framework_tests.py --account-type corporate

# Filter by randomness level
python scripts/run_framework_tests.py --randomness none
python scripts/run_framework_tests.py --randomness low
python scripts/run_framework_tests.py --randomness medium
python scripts/run_framework_tests.py --randomness high

# Custom seed count
python scripts/run_framework_tests.py --seeds 5

# Combined filters
python scripts/run_framework_tests.py --account-type sme --randomness medium --seeds 5
```

**Expected Runtimes:**
- Full suite (120 runs): 12-15 minutes
- Quick suite (36 runs): 4-5 minutes
- Single configuration: 6-8 seconds

### 5.2 Generated Output Files

**Results Directory:** `results/`

| File | Description |
|------|-------------|
| `wmape_results.csv` | Detailed per-run results (120 rows for full suite) |
| `wmape_summary.csv` | Aggregate statistics per configuration (12 rows) |

**Plots Directory:** `plots/`

| File | Description |
|------|-------------|
| `wmape_heatmap.png` | WMAPE matrix (account type vs randomness) |
| `wmape_by_account_type.png` | Grouped bar chart by account type |
| `wmape_by_randomness.png` | Grouped bar chart by randomness level |
| `wmape_horizon_degradation.png` | WMAPE increase over forecast horizon (3 subplots) |
| `pass_rate_matrix.png` | Pass rate heatmap by configuration |
| `forecast_trajectories.png` | Historical + forecast time series (3x4 grid) |
| `decomposition_analysis.png` | Predictable vs residual components (3x4 grid) |

### 5.3 Configuration Files

| File | Purpose |
|------|---------|
| `scripts/framework_config.py` | Account types, randomness levels, test parameters |
| `scripts/data_generator.py` | Synthetic transaction generation logic |

---

## 6. Methodology

### 6.1 WMAPE Calculation

Weighted Mean Absolute Percentage Error:

```
WMAPE = 100 * SUM(|actual - forecast|) / SUM(|actual|)
```

**Properties:**
- Weighted by actual values (accounts for magnitude)
- Symmetric treatment of over/under forecasting
- Robust to scale differences across account types

**Acceptance Criterion:** WMAPE < 20%

**Horizons Evaluated:**
- 3-month: Short-term accuracy
- 6-month: Medium-term accuracy
- 12-month: Full horizon accuracy (primary metric)

### 6.2 Pipeline Stages

The forecasting engine processes data through these stages:

1. **Data Cleaning & Validation**
   - Normalize column names
   - Validate required fields
   - Convert dates to proper format

2. **Transfer Detection & Netting (Layer 0)**
   - Match by explicit TransferLinkID
   - Fallback: amount matching within 2 days
   - Remove internal transfers from external cash flow

3. **Recurrence Detection (Layer 0.5)**
   - Independent pattern discovery
   - Category stability analysis (CV < 15%)
   - Counterparty consistency detection
   - Fixes Single Point of Failure on upstream flags

4. **Monthly Aggregation**
   - Group transactions by month
   - Calculate Net External Cash Flow (NECF)

5. **Decomposition**
   - Split NECF into Deterministic Base and Residual
   - Deterministic: is_recurring=True OR discovered patterns
   - Residual: Variable/discretionary flows

6. **Model Selection**
   - Candidates: ETS, SARIMA, SARIMAX, TiRex
   - Selection: Lowest WMAPE wins
   - Tie-breaker: Simpler model preferred

7. **Forecasting**
   - 12-month horizon
   - 95% confidence intervals

### 6.3 Smart Recurring Mask Selection

The engine automatically detects and compensates for corrupted is_recurring_flag values:

**Decision Logic:**
1. Calculate coverage: % of transaction value marked recurring
2. Calculate stability: Coefficient of Variation of monthly deterministic totals
3. Compare original flags vs discovered patterns

**Selection Rules:**
- If original coverage >= 15% AND stability >= 0.5: Use original flags
- If discovered stability is significantly better (improvement >= 0.1): Use discovered
- Otherwise: Fall back to discovered patterns

**Impact:** With 10% flag corruption, pass rate improved from 40% to 73%.

### 6.4 Trend-Adjusted Projection

Replaces naive mean() with intelligent projection:

- Exponentially weighted recent values (recency_weight=0.7)
- CUSUM level shift detection for salary raises/rent changes
- Projects: base_value + (monthly_trend x months_ahead)

Fixes the "Mean Fallacy" where naive averaging fails on lifestyle changes.

---

## 7. Key Files Reference

### Core Engine

| File | Purpose |
|------|---------|
| `src/cashflow/engine/forecast.py` | Main ForecastEngine class |
| `src/cashflow/engine/config.py` | ForecastConfig parameters |
| `src/cashflow/cli.py` | Command-line interface |

### Pipeline

| File | Purpose |
|------|---------|
| `src/cashflow/pipeline/decomposition.py` | Cash flow decomposition + smart mask |
| `src/cashflow/pipeline/recurrence.py` | Layer 0.5 pattern discovery |
| `src/cashflow/pipeline/transfer.py` | Transfer detection and netting |
| `src/cashflow/pipeline/aggregation.py` | Monthly NECF aggregation |

### Models

| File | Purpose |
|------|---------|
| `src/cashflow/models/ets.py` | Exponential Smoothing (Holt-Winters) |
| `src/cashflow/models/sarima.py` | SARIMA and SARIMAX models |
| `src/cashflow/models/tirex.py` | TiRex ONNX neural network |
| `src/cashflow/models/selection.py` | Model selection by WMAPE |

### Test Framework

| File | Purpose |
|------|---------|
| `scripts/framework_config.py` | Account and randomness configurations |
| `scripts/data_generator.py` | Synthetic data generation |
| `scripts/run_framework_tests.py` | Test orchestration and plotting |

### Results

| File | Purpose |
|------|---------|
| `results/wmape_summary.csv` | Aggregate results (12 rows) |
| `results/wmape_results.csv` | Detailed results (120 rows) |

---

## 8. What to Expect

### Performance by Account Type

**SME Accounts (Best Performance)**
- WMAPE range: 2.6% - 7.0%
- Pass rate: 70-100%
- Characteristics: Multiple predictable revenue streams, stable operational expenses
- Strong performance even at High randomness

**Corporate Accounts (Consistent Performance)**
- WMAPE range: 5.5% - 16.5%
- Pass rate: 90-100%
- Characteristics: High volume smooths variations, predictable large obligations
- Maintains <17% WMAPE across all randomness levels

**Personal Accounts (Most Variable)**
- WMAPE range: 6.9% - 18.5%
- Pass rate: 40-100%
- Characteristics: Single income source, high discretionary spending variance
- Performance degrades significantly at High randomness

### Model Selection

TiRex (neural network) is selected 100% of the time, consistently outperforming:
- ETS (Exponential Smoothing)
- SARIMA (Seasonal ARIMA)
- SARIMAX (SARIMA with exogenous variables)

### Runtime Performance

- Single forecast: 1-3 seconds
- Full test suite (120 runs): 12-15 minutes
- Quick validation (36 runs): 4-5 minutes

### Confidence Intervals

95% confidence intervals provided on all forecasts:
- Personal: EUR 150-470 average width
- SME: EUR 140-980 average width
- Corporate: EUR 1,400-9,700 average width

Width scales with account transaction volume and randomness level.

---

## 9. Verification Steps

To verify the system is working correctly:

1. **Run quick test suite:**
   ```bash
   python scripts/run_framework_tests.py --quick
   ```

2. **Check results match expected ranges:**
   - SME-NONE WMAPE should be 2-5%
   - Corporate-HIGH WMAPE should be <20%
   - Overall pass rate should be >80%

3. **Verify plots generated:**
   - Check `plots/` directory for 7 PNG files
   - Open `plots/wmape_heatmap.png` to see performance matrix

4. **Test web interface:**
   ```bash
   cashflow-web
   # Upload data/PoC_UTF_Dataset.csv
   # Verify forecast completes with WMAPE < 20%
   ```

5. **Validate CSV output:**
   - `results/wmape_summary.csv` should have 12 rows
   - `results/wmape_results.csv` should have 36+ rows (quick) or 120 rows (full)

---

## 10. Contact and Support

For technical questions about the forecasting engine, refer to:
- `docs/sdd.md` - System Design Document v0.05
- `docs/compliance.md` - Acceptance test criteria
- `CLAUDE.md` - Development guidelines

For issues and feature requests:
- GitHub Issues: https://github.com/anthropics/claude-code/issues
