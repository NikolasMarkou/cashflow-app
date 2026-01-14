# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Cash Flow Forecasting Predictive Engine - a production-grade Python package for predicting customer cash flow over a 12-month horizon using historical transaction data. Implements the SDD v0.05 layered forecasting architecture.

**Key metrics:**
- WMAPE < 20% required for model acceptance
- 95% confidence intervals on all forecasts
- Python 3.8+ compatible

## Virtual Environment Setup

**Always activate the virtual environment before running any commands:**

```bash
# Activate virtual environment
source .venv/bin/activate

# First-time setup (creates venv and installs dependencies)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,viz]"
```

## Quick Commands

```bash
# Activate venv first!
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=cashflow --cov-report=term-missing

# Type checking
mypy src/cashflow

# Linting
ruff check src/

# Run CLI forecast
cashflow forecast --utf data/utf.csv --output ./output

# Run framework tests (full suite - 120 runs)
python scripts/run_framework_tests.py

# Run framework tests (quick validation - 36 runs)
python scripts/run_framework_tests.py --quick
```

## Repository Structure

```
├── src/cashflow/          # Main package
│   ├── schemas/           # Pydantic models (UTF, CRF, NECF, Forecast)
│   ├── pipeline/          # Data transformation stages
│   ├── outliers/          # Detection and treatment
│   ├── models/            # ETS, SARIMA, SARIMAX implementations
│   ├── features/          # Time features, exogenous variables
│   ├── engine/            # ForecastEngine orchestrator
│   ├── explainability/    # JSON payload builder
│   ├── cli.py             # Click CLI entry point
│   └── utils.py           # WMAPE, date utilities
├── tests/                 # pytest suite with fixtures
├── scripts/               # Visualization scripts (matplotlib)
├── plots/                 # Generated visualization outputs
└── docs/                  # Reference materials (read-only)
    ├── sdd.md             # SDD v0.05 - authoritative spec
    ├── compliance.md      # Acceptance test criteria
    └── Scripts/           # Client's PoC implementation
```

## Architecture (SDD v0.05)

**Layered Forecasting Stack:**
- **Layer 0:** Deterministic rules (transfer netting, recurrence detection)
- **Layer 1:** Statistical baselines (ETS, SARIMA, SARIMAX)
- **Layer 2:** ML residuals (optional - Ridge/ElasticNet)
- **Layer 3:** Recomposition & JSON explainability output

**Pipeline Data Flow:**
```
UTF → Cleaning → CRF Enrichment → Transfer Netting
    → Recurrence Detection (Layer 0.5) → NECF Aggregation
    → Decomposition → Outlier Treatment → Model Selection
    → Recomposition (Trend-Adjusted) → ExplainabilityPayload JSON
```

**Core Formula:**
```
Forecast_Total = Deterministic_Base + Forecast_Residual + KnownFutureFlow_Delta
```

## Key Implementation Details

### Transfer Detection (`pipeline/transfer.py`)
- Matches by explicit `TransferLinkID` field
- Falls back to amount matching + date proximity (±2 days default)
- Category heuristics: `TRANSFER_IN`, `TRANSFER_OUT`, `SAVINGS_CONTRIBUTION`
- Function: `detect_transfers(df, date_tolerance_days=2)`

### Cash Flow Decomposition (`pipeline/decomposition.py`)
- NECF = Deterministic Base + Residual
- Deterministic: transactions where `is_recurring_flag=True` OR discovered by recurrence detection
- Residual: non-recurring (modeled statistically)
- Trend-adjusted projection for deterministic base (handles salary raises, rent changes)

### Recurrence Detection (`pipeline/recurrence.py`)
- **Layer 0.5** internal pattern discovery independent of upstream `is_recurring_flag`
- Detects patterns by: category stability, counterparty consistency, amount clustering
- Functions:
  - `discover_recurring_patterns(df)` - Finds recurring transaction patterns
  - `apply_discovered_recurrence(df, patterns)` - Tags transactions as recurring
- Fixes Single Point of Failure on external flags
- With 10% flag corruption: Pass rate improved from 40% to 73%

### Trend-Adjusted Projection (`pipeline/decomposition.py`)
- `DeterministicProjection` dataclass with base_value, monthly_trend, confidence
- `compute_deterministic_projection(df)` - Exponentially weighted mean + trend
- `_detect_level_shift(values)` - CUSUM approach to detect salary raises, rent changes
- Fixes "Mean Fallacy" where naive mean() fails on lifestyle changes

### Outlier Detection (`outliers/detector.py`)
- Modified Z-Score using MAD (Median Absolute Deviation)
- Threshold: |MZ| > 3.5 marks as outlier
- Dual-value model: `original_value` and `treated_value` preserved for audit

### Model Selection (`models/selection.py`)
- Candidates: ETS (Holt-Winters), SARIMA, SARIMAX
- Selection: Lowest WMAPE wins (must be < 20%)
- Tie-breaker: prefer simpler model
- All forecasts include 95% confidence intervals
- SARIMAX with exogenous integration: learns contract→cashflow relationships during training

### Explainability Output (`explainability/builder.py`)
- Produces `ExplainabilityPayload` JSON with:
  - Monthly forecast results with CI bounds
  - Model candidates and selection rationale
  - Outliers detected with treatment details
  - Human-readable explanations

## Common Patterns

### Running a Forecast Programmatically
```python
from cashflow.engine import ForecastEngine, ForecastConfig

engine = ForecastEngine(ForecastConfig())
payload = engine.run_from_dataframe(utf_df)  # Returns ExplainabilityPayload
```

### Pipeline Stages
```python
from cashflow.pipeline import (
    clean_utf,
    detect_transfers,
    net_transfers,
    aggregate_monthly,
)
from cashflow.pipeline.decomposition import decompose_cashflow

utf_df = clean_utf(utf_df)
utf_df = detect_transfers(utf_df, date_tolerance_days=2)
external_df, internal_df = net_transfers(utf_df)
monthly_df = aggregate_monthly(external_df)
decomposed_df = decompose_cashflow(monthly_df, external_df)
```

### Outlier Treatment
```python
from cashflow.outliers.detector import detect_outliers
from cashflow.outliers.treatment import apply_residual_treatment

outliers = detect_outliers(residuals, method="modified_zscore", threshold=3.5)
treated_df = apply_residual_treatment(df, outliers)
```

## Testing

Tests are in `tests/` using pytest. Key test files:
- `test_pipeline.py` - Data transformation tests
- `test_models.py` - ETS/SARIMA model tests
- `test_engine.py` - End-to-end forecast tests
- `test_outliers.py` - Outlier detection tests

Run specific test: `pytest tests/test_engine.py -v`

## Testing Framework

See `docs/2026_01_13_framework.md` for the authoritative testing specification.

### Scripts

Located in `scripts/`:
- `framework_config.py` - Configuration dataclasses for account types and randomness levels
- `data_generator.py` - Synthetic transaction data generation
- `run_framework_tests.py` - Main test execution script

### Running Tests

```bash
# Activate venv first
source .venv/bin/activate

# Full test suite (120 runs: 3 account types x 4 randomness levels x 10 seeds)
python scripts/run_framework_tests.py

# Quick validation (36 runs: 3 seeds per config)
python scripts/run_framework_tests.py --quick

# Filter by account type
python scripts/run_framework_tests.py --account-type personal

# Filter by randomness level
python scripts/run_framework_tests.py --randomness low
```

### Account Types
- **Personal**: EUR 3K monthly income, single salary, consumer spending patterns
- **SME**: EUR 25K monthly revenue, multiple customers, business operations
- **Corporate**: EUR 500K monthly revenue, diversified enterprise patterns

### Randomness Levels
- **None**: Perfectly predictable baseline (0% variation)
- **Low**: Minor natural variation (2-5% CV, 5% flag corruption)
- **Medium**: Realistic business conditions (5-15% CV, 15% flag corruption)
- **High**: Stressed/volatile conditions (10-30% CV, 30% flag corruption)

### Output

Results saved to `results/`:
- `wmape_results.csv` - Detailed results per run (120 rows)
- `wmape_summary.csv` - Aggregated statistics per configuration (12 rows)

Plots saved to `plots/`:
- `wmape_by_account_type.png` - WMAPE by account type (grouped by randomness)
- `wmape_by_randomness.png` - WMAPE by randomness (grouped by account type)
- `wmape_heatmap.png` - 3x4 heatmap of 12-month WMAPE
- `wmape_horizon_degradation.png` - WMAPE by forecast month (1-12)
- `pass_rate_matrix.png` - Pass rate heatmap (WMAPE < 20%)
- `forecast_trajectories.png` - Sample forecast vs actual plots

## Important Notes

- **docs/ is read-only** - Reference materials from client, do not modify
- **SDD v0.05 is authoritative** - Implementation must match spec exactly
- **Python 3.8 compatibility** - Use `from __future__ import annotations`, avoid newer syntax
- **Outlier audit trail** - Always preserve original values alongside treated values
