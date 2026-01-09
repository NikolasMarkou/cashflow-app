# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Cash Flow Forecasting Predictive Engine - a production-grade Python package for predicting customer cash flow over a 12-month horizon using historical transaction data. Implements the SDD v0.05 layered forecasting architecture.

**Key metrics:**
- WMAPE < 20% required for model acceptance
- 95% confidence intervals on all forecasts
- Python 3.8+ compatible

## Quick Commands

```bash
# Development setup
pip install -e ".[dev,viz]"

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

# Generate visualization plots
python scripts/generate_all_plots.py
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

## Visualization Scripts

Located in `scripts/`, generate plots to `plots/`:
- `visualize_forecast.py` - Time series, components, model comparison (uses real UTF data)
- `visualize_scenarios.py` - Confidence interval fan chart, model performance
- `visualize_decomposition.py` - Decomposition panels, outlier analysis
- `generate_all_plots.py` - Run all visualizations
- `analyze_noise_sensitivity.py` - Noise sensitivity analysis with synthetic data

### Noise Sensitivity Analysis

Evaluates model robustness under increasing data noise levels:

```bash
python3 scripts/analyze_noise_sensitivity.py
```

**Noise Levels** (with flag corruption and salary raises to test improvements):
- Baseline (No Noise): Clean synthetic data, 0% flag corruption
- Very Low Noise: salary_std=25, 10% flag corruption
- Low Noise: salary_std=50, 20% flag corruption + salary raise at month 12
- Moderate Noise: salary_std=100, 30% flag corruption + salary raise
- High Noise: salary_std=200, 40% flag corruption + salary raise

**Output** (`plots/noise_analysis/`):
- `wmape_vs_noise.png` - WMAPE distribution across noise levels
- `forecast_trajectories.png` - Stacked subplots with historical + forecast per noise level
- `ci_width_vs_noise.png` - Confidence interval width comparison
- `outlier_detection.png` - Outlier counts by noise level
- `threshold_pass_rate.png` - Pass rate (WMAPE < 20%) by noise level
- `summary_table.csv` / `summary_table.png` - Aggregate metrics

Uses 30 random seeds per noise level for statistical robustness.

**Key Finding:** Recurrence detection compensates for corrupted flags - with 10% flag corruption, pass rate improved from 40% to 73%.

## Important Notes

- **docs/ is read-only** - Reference materials from client, do not modify
- **SDD v0.05 is authoritative** - Implementation must match spec exactly
- **Python 3.8 compatibility** - Use `from __future__ import annotations`, avoid newer syntax
- **Outlier audit trail** - Always preserve original values alongside treated values
