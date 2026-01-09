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
UTF → Cleaning → CRF Enrichment → Transfer Netting → NECF
    → Decomposition → Outlier Treatment → Model Selection
    → Recomposition → ExplainabilityPayload JSON
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
- Deterministic: transactions where `is_recurring_flag=True`
- Residual: non-recurring (modeled statistically)

### Outlier Detection (`outliers/detector.py`)
- Modified Z-Score using MAD (Median Absolute Deviation)
- Threshold: |MZ| > 3.5 marks as outlier
- Dual-value model: `original_value` and `treated_value` preserved for audit

### Model Selection (`models/selection.py`)
- Candidates: ETS (Holt-Winters), SARIMA, SARIMAX
- Selection: Lowest WMAPE wins (must be < 20%)
- Tie-breaker: prefer simpler model
- All forecasts include 95% confidence intervals

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
- `visualize_forecast.py` - Time series, components, model comparison
- `visualize_scenarios.py` - Multi-seed scenarios, WMAPE distribution
- `visualize_decomposition.py` - Decomposition panels, outlier analysis
- `generate_all_plots.py` - Run all visualizations

## Important Notes

- **docs/ is read-only** - Reference materials from client, do not modify
- **SDD v0.05 is authoritative** - Implementation must match spec exactly
- **Python 3.8 compatibility** - Use `from __future__ import annotations`, avoid newer syntax
- **Outlier audit trail** - Always preserve original values alongside treated values
