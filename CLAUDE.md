# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cash Flow Forecasting Predictive Engine - a production-grade Python package for predicting customer cash flow for 12 months using historical transaction data. Implements the SDD v0.05 layered forecasting architecture.

## Repository Structure

- `src/cashflow/` - Main package implementation
- `tests/` - pytest test suite with fixtures
- `docs/` - Client-provided reference materials (read-only)
  - `sdd.md` - Software Design Document (v0.05) - authoritative specification
  - `compliance.md` - Acceptance Test Plan and compliance criteria
  - `Scripts/` - Client's PoC implementation for reference

## Build and Development

```bash
# Install package in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=cashflow --cov-report=term-missing

# Type checking
mypy src/cashflow

# Linting
ruff check src/
```

## CLI Usage

```bash
# Run forecast
cashflow forecast --utf data/utf.csv --output ./output

# With CRF enrichment
cashflow forecast --utf data/utf.csv --crf data/crf.csv --output ./output

# Validate UTF data
cashflow validate --utf data/utf.csv

# Generate default config
cashflow init-config --output config.json
```

## Architecture (SDD v0.05)

**Layered Forecasting Stack:**
- Layer 0: Deterministic rules (transfer netting, recurrence detection)
- Layer 1: Statistical baselines (ETS, SARIMA, SARIMAX)
- Layer 2: ML residuals (optional - Ridge/ElasticNet)
- Layer 3: Recomposition & JSON explainability output

**Pipeline Data Flow:**
```
UTF → Cleaning → CRF Enrichment → Transfer Netting → NECF
    → Decomposition → Outlier Treatment → Model Selection
    → Recomposition → Explainability JSON
```

## Package Structure

```
src/cashflow/
├── schemas/       # Pydantic models (UTF, CRF, NECF, Forecast)
├── pipeline/      # Data transformation (ingestion → decomposition)
├── outliers/      # Detection (MZ-Score, IQR) and treatment
├── models/        # ETS, SARIMA, SARIMAX, model selection
├── features/      # Time features, exogenous variables
├── engine/        # ForecastEngine orchestrator, config
├── explainability/# JSON payload builder
├── cli.py         # Click CLI
└── utils.py       # WMAPE, date utilities
```

## Key Implementation Details

**Transfer Detection (pipeline/transfer.py):**
- Matches by explicit TransferLinkID
- Falls back to amount + date proximity (±2 days)
- Category heuristics (TRANSFER_IN/OUT, SAVINGS_CONTRIBUTION)

**Cash Flow Decomposition (pipeline/decomposition.py):**
- NECF = Deterministic Base + Residual
- Deterministic: is_recurring_flag=True transactions
- Residual: Everything else (modeled statistically)

**Outlier Detection (outliers/detector.py):**
- Modified Z-Score (MAD-based), threshold |z| > 3.5
- Dual-value model: preserves original for audit

**Model Selection (models/selection.py):**
- Lowest WMAPE wins (must be < 20%)
- Tie-breaker: simpler model
- 95% confidence intervals

**Recomposition (engine/forecast.py):**
```
Forecast_Total = Forecast_Residual + Deterministic_Base + KnownFutureFlow_Delta
```
