# Cash Flow Forecasting Predictive Engine

A production-grade Python package for multi-account cash flow forecasting with layered architecture, transfer netting, cash flow decomposition, and LLM-ready explainability output.

**Implements SDD v0.05 specification.**

## Features

- **Layered Forecasting Architecture**
  - Layer 0: Deterministic rules (transfer netting, recurrence detection)
  - Layer 1: Statistical baselines (ETS, SARIMA, SARIMAX)
  - Layer 2: ML residuals (optional)
  - Layer 3: Recomposition & explainability

- **Transfer Netting**: Automatically detects and removes internal transfers between accounts

- **Cash Flow Decomposition**: Separates NECF into deterministic base and residual components

- **Outlier Detection**: Modified Z-Score and IQR methods with dual-value audit trail

- **Model Selection**: Automatic selection based on WMAPE with configurable thresholds

- **Explainability**: JSON payload designed for LLM consumption

## Installation

### Requirements

- Python 3.8+
- pandas, numpy, scipy, statsmodels, scikit-learn, pydantic, click

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd cashflow-app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install the package
pip install --upgrade pip
pip install -e .
```

## Quick Start

### Run a Forecast

```bash
# Basic forecast
cashflow forecast --utf data/transactions.csv --output ./output

# With CRF enrichment and verbose output
cashflow forecast --utf data/utf.csv --crf data/crf.csv -o ./output -v

# Filter by customer
cashflow forecast --utf data/utf.csv --customer-id CUST001 -o ./output
```

### Validate Data

```bash
cashflow validate --utf data/transactions.csv -v
```

### Generate Config Template

```bash
cashflow init-config -o config.json
```

## Input Data Formats

### UTF (Unified Transaction Feed)

Required columns:
| Column | Description |
|--------|-------------|
| `TransactionID` / `tx_id` | Unique transaction identifier |
| `TransactionDate` / `tx_date` | Transaction date (YYYY-MM-DD) |
| `AccountID` / `account_id` | Account identifier |
| `Amount` / `amount` | Transaction amount (negative for debits) |
| `CurrencyCode` / `currency` | Currency code (e.g., EUR, USD) |
| `CategoryCode` / `category` | Transaction category |
| `IsRecurringFlag` / `is_recurring_flag` | Boolean for recurring transactions |

Optional columns: `CustomerID`, `TransferLinkID`, `DescriptionRaw`, `CounterpartyKey`

### CRF (Counterparty Reference Feed)

| Column | Description |
|--------|-------------|
| `CounterpartyKey` | Unique counterparty identifier |
| `CustomerID` | Customer identifier |
| `ContractType` | Type of contract (LOAN, SUBSCRIPTION, etc.) |
| `ContractualAmount` | Expected recurring amount |
| `RecurrenceEndDate` | Contract end date (for future adjustments) |

## Output

### Explainability JSON

The forecast produces a comprehensive JSON payload (`forecast_summary.json`):

```json
{
  "model_selected": "ETS",
  "wmape_winner": 1.818,
  "meets_threshold": true,
  "confidence_level": "High",
  "forecast_start": "2026-01",
  "forecast_end": "2026-12",
  "decomposition_summary": {
    "avg_necf": 907.28,
    "avg_deterministic_base": 1720.95,
    "avg_residual": -813.67
  },
  "transfer_netting_summary": {
    "num_transfers_removed": 24,
    "total_volume_removed": 12000.0
  },
  "outliers_detected": [...],
  "forecast_results": [...]
}
```

### Forecast CSV

Monthly forecasts with confidence intervals (`forecast_results.csv`):

| month_key | forecast_total | lower_ci | upper_ci |
|-----------|----------------|----------|----------|
| 2026-01 | 826.15 | 799.12 | 853.18 |
| 2026-02 | 824.38 | 797.35 | 851.41 |
| ... | ... | ... | ... |

## Configuration

Create a `config.json` file to customize behavior:

```json
{
  "forecast_horizon": 12,
  "test_size": 4,
  "wmape_threshold": 20.0,
  "models_to_evaluate": ["ets", "sarima"],
  "outlier_method": "modified_zscore",
  "outlier_threshold": 3.5,
  "outlier_treatment": "median",
  "transfer_date_tolerance_days": 2
}
```

## Project Structure

```
src/cashflow/
├── schemas/           # Pydantic data models
│   ├── utf.py         # Unified Transaction Feed
│   ├── crf.py         # Counterparty Reference Feed
│   ├── necf.py        # Net External Cash Flow
│   └── forecast.py    # Forecast & explainability output
│
├── pipeline/          # Data transformation
│   ├── ingestion.py   # Load/validate UTF & CRF
│   ├── cleaning.py    # Normalize, dedupe, validate
│   ├── enrichment.py  # UTF-CRF join with precedence
│   ├── transfer.py    # Transfer detection & netting
│   ├── aggregation.py # Monthly NECF construction
│   └── decomposition.py # Deterministic/residual split
│
├── outliers/          # Outlier handling
│   ├── detector.py    # MZ-Score, IQR, Isolation Forest
│   └── treatment.py   # Median imputation, dual-value model
│
├── models/            # Forecasting models
│   ├── base.py        # Abstract ForecastModel
│   ├── ets.py         # Exponential Smoothing
│   ├── sarima.py      # SARIMA / SARIMAX
│   └── selection.py   # WMAPE comparison, model selection
│
├── engine/            # Orchestration
│   ├── config.py      # Runtime configuration
│   └── forecast.py    # Main ForecastEngine class
│
├── explainability/    # LLM-ready output
│   └── builder.py     # JSON payload generation
│
├── cli.py             # Click CLI entrypoint
└── utils.py           # Metrics (WMAPE), dates, validation
```

## API Usage

```python
from cashflow import ForecastEngine, ForecastConfig

# Configure the engine
config = ForecastConfig(
    forecast_horizon=12,
    wmape_threshold=20.0,
    models_to_evaluate=["ets", "sarima"],
)

# Initialize and run
engine = ForecastEngine(config=config)
result = engine.run("data/utf.csv", crf_path="data/crf.csv")

# Access results
print(f"Model: {result.model_selected}")
print(f"WMAPE: {result.wmape_winner:.2f}%")

for forecast in result.forecast_results:
    print(f"{forecast.month_key}: {forecast.forecast_total:.2f}")
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=cashflow

# Run specific test module
pytest tests/unit/test_models.py -v
```

### Test Coverage

| Module | Coverage |
|--------|----------|
| schemas | 90%+ |
| pipeline | 60%+ |
| outliers | 70%+ |
| models | 75%+ |
| engine | 74% |

## Key Algorithms

### Transfer Detection (SDD Section 9.2)

Matches internal transfers using:
1. Explicit `TransferLinkID` if present
2. Amount matching with opposite direction (±2 day tolerance)
3. Category heuristics (`TRANSFER_IN`, `TRANSFER_OUT`)

### Cash Flow Decomposition (SDD Section 10)

```
NECF = Deterministic Base + Residual
```
- **Deterministic**: `IsRecurringFlag=True` OR CRF-linked contracts
- **Residual**: Variable/discretionary flows

### Model Selection (SDD Section 13.5)

1. Lowest WMAPE wins
2. Tie-breaker: simpler model (ETS < SARIMA < SARIMAX)
3. Explainability override within 0.5pp tolerance

### Forecast Recomposition (SDD Section 14.1)

```
Forecast_Total = Forecast_Residual + Deterministic_Base + KnownFutureFlow_Delta
```

## Performance Metrics

On the PoC dataset (411 transactions, 24 months):

| Metric | Value |
|--------|-------|
| ETS WMAPE | 1.818% |
| SARIMA WMAPE | 2.761% |
| Threshold | 20.0% |
| Outliers Detected | 3 |
| Transfers Netted | 24 |

## License

Proprietary - All rights reserved.

## References

- SDD v0.05: `docs/sdd.md`
- Compliance Report: `docs/compliance.md`
