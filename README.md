# Cash Flow Forecasting Predictive Engine

A production-grade Python package for multi-account cash flow forecasting with layered architecture, transfer netting, cash flow decomposition, and LLM-ready explainability output.

**Implements SDD v0.05 specification.**

**Compliance: 97.6% (40/41 requirements)** - See [docs/compliance.md](docs/compliance.md)

## Features

- **Layered Forecasting Architecture**
  - Layer 0: Deterministic rules (transfer netting)
  - Layer 0.5: Internal recurrence detection (fixes upstream flag errors)
  - Layer 1: Statistical baselines (ETS, SARIMA, SARIMAX with exogenous integration)
  - Layer 2: ML residuals (optional)
  - Layer 3: Recomposition with trend-adjusted projection & explainability

- **Transfer Netting**: Automatically detects and removes internal transfers between accounts

- **Recurrence Detection**: Internal pattern discovery independent of upstream `is_recurring_flag`

- **Cash Flow Decomposition**: Separates NECF into deterministic base and residual components

- **Trend-Adjusted Projection**: Handles salary raises, rent changes with level shift detection

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

# Install with web interface support
pip install -e ".[web]"
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

## Web Interface

A FastAPI-based web interface provides interactive forecasting with Plotly.js charts.

### Start the Server

```bash
# Using the CLI command
cashflow-web

# Or using uvicorn directly
uvicorn cashflow.web.app:app --host 0.0.0.0 --port 8000 --reload
```

Access the interface at **http://localhost:8000**

### Features

- **CSV Upload**: Upload UTF transaction data directly in the browser
- **Full Configuration**: Customize all forecast parameters
  - Forecast horizon (1-24 months)
  - WMAPE threshold
  - Outlier detection method and threshold
  - Outlier treatment method
  - Model selection (ETS, SARIMA, SARIMAX)
  - Confidence level (90%, 95%, 99%)
- **Interactive Charts** (Plotly.js):
  - Historical + Forecast time series with confidence intervals
  - Model comparison (WMAPE bar chart)
  - Forecast component breakdown (deterministic + residual)
  - Outlier analysis (original vs treated values)
- **Metrics Dashboard**: WMAPE, selected model, threshold status, confidence level
- **Statistics Summary**: Decomposition metrics, transfer netting summary

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main forecast page |
| `/api/forecast` | POST | Run forecast with CSV upload |
| `/docs` | GET | Swagger API documentation |
| `/openapi.json` | GET | OpenAPI specification |

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
│   ├── recurrence.py  # Layer 0.5 pattern discovery
│   ├── aggregation.py # Monthly NECF construction
│   └── decomposition.py # Deterministic/residual split + trend projection
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
├── web/               # FastAPI web interface
│   ├── app.py         # Application factory
│   ├── routes/        # API and page routes
│   ├── schemas/       # Response models
│   ├── templates/     # Jinja2 HTML templates
│   ├── static/        # CSS and JavaScript
│   └── README.md      # Web module documentation
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

## Visualization

Generate plots for analysis and presentation:

```bash
# Generate all standard plots (uses real UTF data)
python3 scripts/generate_all_plots.py

# Run noise sensitivity analysis (uses synthetic data)
python3 scripts/analyze_noise_sensitivity.py
```

**Standard Plots** (`plots/`):
- Forecast time series with confidence intervals
- Forecast component breakdown
- Model comparison charts
- Cash flow decomposition
- Outlier analysis

**Noise Sensitivity Analysis** (`plots/noise_analysis/`):
- Evaluates model robustness under 5 noise levels
- Uses 30 random seeds per level for statistical significance
- Generates WMAPE distributions, forecast trajectories, CI width comparisons
- Outputs summary table with pass rates and metrics

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

### Recurrence Detection (Layer 0.5)

Internal pattern discovery independent of upstream flags:
1. Category stability analysis (low coefficient of variation)
2. Counterparty consistency detection
3. Amount cluster detection (fixed payments)

Compensates for corrupted/missing `is_recurring_flag` values.

### Cash Flow Decomposition (SDD Section 10)

```
NECF = Deterministic Base + Residual
```
- **Deterministic**: `IsRecurringFlag=True` OR CRF-linked OR discovered by recurrence detection
- **Residual**: Variable/discretionary flows

### Trend-Adjusted Projection

Replaces naive mean() with intelligent projection:
1. Exponentially weighted recent values (recency_weight=0.7)
2. Level shift detection using CUSUM approach
3. Linear trend projection for lifestyle changes (salary raises, rent changes)

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

### Noise Sensitivity Results

Model robustness under synthetic data with increasing noise and flag corruption (30 seeds per level):

| Noise Level | Flag Corruption | WMAPE Mean | Pass Rate |
|-------------|-----------------|------------|-----------|
| Baseline (No Noise) | 0% | 20.32% | 63% |
| Very Low Noise | 10% | 17.79% | **73%** |
| Low Noise + Salary Raise | 20% | 46.43% | 30% |
| Moderate Noise + Raise | 30% | 87.97% | 10% |
| High Noise + Raise | 40% | 88.82% | 0% |

**Key Finding:** With 10% flag corruption, the recurrence detection improvement raises pass rate from 40% to 73% - the system performs BETTER with corrupted flags than the old system with perfect flags.

## SDD v0.05 Compliance

| Category | Score |
|----------|-------|
| UTF Schema | 100% |
| CRF Schema | 100% |
| Data Cleaning | 100% |
| Transfer Detection | 83% |
| Decomposition | 100% |
| Outlier Detection | 100% |
| Feature Engineering | 100% |
| Predictive Modeling | 100% |
| Recomposition | 100% |
| Explainability | 100% |
| **Overall** | **97.6%** |

See [docs/compliance.md](docs/compliance.md) for detailed verification.

## License

Proprietary - All rights reserved.

## References

- SDD v0.05: `docs/sdd.md`
- Compliance Report: `docs/compliance.md`
