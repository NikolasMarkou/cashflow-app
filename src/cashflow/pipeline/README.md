# Pipeline

Data transformation pipeline for ingestion, cleaning, enrichment, and decomposition of financial transaction data.

**SDD Reference:** Sections 4-5, 8-10

## Files

| File | Description |
|------|-------------|
| `ingestion.py` | Load and validate UTF/CRF from CSV files |
| `cleaning.py` | Normalize, deduplicate, and validate data quality |
| `enrichment.py` | Merge UTF with CRF using precedence rules |
| `transfer.py` | Detect and net internal transfers (Layer 0) |
| `aggregation.py` | Aggregate transactions to monthly NECF |
| `decomposition.py` | Split NECF into deterministic and residual components |

## Key Functions

### `ingestion.py` - Data Loading

```python
def load_utf(path: str | Path, customer_id: Optional[str] = None) -> pd.DataFrame:
    """Load UTF CSV with flexible column mapping."""

def load_crf(path: str | Path) -> pd.DataFrame:
    """Load CRF CSV with normalized columns."""

def validate_utf(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Validate records against UTFRecord schema."""
```

### `cleaning.py` - Data Quality

```python
def clean_utf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean UTF data:
    - Remove invalid dates/amounts
    - Deduplicate by (account_id, tx_id)
    - Normalize booleans and currencies
    - Derive month_key from tx_date
    """

def validate_data_quality(df: pd.DataFrame) -> dict:
    """Generate quality metrics: completeness, date range, unique counts."""
```

### `enrichment.py` - CRF Merge

```python
def enrich_with_crf(utf_df: pd.DataFrame, crf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply CRF enrichment with precedence:
    End date: UTF > Loan > Card > Mandate > CRF
    Amount: UTF > Contractual > Historical median
    """
```

### `transfer.py` - Transfer Netting

```python
def detect_transfers(df: pd.DataFrame, tolerance_days: int = 2) -> pd.DataFrame:
    """
    Three-stage transfer detection:
    1. Explicit TransferLinkID matching
    2. Amount + date proximity (same customer, opposite direction, ±2 days)
    3. Category heuristics (TRANSFER_IN/OUT)
    """

def net_transfers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove matched transfer pairs, return external transactions only."""
```

### `aggregation.py` - Monthly Rollup

```python
def aggregate_monthly(df: pd.DataFrame, customer_id: Optional[str] = None) -> pd.DataFrame:
    """
    Aggregate to monthly NECF:
    - Sum credits and debits per month
    - Compute net external cash flow
    - Count transactions
    """

def fill_missing_months(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate gaps in monthly time series."""
```

### `decomposition.py` - Component Separation

```python
def decompose_cashflow(monthly_df: pd.DataFrame, utf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Split NECF = Deterministic Base + Residual
    - Deterministic: is_recurring_flag=True OR CRF-linked
    - Residual: Variable/discretionary flows
    """

def compute_known_future_delta(crf_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Extract future contract events (loan maturity, subscription end)."""
```

## Pipeline Flow

```
UTF CSV ─┬─► load_utf() ─► clean_utf() ─┬─► enrich_with_crf() ─► detect_transfers()
         │                               │
CRF CSV ─┴─► load_crf() ────────────────┘
                                         │
                                         ▼
                              net_transfers() ─► aggregate_monthly() ─► decompose_cashflow()
                                                                              │
                                                                              ▼
                                                                    DecomposedNECF DataFrame
```

## Usage

```python
from cashflow.pipeline import (
    load_utf, load_crf, clean_utf, enrich_with_crf,
    detect_transfers, net_transfers, aggregate_monthly, decompose_cashflow
)

# Load and clean
utf_df = load_utf("data/transactions.csv")
utf_df = clean_utf(utf_df)

# Optional CRF enrichment
crf_df = load_crf("data/counterparties.csv")
utf_df = enrich_with_crf(utf_df, crf_df)

# Transfer netting
utf_df = detect_transfers(utf_df, tolerance_days=2)
external_df, transfer_summary = net_transfers(utf_df)

# Aggregate and decompose
monthly_df = aggregate_monthly(external_df)
decomposed_df = decompose_cashflow(monthly_df, external_df)
```

## Dependencies

- `pandas` - Data manipulation
- `cashflow.schemas` - UTFRecord, CRFRecord validation
