# Schemas

Pydantic data models defining the data contracts for all pipeline stages.

**SDD Reference:** Sections 4, 5, 9-10, 14-15

## Files

| File | Description |
|------|-------------|
| `utf.py` | Unified Transaction Feed record schema |
| `crf.py` | Counterparty Reference Feed schema |
| `necf.py` | Net External Cash Flow and decomposed records |
| `forecast.py` | Forecast output and explainability payload |

## Key Classes

### `utf.py` - Transaction Records

```python
class Direction(str, Enum):
    CREDIT = "CREDIT"
    DEBIT = "DEBIT"

class UTFRecord(BaseModel):
    customer_id: str
    account_id: str
    tx_id: str
    tx_date: date
    amount: float          # Always positive
    currency: str          # ISO 4217 (EUR, USD)
    direction: Direction
    category: str
    is_recurring_flag: bool = False
    transfer_link_id: Optional[str] = None
    counterparty_key: Optional[str] = None
```

### `crf.py` - Counterparty Contracts

```python
class ContractType(str, Enum):
    LOAN = "LOAN"
    CARD_INSTALLMENT = "CARD_INSTALLMENT"
    MANDATE = "MANDATE"
    SUBSCRIPTION = "SUBSCRIPTION"
    GENERIC = "GENERIC"

class CRFRecord(BaseModel):
    counterparty_key: str
    customer_id: str
    contract_type: ContractType
    contractual_amount: Optional[float]
    recurrence_end_date: Optional[date]  # Authoritative end date
    is_variable_amount: bool = False
```

### `necf.py` - Cash Flow Records

```python
class NECFRecord(BaseModel):
    customer_id: str
    month_key: str         # YYYY-MM format
    necf: float            # Net External Cash Flow
    credit_total: float
    debit_total: float
    transaction_count: int

class DecomposedNECF(NECFRecord):
    deterministic_base: float
    residual: float
    residual_original: Optional[float]  # Pre-treatment value
    residual_clean: Optional[float]     # Post-treatment value
    is_outlier: bool = False
    outlier_score: Optional[float]
    treatment_tag: Optional[str]        # ABNORMAL_EXTERNAL_FLOW
```

### `forecast.py` - Output Schemas

```python
class ForecastResult(BaseModel):
    month_key: str
    forecast_total: float
    forecast_residual: float
    deterministic_base: float
    known_future_delta: float = 0.0
    lower_ci: float
    upper_ci: float

class ModelCandidate(BaseModel):
    model_name: str
    wmape: float
    is_winner: bool = False

class ExplainabilityPayload(BaseModel):
    model_selected: str
    model_candidates: List[ModelCandidate]
    wmape_winner: float
    meets_threshold: bool
    decomposition_summary: DecompositionSummary
    transfer_netting_summary: TransferNettingSummary
    outliers_detected: List[OutlierRecord]
    forecast_results: List[ForecastResult]
```

## Usage

```python
from cashflow.schemas import UTFRecord, Direction, ForecastResult

# Create a transaction record
tx = UTFRecord(
    customer_id="CUST001",
    account_id="ACC001",
    tx_id="TX123",
    tx_date=date(2024, 1, 15),
    amount=1200.00,
    currency="EUR",
    direction=Direction.DEBIT,
    category="RENT_MORTGAGE",
    is_recurring_flag=True,
)

# Access computed properties
print(tx.signed_amount)  # -1200.00
print(tx.month_key)      # "2024-01"
```

## Dependencies

- `pydantic` - Data validation and serialization
- No internal cashflow dependencies (base layer)
