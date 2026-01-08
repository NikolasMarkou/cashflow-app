# Engine

Main orchestration layer implementing the complete SDD v0.05 forecasting pipeline.

**SDD Reference:** Sections 13-15

## Files

| File | Description |
|------|-------------|
| `config.py` | Runtime configuration with validation |
| `forecast.py` | ForecastEngine - main orchestrator class |

## Configuration

### `config.py`

```python
class ForecastConfig(BaseModel):
    """Runtime parameters for the forecast pipeline."""

    # Data settings
    lookback_min_months: int = 24        # Minimum history required
    forecast_horizon: int = 12           # Months to forecast
    test_size: int = 4                   # Holdout months for validation

    # Outlier settings
    outlier_method: str = "modified_zscore"
    outlier_threshold: float = 3.5
    outlier_treatment: str = "median"

    # Model settings
    models_to_evaluate: List[str] = ["ets", "sarima"]
    arima_order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 12)
    enable_ml_layer: bool = False        # Layer 2 (optional)

    # Thresholds
    wmape_threshold: float = 20.0        # Maximum acceptable WMAPE
    model_tie_tolerance: float = 0.5     # Percentage points

    # Transfer detection
    transfer_date_tolerance_days: int = 2

def get_default_config() -> ForecastConfig:
    """Get default configuration."""

def load_config(path: str | Path) -> ForecastConfig:
    """Load configuration from JSON file."""
```

## Forecast Engine

### `forecast.py`

```python
class ForecastEngine:
    """
    Main orchestrator implementing SDD v0.05 layered architecture:

    Layer 0: Deterministic rules (transfer netting, recurrence)
    Layer 1: Statistical baselines (ETS, SARIMA, SARIMAX)
    Layer 2: ML residuals (optional)
    Layer 3: Recomposition & explainability
    """

    def __init__(self, config: Optional[ForecastConfig] = None):
        self.config = config or ForecastConfig()

    def run(
        self,
        utf_path: str | Path,
        crf_path: Optional[str | Path] = None,
        customer_id: Optional[str] = None,
    ) -> ExplainabilityPayload:
        """
        Run complete forecast pipeline from files.

        Returns ExplainabilityPayload with full results.
        """

    def run_from_dataframe(
        self,
        utf_df: pd.DataFrame,
        crf_df: Optional[pd.DataFrame] = None,
    ) -> ExplainabilityPayload:
        """Run forecast from DataFrames (for API/testing)."""
```

## Pipeline Phases

The `run()` method executes these phases sequentially:

| Phase | Method | Description |
|-------|--------|-------------|
| 1 | `_ingest_and_clean()` | Load UTF, clean, validate |
| 2 | (inline) | Load and merge CRF if provided |
| 3 | (inline) | Detect transfers (Layer 0) |
| 4 | (inline) | Net internal transfers |
| 5 | (inline) | Aggregate to monthly NECF |
| 6 | (inline) | Decompose (deterministic + residual) |
| 7 | (inline) | Detect and treat outliers |
| 8 | `_train_and_select_model()` | Train models, select winner (Layer 1) |
| 9 | `_recompose_forecast()` | Combine components (Layer 3) |
| 10 | `_generate_explainability()` | Build output payload |

## Forecast Recomposition

Per SDD Section 14.1:

```python
Forecast_Total = Forecast_Residual + Deterministic_Base + KnownFutureFlow_Delta
```

Where:
- `Forecast_Residual`: Model output (ETS/SARIMA)
- `Deterministic_Base`: Average of historical deterministic component
- `KnownFutureFlow_Delta`: Contract events from CRF (loan endings, etc.)

## Usage

```python
from cashflow.engine import ForecastEngine, ForecastConfig

# Custom configuration
config = ForecastConfig(
    forecast_horizon=12,
    wmape_threshold=20.0,
    outlier_method="modified_zscore",
    outlier_threshold=3.5,
    models_to_evaluate=["ets", "sarima"],
)

# Initialize engine
engine = ForecastEngine(config=config)

# Run from files
result = engine.run(
    utf_path="data/transactions.csv",
    crf_path="data/counterparties.csv",
)

# Access results
print(f"Model: {result.model_selected}")
print(f"WMAPE: {result.wmape_winner:.2f}%")
print(f"Confidence: {result.confidence_level}")

for fc in result.forecast_results:
    print(f"{fc.month_key}: {fc.forecast_total:.2f} ({fc.lower_ci:.2f} - {fc.upper_ci:.2f})")
```

### From DataFrames

```python
import pandas as pd

utf_df = pd.read_csv("data/transactions.csv")
crf_df = pd.read_csv("data/counterparties.csv")

result = engine.run_from_dataframe(utf_df, crf_df)
```

### With JSON Config

```python
from cashflow.engine.config import load_config

config = load_config("config.json")
engine = ForecastEngine(config=config)
```

## Internal State

The engine maintains internal state for explainability:

```python
engine._transfer_summary      # Transfer netting results
engine._decomposition_summary # Decomposition statistics
engine._outlier_records       # Detected outliers
engine._model_selector        # Model evaluation results
```

## Dependencies

- `cashflow.pipeline` - All pipeline modules
- `cashflow.outliers` - Detection and treatment
- `cashflow.models` - Forecasting models
- `cashflow.schemas.forecast` - Output schemas
