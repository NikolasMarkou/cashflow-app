# Models

Time series forecasting models implementing Layer 1 of the SDD architecture.

**SDD Reference:** Section 13

## Files

| File | Description |
|------|-------------|
| `base.py` | Abstract base class and shared utilities |
| `ets.py` | Exponential Smoothing (ETS) model |
| `sarima.py` | SARIMA and SARIMAX models |
| `selection.py` | Model comparison and winner selection |

## Model Hierarchy

```
ForecastModel (ABC)
├── ETSModel        (complexity: 1)
├── SARIMAModel     (complexity: 2)
└── SARIMAXModel    (complexity: 3)
```

## Key Classes

### `base.py` - Abstract Interface

```python
class ForecastModel(ABC):
    """Abstract base for all forecasting models."""

    @abstractmethod
    def fit(self, series: pd.Series) -> "ForecastModel":
        """Fit model to historical data."""

    @abstractmethod
    def predict(self, steps: int, confidence: float = 0.95) -> ForecastOutput:
        """Generate forecast with confidence intervals."""

    def fit_predict(self, series: pd.Series, steps: int) -> ForecastOutput:
        """Convenience method: fit then predict."""

    def evaluate(self, train: pd.Series, test: pd.Series) -> float:
        """Calculate WMAPE on test set."""

@dataclass
class ForecastOutput:
    forecast_mean: np.ndarray
    forecast_lower: np.ndarray
    forecast_upper: np.ndarray
    month_keys: list[str]
    wmape: Optional[float] = None
```

### `ets.py` - Exponential Smoothing

```python
class ETSModel(ForecastModel):
    """
    Exponential Smoothing with additive trend and seasonality.
    Complexity score: 1 (simplest, preferred for tie-breaking)
    """

    def __init__(
        self,
        trend: str = "add",        # "add", "mul", or None
        seasonal: str = "add",      # "add", "mul", or None
        seasonal_periods: int = 12,
    ):
        ...
```

### `sarima.py` - SARIMA/SARIMAX

```python
class SARIMAModel(ForecastModel):
    """
    Seasonal ARIMA model.
    Complexity score: 2
    """

    def __init__(
        self,
        order: tuple = (1, 1, 1),           # (p, d, q)
        seasonal_order: tuple = (1, 1, 0, 12),  # (P, D, Q, s)
    ):
        ...

class SARIMAXModel(SARIMAModel):
    """
    SARIMAX with exogenous variables.
    Complexity score: 3 (most complex)
    """

    def fit(self, series: pd.Series, exog: pd.DataFrame = None):
        ...

    def predict(self, steps: int, exog_future: pd.DataFrame = None):
        ...
```

### `selection.py` - Model Selection

```python
class ModelSelector:
    """
    Compare models and select winner based on WMAPE.

    Selection rules (SDD Section 13.5):
    1. Lowest WMAPE wins
    2. Tie-breaker: simpler model if within tolerance
    3. Warn if winner exceeds threshold
    """

    def __init__(
        self,
        wmape_threshold: float = 20.0,
        tie_tolerance: float = 0.5,  # percentage points
    ):
        ...

    def evaluate_model(
        self,
        model: ForecastModel,
        train_series: pd.Series,
        test_series: pd.Series,
        forecast_steps: int,
    ) -> dict:
        """Evaluate a single model, store results."""

    def select_winner(self) -> ForecastModel:
        """Select best model based on evaluation results."""

    def get_summary(self) -> dict:
        """Get summary with all results and winner info."""
```

## WMAPE Metric

```python
def calculate_wmape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error.

    WMAPE = sum(|actual - forecast|) / sum(|actual|) * 100

    - Handles zero actuals gracefully
    - Returns percentage (e.g., 5.5 means 5.5%)
    """
```

## Usage

```python
from cashflow.models import ETSModel, SARIMAModel, ModelSelector
from cashflow.utils import split_train_test

# Prepare data
series = df.set_index("month_key")["residual_clean"]
train, test = split_train_test(series, test_size=4)

# Initialize selector
selector = ModelSelector(wmape_threshold=20.0, tie_tolerance=0.5)

# Evaluate models
selector.evaluate_model(ETSModel(), train, test, forecast_steps=12)
selector.evaluate_model(SARIMAModel(), train, test, forecast_steps=12)

# Select winner
winner = selector.select_winner()
summary = selector.get_summary()

print(f"Winner: {summary['winner']} with WMAPE {summary['winner_wmape']:.2f}%")

# Generate forecast
forecast_output = winner.predict(steps=12)
```

## Complexity Tie-Breaking

When models are within `tie_tolerance` (default 0.5pp), the simpler model wins:

| Model | Complexity |
|-------|------------|
| ETS | 1 (preferred) |
| SARIMA | 2 |
| SARIMAX | 3 |

Example: If ETS has WMAPE 5.2% and SARIMA has 5.0%, SARIMA wins (0.2pp difference).
If ETS has 5.2% and SARIMA has 5.5%, ETS wins (simpler and within 0.3pp).

## Dependencies

- `statsmodels` - ETS, SARIMAX implementations
- `numpy` - Array operations
- `cashflow.utils` - WMAPE calculation, train/test split
