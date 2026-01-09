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

---

## Model Formulations

### ETS (Error-Trend-Seasonality) / Exponential Smoothing

**Introduction:**
ETS models decompose a time series into error, trend, and seasonal components. They are well-suited for data with clear trends and seasonality, offering interpretable parameters and reliable forecasts. ETS is the simplest model in our hierarchy, making it preferred when accuracy is comparable to more complex alternatives.

**Formulation (Additive Trend, Additive Seasonality - Holt-Winters):**

The model maintains three state equations updated at each time step:

```
Level:      l_t = α(y_t - s_{t-m}) + (1 - α)(l_{t-1} + b_{t-1})
Trend:      b_t = β(l_t - l_{t-1}) + (1 - β)b_{t-1}
Seasonal:   s_t = γ(y_t - l_{t-1} - b_{t-1}) + (1 - γ)s_{t-m}
```

**Forecast equation:**
```
ŷ_{t+h} = l_t + h·b_t + s_{t+h-m}
```

Where:
- `y_t` = observed value at time t
- `l_t` = level (smoothed mean)
- `b_t` = trend (slope)
- `s_t` = seasonal component
- `m` = seasonal period (12 for monthly data)
- `α` = level smoothing parameter (0 < α < 1)
- `β` = trend smoothing parameter (0 < β < 1)
- `γ` = seasonal smoothing parameter (0 < γ < 1)
- `h` = forecast horizon

**When to use:** Short-to-medium term forecasting with stable patterns, interpretable components needed, or as a baseline model.

---

### SARIMA (Seasonal AutoRegressive Integrated Moving Average)

**Introduction:**
SARIMA extends ARIMA to handle seasonality by incorporating seasonal differencing and seasonal AR/MA terms. It models both short-term autocorrelations and seasonal patterns, making it effective for data with complex temporal dependencies.

**Model specification:** SARIMA(p, d, q)(P, D, Q, m)

**Formulation:**

The general SARIMA model combines non-seasonal and seasonal components:

```
φ(B)Φ(B^m)(1 - B)^d(1 - B^m)^D · y_t = θ(B)Θ(B^m) · ε_t
```

Where:
- `B` = backshift operator (B·y_t = y_{t-1})
- `φ(B) = 1 - φ₁B - φ₂B² - ... - φ_pB^p` (AR polynomial)
- `Φ(B^m) = 1 - Φ₁B^m - Φ₂B^{2m} - ... - Φ_PB^{Pm}` (seasonal AR polynomial)
- `θ(B) = 1 + θ₁B + θ₂B² + ... + θ_qB^q` (MA polynomial)
- `Θ(B^m) = 1 + Θ₁B^m + Θ₂B^{2m} + ... + Θ_QB^{Qm}` (seasonal MA polynomial)
- `(1 - B)^d` = non-seasonal differencing of order d
- `(1 - B^m)^D` = seasonal differencing of order D
- `ε_t` = white noise error term ~ N(0, σ²)
- `m` = seasonal period (12 for monthly)

**Parameter meanings:**
| Parameter | Description |
|-----------|-------------|
| p | Non-seasonal AR order (past values) |
| d | Non-seasonal differencing order |
| q | Non-seasonal MA order (past errors) |
| P | Seasonal AR order |
| D | Seasonal differencing order |
| Q | Seasonal MA order |
| m | Seasonal period |

**Default configuration:** SARIMA(1,1,1)(1,1,0,12)
- AR(1): Current value depends on previous value
- I(1): First differencing for stationarity
- MA(1): Current error depends on previous error
- Seasonal AR(1): Annual pattern from same month last year
- Seasonal I(1): Seasonal differencing
- Period 12: Monthly seasonality

**When to use:** Data with strong autocorrelation, complex seasonal patterns, or when ETS underperforms.

---

### SARIMAX (SARIMA with eXogenous variables)

**Introduction:**
SARIMAX extends SARIMA by incorporating external (exogenous) regressors that may influence the target variable. In cash flow forecasting, these include known future events like loan terminations or contract changes from the CRF (Counterparty Reference Feed).

**Formulation:**

```
φ(B)Φ(B^m)(1 - B)^d(1 - B^m)^D · y_t = β'X_t + θ(B)Θ(B^m) · ε_t
```

Where:
- All SARIMA terms as defined above
- `X_t` = vector of exogenous variables at time t
- `β` = vector of regression coefficients

**Exogenous variables used:**
| Variable | Description | Source |
|----------|-------------|--------|
| `known_future_delta` | Contract termination impacts | CRF recurrence_end_date |
| `is_december` | Holiday spending indicator | Calendar |
| `is_summer` | Vacation period (Jul-Aug) | Calendar |

**Example:**
```
# Loan ending in June 2024 with €500 monthly payment
# Creates +500 delta starting July 2024

X_t = [known_future_delta_t, is_december_t, is_summer_t]
```

**When to use:** When external factors significantly impact cash flows, known future events exist (loan maturity, subscription end), or when pure time series models show systematic bias.

---

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
