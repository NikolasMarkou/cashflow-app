# Features

Feature engineering for time series models and explainability.

**SDD Reference:** Section 12

## Files

| File | Description |
|------|-------------|
| `time.py` | Temporal features (lags, rolling stats, cyclical encoding) |
| `exogenous.py` | External variables from CRF and calendar events |

## Temporal Features

### `time.py`

```python
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features:
    - month: 1-12
    - quarter: 1-4
    - year: YYYY
    - month_sin, month_cos: Cyclical encoding
    - is_quarter_start, is_quarter_end
    - is_year_boundary
    """

def add_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: list[int] = [1, 2, 12],
) -> pd.DataFrame:
    """
    Add lagged values:
    - {column}_lag_1: Previous month
    - {column}_lag_2: Two months ago
    - {column}_lag_12: Same month last year
    """

def add_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: list[int] = [3, 6, 12],
) -> pd.DataFrame:
    """
    Add rolling statistics:
    - {column}_roll_mean_3: 3-month moving average
    - {column}_roll_std_3: 3-month rolling std
    - {column}_roll_mean_6, {column}_roll_std_6
    - {column}_roll_mean_12, {column}_roll_std_12
    """
```

### Cyclical Encoding

Month-of-year is encoded as sin/cos to capture seasonal periodicity:

```python
# January (month=1) and December (month=12) are close in cyclical space
month_sin = sin(2 * pi * month / 12)
month_cos = cos(2 * pi * month / 12)
```

## Exogenous Features

### `exogenous.py`

```python
def build_exogenous_matrix(
    month_keys: list[str],
    crf_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build feature matrix for SARIMAX:
    - known_future_delta: Contract end events from CRF
    - Initialized to zeros, merged with CRF events
    """

def create_holiday_indicators(month_keys: list[str]) -> pd.DataFrame:
    """
    Calendar-based indicators:
    - is_december: Holiday spending month
    - is_summer: July-August (vacation period)
    - is_back_to_school: September
    """

def create_step_function(
    month_keys: list[str],
    event_month: str,
    direction: str = "up",  # "up" or "down"
) -> pd.Series:
    """
    Binary step indicator for contract events:
    - 0 before event_month, 1 after (direction="up")
    - 1 before event_month, 0 after (direction="down")
    """
```

### Known Future Delta

From CRF contracts with `recurrence_end_date`:

```python
# Loan ending in 2024-06 with monthly payment of 500
# Creates delta of +500 starting 2024-07 (payment stops)

delta_df = compute_known_future_delta(crf_df, "2024-01", "2024-12")
# month_key | delta_value
# 2024-07   | 500.0
```

## Usage

```python
from cashflow.features import add_time_features, add_lag_features, build_exogenous_matrix

# Add temporal features
df = add_time_features(df)
df = add_lag_features(df, column="residual_clean", lags=[1, 2, 12])
df = add_rolling_features(df, column="residual_clean", windows=[3, 6, 12])

# Build exogenous matrix for SARIMAX
exog = build_exogenous_matrix(
    month_keys=["2024-01", "2024-02", ..., "2024-12"],
    crf_df=crf_df,
)

# Use with SARIMAX
from cashflow.models import SARIMAXModel
model = SARIMAXModel()
model.fit(series, exog=exog)
forecast = model.predict(steps=12, exog_future=future_exog)
```

## Feature Summary

| Feature Type | Features | Use Case |
|--------------|----------|----------|
| Calendar | month, quarter, year | Seasonality |
| Cyclical | month_sin, month_cos | Smooth seasonal patterns |
| Lags | lag_1, lag_2, lag_12 | Autoregressive input |
| Rolling | roll_mean_*, roll_std_* | Trend and volatility |
| Events | known_future_delta | Contract terminations |
| Holidays | is_december, is_summer | Spending patterns |

## Dependencies

- `numpy` - Mathematical operations
- `pandas` - DataFrame manipulation
- `cashflow.pipeline.decomposition` - `compute_known_future_delta`
