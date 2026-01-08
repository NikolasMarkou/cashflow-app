# Outliers

Anomaly detection and treatment for residual cash flow series.

**SDD Reference:** Section 11

**Key Principle:** Outliers are detected and treated only on the residual component, never on the deterministic base.

## Files

| File | Description |
|------|-------------|
| `detector.py` | Four outlier detection methods |
| `treatment.py` | Treatment strategies with dual-value audit model |

## Detection Methods

### `detector.py`

```python
def detect_outliers(
    series: pd.Series,
    method: str = "modified_zscore",
    threshold: float = 3.5,
) -> tuple[pd.Series, pd.Series]:
    """
    Detect outliers using specified method.

    Returns:
        (is_outlier: bool Series, scores: float Series)
    """
```

| Method | Function | Description | Recommended Threshold |
|--------|----------|-------------|----------------------|
| `modified_zscore` | `modified_zscore()` | MAD-based, robust to outliers | 3.5 |
| `zscore` | `zscore_outliers()` | Standard deviation based | 3.0 |
| `iqr` | `iqr_outliers()` | Interquartile range | 1.5 |
| `isolation_forest` | `isolation_forest_outliers()` | Tree-based anomaly scoring | 0.1 (contamination) |

### Modified Z-Score (Recommended)

```python
def modified_zscore(series: pd.Series, threshold: float = 3.5) -> tuple[pd.Series, pd.Series]:
    """
    MAD-based detection:
    M_i = 0.6745 * (x_i - median) / MAD

    Outlier if |M_i| > threshold
    """
```

## Treatment Methods

### `treatment.py`

```python
def treat_outliers(
    df: pd.DataFrame,
    value_col: str,
    outlier_mask: pd.Series,
    method: str = "median",
) -> pd.DataFrame:
    """
    Replace outliers while preserving originals for audit.

    Creates:
        - {value_col}_original: Pre-treatment values
        - {value_col}_clean: Post-treatment values
        - treatment_tag: ABNORMAL_EXTERNAL_FLOW
    """
```

| Method | Function | Description |
|--------|----------|-------------|
| `median` | `_median_treatment()` | Replace with overall non-outlier median |
| `rolling_median` | `_rolling_median_treatment()` | Window-based median (preserves trends) |
| `capped` | `_capped_treatment()` | Clip to percentile bounds (5th/95th) |
| `winsorize` | `_winsorize_treatment()` | Symmetric tail replacement |

### Combined Pipeline

```python
def apply_residual_treatment(
    df: pd.DataFrame,
    detection_method: str = "modified_zscore",
    detection_threshold: float = 3.5,
    treatment_method: str = "median",
) -> pd.DataFrame:
    """
    Full outlier pipeline:
    1. Detect outliers on 'residual' column
    2. Apply treatment
    3. Add audit columns (is_outlier, outlier_score, treatment_tag)
    """
```

## Dual-Value Audit Model

Per SDD Section 11.4, both original and treated values are preserved:

```python
# After treatment, DataFrame contains:
df["residual"]          # Alias for residual_clean
df["residual_original"] # Pre-treatment value (for audit)
df["residual_clean"]    # Post-treatment value (for modeling)
df["is_outlier"]        # Boolean flag
df["outlier_score"]     # Detection score (e.g., M_i value)
df["treatment_tag"]     # "ABNORMAL_EXTERNAL_FLOW"
```

## Usage

```python
from cashflow.outliers import detect_outliers, apply_residual_treatment

# Detect outliers
is_outlier, scores = detect_outliers(
    series=df["residual"],
    method="modified_zscore",
    threshold=3.5,
)

# Full pipeline with treatment
treated_df = apply_residual_treatment(
    df=decomposed_df,
    detection_method="modified_zscore",
    detection_threshold=3.5,
    treatment_method="median",
)

# Access results
outlier_count = treated_df["is_outlier"].sum()
outlier_months = treated_df[treated_df["is_outlier"]]["month_key"].tolist()
```

## Dependencies

- `numpy`, `scipy` - Statistical computations
- `sklearn.ensemble.IsolationForest` - Tree-based detection
- `cashflow.schemas.necf` - DecomposedNECF model
