# Explainability

LLM-ready output generation for forecast results.

**SDD Reference:** Section 15

## Files

| File | Description |
|------|-------------|
| `builder.py` | JSON payload builder and narrative generation |

## Key Functions

### `builder.py`

```python
def build_explainability_json(payload: ExplainabilityPayload) -> dict:
    """
    Convert ExplainabilityPayload to JSON-serializable dict.

    Handles:
    - Datetime serialization
    - Pydantic model conversion
    - Nested object flattening
    """

def save_explainability_json(
    payload: ExplainabilityPayload,
    path: str | Path,
    indent: int = 2,
) -> None:
    """Write payload to JSON file with formatting."""

def format_llm_facts(payload: ExplainabilityPayload) -> list[str]:
    """
    Generate grounded facts for LLM consumption.

    Returns list of factual statements:
    - "Model selected: ETS with WMAPE 5.2%"
    - "3 outliers detected and treated"
    - "24 internal transfers removed"
    """

def generate_forecast_summary(payload: ExplainabilityPayload) -> dict:
    """
    Quick-view summary:
    - model: Selected model name
    - accuracy: WMAPE percentage
    - confidence: High/Medium/Low
    - forecast_range: (min, max) of forecast values
    - outlier_count: Number detected
    - transfer_count: Number netted
    """
```

## Output Structure

### ExplainabilityPayload JSON

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "sdd_version": "v0.05",

  "model_selected": "ETS",
  "model_candidates": [
    {"model_name": "ETS", "wmape": 5.2, "is_winner": true},
    {"model_name": "SARIMA", "wmape": 6.1, "is_winner": false}
  ],
  "wmape_winner": 5.2,
  "wmape_threshold": 20.0,
  "meets_threshold": true,

  "forecast_start": "2024-01",
  "forecast_end": "2024-12",
  "horizon_months": 12,
  "confidence_level": "High",

  "decomposition_summary": {
    "avg_necf": 1500.00,
    "avg_deterministic_base": 2000.00,
    "avg_residual": -500.00
  },

  "transfer_netting_summary": {
    "num_transfers_removed": 24,
    "total_volume_removed": 12000.00
  },

  "outliers_detected": [
    {
      "month_key": "2023-07",
      "original_value": -5000.00,
      "treated_value": -800.00,
      "detection_method": "modified_zscore",
      "score": 4.2,
      "treatment_tag": "ABNORMAL_EXTERNAL_FLOW"
    }
  ],

  "forecast_results": [
    {
      "month_key": "2024-01",
      "forecast_total": 1450.00,
      "forecast_residual": -550.00,
      "deterministic_base": 2000.00,
      "known_future_delta": 0.00,
      "lower_ci": 1200.00,
      "upper_ci": 1700.00
    }
  ],

  "exogenous_events": []
}
```

## LLM Integration

### Grounded Facts

```python
facts = format_llm_facts(payload)
# [
#   "Model selected: ETS",
#   "Model accuracy (WMAPE): 5.2%",
#   "Accuracy threshold met: Yes",
#   "Confidence level: High",
#   "Forecast period: 2024-01 to 2024-12 (12 months)",
#   "Average NECF: 1500.00",
#   "Deterministic base: 2000.00",
#   "Average residual: -500.00",
#   "Internal transfers removed: 24",
#   "Transfer volume netted: 12000.00",
#   "Outliers detected: 1",
#   "Outlier treatment: median replacement"
# ]
```

### Usage with LLM

```python
from cashflow.explainability import format_llm_facts, save_explainability_json

# Save full JSON for reference
save_explainability_json(payload, "output/forecast.json")

# Get facts for LLM prompt
facts = format_llm_facts(payload)
prompt = f"""
Based on these forecast facts, provide a summary:

{chr(10).join(f'- {fact}' for fact in facts)}

Key constraint: Only use the provided facts. Do not invent additional data.
"""
```

## Confidence Level

Determined by `cashflow.utils.determine_confidence_level()`:

| Level | Criteria |
|-------|----------|
| High | WMAPE < 10%, 24+ months history, quality > 95% |
| Medium | WMAPE < 20%, 12+ months history |
| Low | Otherwise |

## Usage

```python
from cashflow.explainability import (
    build_explainability_json,
    save_explainability_json,
    format_llm_facts,
    generate_forecast_summary,
)

# From ForecastEngine result
result = engine.run("data/transactions.csv")

# Save to file
save_explainability_json(result, "output/forecast_summary.json")

# Get JSON dict
json_data = build_explainability_json(result)

# Quick summary
summary = generate_forecast_summary(result)
print(f"Model: {summary['model']}, WMAPE: {summary['accuracy']:.1f}%")

# LLM facts
facts = format_llm_facts(result)
for fact in facts:
    print(f"- {fact}")
```

## Dependencies

- `json` - Serialization
- `pathlib` - File operations
- `cashflow.schemas.forecast` - ExplainabilityPayload
