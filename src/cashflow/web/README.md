# Web Interface

FastAPI-based web interface for interactive cash flow forecasting with Plotly.js visualization.

**SDD Reference:** User Interface Layer

## Files

| File | Description |
|------|-------------|
| `app.py` | FastAPI application factory and configuration |
| `routes/pages.py` | HTML page routes (main UI) |
| `routes/forecast.py` | REST API endpoints |
| `schemas/response.py` | API response models |
| `templates/` | Jinja2 HTML templates |
| `static/` | CSS and JavaScript assets |

## Quick Start

```bash
# Start the server
cashflow-web

# Or with uvicorn directly
uvicorn cashflow.web.app:app --host 0.0.0.0 --port 8000 --reload
```

Access at **http://localhost:8000**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Browser                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ HTML Form   │  │ Plotly.js   │  │ Metrics Dashboard   │  │
│  └──────┬──────┘  └──────▲──────┘  └──────────▲──────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────┘
          │                │                     │
          ▼                │                     │
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ POST        │  │ Response    │  │ ForecastEngine      │  │
│  │ /api/       │─▶│ Transform   │─▶│ Pipeline            │  │
│  │ forecast    │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

### `GET /` - Main Page

Renders the forecast interface with:
- CSV file upload form
- Configuration options
- Chart containers
- Metrics dashboard

### `POST /api/forecast` - Run Forecast

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | UTF CSV upload |
| `forecast_horizon` | int | Months to forecast (1-24) |
| `wmape_threshold` | float | Maximum acceptable WMAPE |
| `outlier_method` | string | Detection method |
| `outlier_threshold` | float | Detection threshold |
| `outlier_treatment` | string | Treatment method |
| `models` | list | Models to evaluate |
| `confidence_level` | float | CI level (0.90, 0.95, 0.99) |

**Response:** `ForecastAPIResponse`

```json
{
  "success": true,
  "metrics": {
    "model_selected": "ETS",
    "wmape": 1.818,
    "meets_threshold": true,
    "confidence_level": "High",
    "forecast_horizon": 12
  },
  "chart_data": {
    "historical": { "months": [...], "necf": [...] },
    "forecast": { "months": [...], "totals": [...], "lower_ci": [...], "upper_ci": [...] },
    "components": { "months": [...], "deterministic_base": [...], "residual": [...] },
    "model_comparison": [...]
  },
  "statistics": {
    "decomposition": {...},
    "transfer_netting": {...}
  },
  "outliers": [...]
}
```

## Configuration Options

| Option | Values | Default |
|--------|--------|---------|
| Forecast Horizon | 1-24 months | 12 |
| WMAPE Threshold | 1-50% | 20.0 |
| Outlier Method | modified_zscore, zscore, iqr, isolation_forest | modified_zscore |
| Outlier Threshold | 1.0-5.0 | 3.5 |
| Outlier Treatment | median, rolling_median, capped | median |
| Models | ETS, SARIMA, SARIMAX | ETS, SARIMA |
| Confidence Level | 90%, 95%, 99% | 95% |

## Charts (Plotly.js)

### 1. Time Series Chart
- Historical NECF (steel blue line)
- Forecast projection (green line)
- 95% confidence interval (green band)
- Vertical line marking forecast start
- Outlier markers (red circles)

### 2. Model Comparison
- Bar chart of WMAPE by model
- Winner highlighted in green
- Threshold line shown in red

### 3. Component Breakdown
- Stacked bar chart
- Deterministic base (cyan)
- Residual positive (amber)
- Residual negative (light red)
- Forecast total line overlay

### 4. Outlier Analysis
- Grouped bar chart
- Original vs treated values
- Only shown when outliers detected

## Templates

### `base.html`
Base layout with:
- Plotly.js CDN include
- Common CSS
- Footer

### `index.html`
Main interface with:
- Configuration form
- Chart containers
- Metrics cards
- Statistics panel

## Static Assets

### CSS (`static/css/style.css`)
- Form styling
- Chart container layouts
- Metrics dashboard
- Responsive design

### JavaScript

**`static/js/charts.js`**
- `renderTimeSeriesChart()` - Historical + forecast plot
- `renderModelComparisonChart()` - WMAPE bar chart
- `renderComponentsChart()` - Stacked decomposition
- `renderOutlierChart()` - Original vs treated

**`static/js/form.js`**
- Form submission handler
- Response processing
- Chart rendering orchestration
- Error display

## Response Transformation

The `ForecastAPIResponse.from_payload()` method transforms `ExplainabilityPayload` into chart-ready format:

```python
@classmethod
def from_payload(cls, payload: ExplainabilityPayload, historical_df: pd.DataFrame):
    """
    Transform engine output to API response:
    - Extract historical months and NECF values
    - Map forecast results to chart arrays
    - Build component breakdown data
    - Format model candidates for comparison
    """
```

## Usage

### Programmatic Access

```python
import requests

# Upload CSV and run forecast
with open("data/transactions.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/forecast",
        files={"file": f},
        data={
            "forecast_horizon": 12,
            "wmape_threshold": 20.0,
            "models": ["ets", "sarima"],
        },
    )

result = response.json()
print(f"Model: {result['metrics']['model_selected']}")
print(f"WMAPE: {result['metrics']['wmape']:.2f}%")
```

### Embedding Charts

```javascript
// After receiving API response
const data = await response.json();

renderTimeSeriesChart('chart-timeseries', data.chart_data);
renderModelComparisonChart('chart-models', data.chart_data.model_comparison, 20.0);
renderComponentsChart('chart-components', data.chart_data.components);
renderOutlierChart('chart-outliers', data.outliers);
```

## Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `jinja2` - Template engine
- `python-multipart` - File upload handling
- `cashflow.engine` - ForecastEngine
- `cashflow.schemas.forecast` - ExplainabilityPayload
