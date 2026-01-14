"""TiRex ONNX model - LSTM-based time series forecasting.

TiRex (Time series Rex) is a pre-trained neural network model for
univariate time series forecasting with uncertainty quantification.

The model uses Reversible Instance Normalization (RevIN) internally:
- No manual normalization needed - model handles it automatically
- Output is already in original scale - no denormalization needed

Quantile outputs: [0.1, 0.25, 0.5, 0.75, 0.9]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from cashflow.models.base import ForecastModel, ForecastOutput, generate_future_month_keys


# Default model path relative to project root
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent.parent / "models" / "20260113_tirex_model.onnx"


class TiRexModel(ForecastModel):
    """TiRex ONNX model - LSTM-based time series forecasting with quantile outputs.

    This model uses a pre-trained ONNX neural network (TiRex architecture)
    for time series forecasting. It requires exactly 24 months of historical
    data and produces 12-month forecasts with 5 quantile outputs.

    The model uses Reversible Instance Normalization (RevIN) internally:
    - No manual normalization needed - pass raw data directly
    - Output is already in original scale - no denormalization needed
    - Quantile levels: [0.1, 0.25, 0.5, 0.75, 0.9]

    Output channels:
        - Channel 0: 0.10 quantile (10th percentile)
        - Channel 1: 0.25 quantile (25th percentile)
        - Channel 2: 0.50 quantile (median/point forecast)
        - Channel 3: 0.75 quantile (75th percentile)
        - Channel 4: 0.90 quantile (90th percentile)

    Attributes:
        name: Model identifier string
        complexity_score: Relative complexity (4 = neural network, most complex)
    """

    name: str = "TiRex"
    complexity_score: int = 4  # Most complex (neural network)

    # Model expects 24 months input, produces 12 months output
    INPUT_MONTHS: int = 24
    OUTPUT_MONTHS: int = 12

    # Output quantile channels (matches training: [0.1, 0.25, 0.5, 0.75, 0.9])
    QUANTILE_10: int = 0   # 0.10 quantile
    QUANTILE_25: int = 1   # 0.25 quantile
    QUANTILE_50: int = 2   # 0.50 quantile (median/point forecast)
    QUANTILE_75: int = 3   # 0.75 quantile
    QUANTILE_90: int = 4   # 0.90 quantile

    # Quantile levels for reference
    QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]

    def __init__(self, model_path: Optional[str] = None):
        """Initialize TiRex model with ONNX runtime session.

        Args:
            model_path: Path to ONNX model file. If None, uses default path.

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If ONNX runtime fails to load model
        """
        import onnxruntime as ort

        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not self._model_path.exists():
            raise FileNotFoundError(f"TiRex model not found: {self._model_path}")

        try:
            self._session = ort.InferenceSession(str(self._model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load TiRex ONNX model: {e}") from e

        # Cache input/output metadata
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._attention_inputs = [
            inp.name for inp in self._session.get_inputs()
            if "attention" in inp.name and "Reshape" in inp.name
        ]

        # Will be populated during fit()
        self._input_data: Optional[np.ndarray] = None
        self._last_date: Optional[pd.Timestamp] = None
        self._fitted = False

    def fit(self, series: pd.Series) -> "TiRexModel":
        """Prepare input data from historical series.

        No manual normalization needed - the model uses RevIN internally
        which handles normalization and denormalization automatically.

        If the series has fewer than 24 months, it will be padded at the
        beginning with the mean value to reach 24 months.

        Args:
            series: Time series (will be padded if < 24 months)

        Returns:
            Self for method chaining
        """
        if len(series) >= self.INPUT_MONTHS:
            # Take last 24 months
            input_series = series.iloc[-self.INPUT_MONTHS:]
            raw_values = input_series.values.flatten().astype(np.float32)
        else:
            # Pad with mean value at the beginning
            raw_values = series.values.flatten().astype(np.float32)
            pad_length = self.INPUT_MONTHS - len(raw_values)
            pad_value = np.mean(raw_values)
            raw_values = np.concatenate([
                np.full(pad_length, pad_value, dtype=np.float32),
                raw_values
            ])
            input_series = series

        # Reshape to [batch=1, timesteps=24, features=1]
        # No normalization needed - RevIN handles it internally
        self._input_data = raw_values.reshape(1, self.INPUT_MONTHS, 1)

        # Store last date for month key generation
        if hasattr(input_series.index, "__getitem__"):
            last_idx = input_series.index[-1]
            if hasattr(last_idx, "to_timestamp"):
                self._last_date = last_idx.to_timestamp()
            elif isinstance(last_idx, pd.Timestamp):
                self._last_date = last_idx
            else:
                self._last_date = pd.Timestamp(last_idx)
        else:
            self._last_date = pd.Timestamp.now()

        self._fitted = True
        return self

    def predict(
        self,
        steps: int,
        confidence_level: float = 0.95,
    ) -> ForecastOutput:
        """Generate forecast using ONNX model inference.

        The model uses RevIN internally, so outputs are already in the
        original scale - no denormalization needed.

        Args:
            steps: Number of months to forecast (max 12)
            confidence_level: Confidence level for intervals (default 95%)

        Returns:
            ForecastOutput with predictions and confidence intervals

        Raises:
            RuntimeError: If fit() hasn't been called
            ValueError: If steps > 12
        """
        if not self._fitted or self._input_data is None:
            raise RuntimeError("Model must be fitted before prediction")

        if steps > self.OUTPUT_MONTHS:
            raise ValueError(
                f"TiRex outputs max {self.OUTPUT_MONTHS} months, "
                f"requested {steps}"
            )

        # Prepare input feed
        input_feed = {self._input_name: self._input_data}

        # Add attention masks (identity matrices)
        attention_mask = np.eye(16, dtype=np.int32)
        for name in self._attention_inputs:
            input_feed[name] = attention_mask

        # Run inference
        outputs = self._session.run(None, input_feed)
        result = outputs[0]  # Shape: [1, 12, 5]

        # Extract quantiles - already in original scale (RevIN handles denorm)
        quantiles = result[0]  # Shape: [12, 5]

        # Point forecast is median (channel 2)
        forecast_mean = quantiles[:steps, self.QUANTILE_50]

        # Use model's quantile outputs for confidence intervals
        q10 = quantiles[:steps, self.QUANTILE_10]
        q90 = quantiles[:steps, self.QUANTILE_90]

        if confidence_level == 0.80:
            # Use 0.10 and 0.90 directly (80% CI)
            forecast_lower = q10
            forecast_upper = q90
        else:
            # Extrapolate for other confidence levels
            # Estimate std from interquantile range (0.10 to 0.90 = 2.56 sigma)
            sigma_est = (q90 - q10) / 2.56

            z = stats.norm.ppf((1 + confidence_level) / 2)
            forecast_lower = forecast_mean - z * sigma_est
            forecast_upper = forecast_mean + z * sigma_est

        # Generate month keys
        month_keys = generate_future_month_keys(self._last_date, steps)

        return ForecastOutput(
            model_name=self.name,
            forecast_mean=forecast_mean,
            forecast_lower=forecast_lower,
            forecast_upper=forecast_upper,
            month_keys=month_keys,
            params=self.get_params(),
        )

    def get_params(self) -> dict:
        """Get model parameters and metadata.

        Returns:
            Dict with model path, architecture info, and status
        """
        return {
            "model_path": str(self._model_path),
            "architecture": "TiRex (LSTM + Transformer + RevIN)",
            "normalization": "RevIN (automatic, no manual pre/post-processing)",
            "input_months": self.INPUT_MONTHS,
            "output_months": self.OUTPUT_MONTHS,
            "quantile_levels": self.QUANTILE_LEVELS,
            "fitted": self._fitted,
        }
