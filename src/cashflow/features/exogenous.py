"""Exogenous feature engineering - SDD Section 12.3."""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


def build_exogenous_matrix(
    month_keys: list[str],
    known_deltas: Optional[pd.DataFrame] = None,
    external_events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build exogenous variable matrix for SARIMAX.

    Per SDD Section 12.3, the KnownFutureFlow_Delta vector represents
    deterministic future changes from CRF events.

    Args:
        month_keys: List of month keys (YYYY-MM)
        known_deltas: DataFrame with month_key and delta_value from CRF
        external_events: Optional external event indicators

    Returns:
        DataFrame with exogenous variables indexed by month_key
    """
    # Initialize with zeros
    exog = pd.DataFrame({"month_key": month_keys})
    exog["known_future_delta"] = 0.0

    # Add known deltas from CRF
    if known_deltas is not None and len(known_deltas) > 0:
        delta_map = known_deltas.set_index("month_key")["delta_value"].to_dict()
        exog["known_future_delta"] = exog["month_key"].map(delta_map).fillna(0.0)

    # Add external event indicators if provided
    if external_events is not None:
        exog = exog.merge(external_events, on="month_key", how="left")
        # Fill NaN with 0 for event indicators
        for col in external_events.columns:
            if col != "month_key":
                exog[col] = exog[col].fillna(0)

    exog = exog.set_index("month_key")

    return exog


def create_holiday_indicators(month_keys: list[str]) -> pd.DataFrame:
    """Create holiday month indicators.

    Marks months with major holidays that may affect spending patterns.

    Args:
        month_keys: List of month keys

    Returns:
        DataFrame with holiday indicators
    """
    df = pd.DataFrame({"month_key": month_keys})
    month_nums = pd.to_datetime(df["month_key"]).dt.month

    # December (Christmas/New Year spending)
    df["is_holiday_month"] = month_nums == 12

    # Summer vacation months
    df["is_summer"] = month_nums.isin([7, 8])

    # Back to school (September)
    df["is_back_to_school"] = month_nums == 9

    return df


def create_step_function(
    month_keys: list[str],
    event_month: str,
    pre_value: float = 0.0,
    post_value: float = 1.0,
) -> pd.Series:
    """Create a step function indicator for contract changes.

    Useful for modeling the effect of a contract ending (e.g., loan payoff).

    Args:
        month_keys: List of month keys
        event_month: Month when change occurs
        pre_value: Value before event
        post_value: Value from event month onwards

    Returns:
        Series with step function values
    """
    dates = pd.to_datetime(month_keys)
    event_date = pd.to_datetime(event_month)

    values = np.where(dates >= event_date, post_value, pre_value)

    return pd.Series(values, index=month_keys, name=f"step_{event_month}")
