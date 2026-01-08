"""Explainability JSON builder - SDD Section 15.3."""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from cashflow.schemas.forecast import ExplainabilityPayload


def build_explainability_json(payload: ExplainabilityPayload) -> dict:
    """Convert ExplainabilityPayload to JSON-serializable dict.

    Per SDD Section 15.3, this produces the authoritative explainability
    payload for LLM consumption.

    Args:
        payload: ExplainabilityPayload dataclass

    Returns:
        JSON-serializable dictionary
    """
    return payload.model_dump(mode="json")


def save_explainability_json(
    payload: ExplainabilityPayload,
    output_path: str | Path,
    indent: int = 2,
) -> None:
    """Save explainability payload to JSON file.

    Args:
        payload: ExplainabilityPayload to save
        output_path: Path for output JSON file
        indent: JSON indentation (default 2)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_explainability_json(payload)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def format_llm_facts(payload: ExplainabilityPayload) -> list[str]:
    """Format payload as LLM-ready structured facts.

    Per SDD Section 15.4, produces grounded facts for LLM consumption.
    LLM must use only provided fields and never invent data.

    Args:
        payload: ExplainabilityPayload

    Returns:
        List of structured fact strings
    """
    facts = []

    # Model selection facts
    facts.append(
        f"The {payload.model_selected} model was selected with WMAPE of {payload.wmape_winner:.2f}%."
    )

    if payload.meets_threshold:
        facts.append(
            f"The forecast meets the accuracy threshold of {payload.wmape_threshold}%."
        )
    else:
        facts.append(
            f"Warning: The forecast WMAPE ({payload.wmape_winner:.2f}%) exceeds "
            f"the threshold ({payload.wmape_threshold}%)."
        )

    # Forecast period facts
    facts.append(
        f"The forecast covers {payload.horizon_months} months from "
        f"{payload.forecast_start} to {payload.forecast_end}."
    )

    facts.append(f"Confidence level: {payload.confidence_level}")

    # Decomposition facts
    decomp = payload.decomposition_summary
    facts.append(
        f"Average historical NECF: {decomp.avg_necf:.2f}, "
        f"consisting of {decomp.avg_deterministic_base:.2f} deterministic base "
        f"and {decomp.avg_residual:.2f} residual."
    )

    # Transfer netting facts
    netting = payload.transfer_netting_summary
    if netting.num_transfers_removed > 0:
        facts.append(
            f"{netting.num_transfers_removed} internal transfers were removed, "
            f"totaling {netting.total_volume_removed:.2f} in volume."
        )

    # Outlier facts
    if payload.outliers_detected:
        facts.append(
            f"{len(payload.outliers_detected)} months were flagged as outliers and treated."
        )
        for outlier in payload.outliers_detected:
            facts.append(
                f"  - {outlier.month_key}: Original {outlier.original_value:.2f} "
                f"treated to {outlier.treated_value:.2f}"
            )

    # Exogenous event facts
    if payload.exogenous_events:
        for event in payload.exogenous_events:
            facts.append(
                f"Known future event: {event.get('counterparty_display_name')} "
                f"ending in {event.get('end_month_key')} with delta {event.get('delta_value'):.2f}"
            )

    return facts


def generate_forecast_summary(payload: ExplainabilityPayload) -> dict:
    """Generate a concise summary for quick review.

    Args:
        payload: ExplainabilityPayload

    Returns:
        Summary dictionary
    """
    forecast_values = [r.forecast_total for r in payload.forecast_results]

    return {
        "model": payload.model_selected,
        "accuracy_wmape": payload.wmape_winner,
        "meets_threshold": payload.meets_threshold,
        "confidence": payload.confidence_level,
        "forecast_range": {
            "start": payload.forecast_start,
            "end": payload.forecast_end,
            "months": payload.horizon_months,
        },
        "forecast_stats": {
            "min": min(forecast_values),
            "max": max(forecast_values),
            "mean": sum(forecast_values) / len(forecast_values),
        },
        "outliers_count": len(payload.outliers_detected),
        "transfers_removed": payload.transfer_netting_summary.num_transfers_removed,
    }
