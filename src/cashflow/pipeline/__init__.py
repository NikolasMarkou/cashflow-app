"""Data transformation pipeline - ingestion through decomposition."""

from __future__ import annotations
from cashflow.pipeline.ingestion import load_utf, load_crf, validate_utf
from cashflow.pipeline.cleaning import clean_utf
from cashflow.pipeline.enrichment import enrich_with_crf
from cashflow.pipeline.transfer import detect_transfers, net_transfers
from cashflow.pipeline.aggregation import aggregate_monthly
from cashflow.pipeline.decomposition import (
    decompose_cashflow,
    compute_deterministic_projection,
    DeterministicProjection,
)
from cashflow.pipeline.recurrence import (
    discover_recurring_patterns,
    apply_discovered_recurrence,
    get_recurrence_summary,
)

__all__ = [
    "load_utf",
    "load_crf",
    "validate_utf",
    "clean_utf",
    "enrich_with_crf",
    "detect_transfers",
    "net_transfers",
    "aggregate_monthly",
    "decompose_cashflow",
    "compute_deterministic_projection",
    "DeterministicProjection",
    "discover_recurring_patterns",
    "apply_discovered_recurrence",
    "get_recurrence_summary",
]
