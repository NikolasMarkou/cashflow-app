"""Data transformation pipeline - ingestion through decomposition."""

from __future__ import annotations
from cashflow.pipeline.ingestion import load_utf, load_crf, validate_utf
from cashflow.pipeline.cleaning import clean_utf
from cashflow.pipeline.enrichment import enrich_with_crf
from cashflow.pipeline.transfer import detect_transfers, net_transfers
from cashflow.pipeline.aggregation import aggregate_monthly
from cashflow.pipeline.decomposition import decompose_cashflow

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
]
