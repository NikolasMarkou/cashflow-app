"""Unit tests for transfer detection and netting."""

import pandas as pd
import pytest
from datetime import date

from cashflow.pipeline.transfer import detect_transfers, net_transfers


class TestTransferDetection:
    """Tests for internal transfer detection."""

    def test_detects_explicit_transfer_link(self):
        """Should detect transfers with explicit TransferLinkID."""
        df = pd.DataFrame([
            {
                "tx_id": "T1",
                "customer_id": "C1",
                "account_id": "A1",
                "tx_date": date(2024, 1, 15),
                "amount": -500.0,
                "transfer_link_id": "LINK001",
                "category": "TRANSFER_OUT",
            },
            {
                "tx_id": "T2",
                "customer_id": "C1",
                "account_id": "A2",
                "tx_date": date(2024, 1, 15),
                "amount": 500.0,
                "transfer_link_id": "LINK001",
                "category": "TRANSFER_IN",
            },
        ])

        result = detect_transfers(df)

        assert result["is_internal_transfer"].all()
        assert result["transfer_detection_method"].iloc[0] == "transfer_link_id"

    def test_detects_by_amount_and_date(self):
        """Should detect transfers by matching amount and date."""
        df = pd.DataFrame([
            {
                "tx_id": "T1",
                "customer_id": "C1",
                "account_id": "A1",
                "tx_date": date(2024, 1, 15),
                "amount": -500.0,
                "category": "TRANSFER",
            },
            {
                "tx_id": "T2",
                "customer_id": "C1",
                "account_id": "A2",
                "tx_date": date(2024, 1, 16),  # Next day
                "amount": 500.0,
                "category": "TRANSFER",
            },
        ])

        result = detect_transfers(df, date_tolerance_days=2)

        assert result["is_internal_transfer"].all()

    def test_respects_date_tolerance(self):
        """Should not match transfers beyond date tolerance."""
        df = pd.DataFrame([
            {
                "tx_id": "T1",
                "customer_id": "C1",
                "account_id": "A1",
                "tx_date": date(2024, 1, 1),
                "amount": -500.0,
                "category": "TRANSFER",
            },
            {
                "tx_id": "T2",
                "customer_id": "C1",
                "account_id": "A2",
                "tx_date": date(2024, 1, 10),  # 9 days later
                "amount": 500.0,
                "category": "TRANSFER",
            },
        ])

        result = detect_transfers(df, date_tolerance_days=2)

        # Should not match due to date difference
        assert not result["is_internal_transfer"].iloc[0]
        assert not result["is_internal_transfer"].iloc[1]

    def test_detects_by_category_heuristic(self):
        """Should detect transfers by category patterns."""
        df = pd.DataFrame([
            {
                "tx_id": "T1",
                "customer_id": "C1",
                "account_id": "A1",
                "tx_date": date(2024, 1, 15),
                "amount": -500.0,
                "category": "SAVINGS_CONTRIBUTION",
            },
        ])

        result = detect_transfers(df)

        assert result["is_internal_transfer"].iloc[0]
        assert result["transfer_detection_method"].iloc[0] == "category_heuristic"


class TestTransferNetting:
    """Tests for transfer netting."""

    def test_removes_internal_transfers(self):
        """Should remove internal transfers from data."""
        df = pd.DataFrame([
            {"tx_id": "T1", "amount": 100.0, "is_internal_transfer": False},
            {"tx_id": "T2", "amount": -500.0, "is_internal_transfer": True},
            {"tx_id": "T3", "amount": 500.0, "is_internal_transfer": True},
            {"tx_id": "T4", "amount": -50.0, "is_internal_transfer": False},
        ])

        external_df, summary = net_transfers(df)

        assert len(external_df) == 2
        assert summary["num_transfers_removed"] == 2
        assert summary["total_volume_removed"] == 1000.0

    def test_preserves_external_transactions(self):
        """Should preserve all external transactions."""
        df = pd.DataFrame([
            {"tx_id": "T1", "amount": 100.0, "is_internal_transfer": False},
            {"tx_id": "T2", "amount": -50.0, "is_internal_transfer": False},
        ])

        external_df, summary = net_transfers(df)

        assert len(external_df) == 2
        assert summary["num_transfers_removed"] == 0
