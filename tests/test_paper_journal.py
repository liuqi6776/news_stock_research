# -*- coding: utf-8 -*-
"""
Unit Tests: Append-Only Journal Verification (Phase 22)
======================================================
Verifies:
1. test_journal_is_append_only
"""

import json
from crypto_quant.paper.journal import EventTypes, PaperJournal


def test_journal_is_append_only(tmp_path):
    """Verifies that journal appends lines sequentially without modifying existing records."""
    journal_path = tmp_path / "test_journal.jsonl"
    journal = PaperJournal(journal_path)

    # Log 3 events
    e1 = journal.log_event(EventTypes.SERVICE_STARTED, {"param": 1})
    e2 = journal.log_event(EventTypes.SIGNAL_CREATED, {"signal": "BUY"}, symbol="ETHUSDT")
    e3 = journal.log_event(EventTypes.ORDER_FILLED, {"fill_price": 2500.0}, symbol="ETHUSDT")

    events = journal.read_all_events()
    assert len(events) == 3
    assert events[0]["event_type"] == EventTypes.SERVICE_STARTED
    assert events[1]["event_type"] == EventTypes.SIGNAL_CREATED
    assert events[2]["event_type"] == EventTypes.ORDER_FILLED

    # Check that each line has required fields
    for ev in events:
        for req in ["event_id", "event_type", "timestamp_utc", "code_commit", "config_hash"]:
            assert req in ev
            assert ev[req] is not None
