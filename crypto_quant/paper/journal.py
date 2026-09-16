# -*- coding: utf-8 -*-
"""
Journal Module: Append-Only Institutional Event Ledger (Phase 22)
================================================================
Maintains an append-only JSON Lines ledger for full auditability
and forensic reconstruction of all paper trading events.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from crypto_quant.paper.config import (
    CONFIG_HASH,
    EXPERIMENT_ID,
    JOURNAL_PATH,
    get_code_commit,
)


class EventTypes:
    SERVICE_STARTED = "SERVICE_STARTED"
    DATA_FETCHED = "DATA_FETCHED"
    BAR_ACCEPTED = "BAR_ACCEPTED"
    BAR_REJECTED = "BAR_REJECTED"
    SIGNAL_CREATED = "SIGNAL_CREATED"
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_FILLED = "ORDER_FILLED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_UPDATED = "POSITION_UPDATED"
    POSITION_CLOSED = "POSITION_CLOSED"
    STOP_UPDATED = "STOP_UPDATED"
    STATE_SAVED = "STATE_SAVED"
    ERROR = "ERROR"
    SERVICE_STOPPED = "SERVICE_STOPPED"


class PaperJournal:
    """
    Append-only JSON Lines journal manager.
    """

    def __init__(self, journal_path: Union[str, Path] = JOURNAL_PATH):
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.code_commit = get_code_commit()
        self.config_hash = CONFIG_HASH

    def log_event(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
        bar_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Appends an immutable event record to the journal file.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        payload = payload or {}

        # Construct unique canonical event ID
        event_content = f"{EXPERIMENT_ID}:{symbol}:{bar_time}:{event_type}:{now_utc}:{json.dumps(payload, sort_keys=True)}"
        event_id = hashlib.sha256(event_content.encode("utf-8")).hexdigest()[:16]

        record = {
            "event_id": event_id,
            "event_type": event_type,
            "timestamp_utc": now_utc,
            "symbol": symbol,
            "bar_time": str(bar_time) if bar_time is not None else None,
            "payload": payload,
            "code_commit": self.code_commit,
            "config_hash": self.config_hash,
        }

        line = json.dumps(record, sort_keys=True) + "\n"
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(line)

        return record

    def read_all_events(self) -> List[Dict[str, Any]]:
        """Reads all events from the journal."""
        if not self.journal_path.exists():
            return []
        events = []
        with open(self.journal_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events

    def get_event_count(self) -> int:
        """Returns total number of lines in the journal."""
        if not self.journal_path.exists():
            return 0
        with open(self.journal_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
