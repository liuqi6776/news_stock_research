# -*- coding: utf-8 -*-
"""
Unit Tests: State Management & Safety Guards (Phase 22)
======================================================
Verifies:
1. test_atomic_state_save
2. test_corrupt_state_refuses_start
3. test_service_refuses_real_order_mode
"""

import json
import os
import tempfile
from pathlib import Path
import pytest

from crypto_quant.paper.service import PaperService
from crypto_quant.paper.state import (
    PaperStrategyState,
    PortfolioPaperState,
    StateCorruptionError,
)
import crypto_quant.paper.config as config_mod


def test_atomic_state_save(tmp_path):
    """Verifies state is saved atomically via temporary file and replace."""
    state_file = tmp_path / "test_state.json"
    state = PortfolioPaperState.create_initial()
    state.eth_state.position = 1
    state.eth_state.position_size = 1.0
    state.eth_state.entry_price = 2500.0

    state.save_atomic(state_file)

    assert state_file.exists()
    loaded = PortfolioPaperState.load(state_file)
    assert loaded.eth_state.position == 1
    assert loaded.eth_state.entry_price == 2500.0
    assert loaded.portfolio_equity == 1.0


def test_corrupt_state_refuses_start(tmp_path):
    """Corrupted JSON or invalid schema in state file must halt with StateCorruptionError."""
    state_file = tmp_path / "corrupt_state.json"

    # Case 1: Broken JSON syntax
    state_file.write_text("{broken json ...", encoding="utf-8")
    with pytest.raises(StateCorruptionError, match="invalid JSON"):
        PortfolioPaperState.load(state_file)

    # Case 2: Missing required schema fields
    state_file.write_text(json.dumps({"some_key": "some_value"}), encoding="utf-8")
    with pytest.raises(StateCorruptionError, match="Missing required portfolio state field"):
        PortfolioPaperState.load(state_file)


def test_service_refuses_real_order_mode(monkeypatch):
    """If ENABLE_REAL_ORDERS is ever set to True, PaperService must refuse to boot."""
    monkeypatch.setattr(config_mod, "ENABLE_REAL_ORDERS", True)
    with pytest.raises(RuntimeError, match="Real orders are strictly forbidden"):
        PaperService()
