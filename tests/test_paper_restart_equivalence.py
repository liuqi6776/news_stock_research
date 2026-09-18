# -*- coding: utf-8 -*-
"""
Unit Tests: Restart Recovery & Equivalence (Phase 22)
=====================================================
Verifies:
1. test_pending_order_survives_restart
2. test_restart_equivalence
"""

import pandas as pd
import numpy as np
import pytest

from crypto_quant.paper.execution import PaperExecutionEngine
from crypto_quant.paper.service import PaperService
from crypto_quant.paper.state import PortfolioPaperState


def create_test_series(n_bars: int = 220) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=n_bars, freq="4h", tz="UTC")
    closes = 2000.0 + np.cumsum(np.random.normal(0, 5, n_bars))
    highs = closes + 10.0
    lows = closes - 10.0
    opens = closes - 1.0
    vols = np.random.uniform(50, 100, n_bars)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols
    }, index=times)


def test_pending_order_survives_restart(tmp_path):
    """Pending order saved in state file must survive shutdown and execute on next bar arrival."""
    state_file = tmp_path / "restart_state.json"
    journal_file = tmp_path / "restart_journal.jsonl"

    state = PortfolioPaperState.create_initial()
    exec_engine = PaperExecutionEngine()
    order = exec_engine.create_order(
        symbol="ETHUSDT",
        side="BUY",
        quantity_fraction=1.0,
        signal_time="2024-01-01 04:00:00",
        intended_fill_time="2024-01-01 08:00:00",
        reason="BOLLINGER_BREAKOUT",
        metadata={"initial_stop": 1900.0},
    )
    state.eth_state.pending_order = order.to_dict()
    state.save_atomic(state_file)

    # Initialize new service instance pointing to the same state file
    service = PaperService(state_path=state_file, journal_path=journal_file, symbols=["ETHUSDT"])
    loaded_state = service.load_or_initialize_state()

    assert loaded_state.eth_state.pending_order is not None
    assert loaded_state.eth_state.pending_order["side"] == "BUY"
    assert loaded_state.eth_state.pending_order["order_id"] == order.order_id


def test_restart_equivalence(tmp_path):
    """
    Running N bars continuously in one run vs running with service restart
    at each bar must produce an identical state and portfolio trajectory.
    """
    df = create_test_series(n_bars=220)
    candles = {"ETHUSDT": df, "SOLUSDT": df}

    # Run A: Continuous run
    state_file_a = tmp_path / "state_a.json"
    journal_file_a = tmp_path / "journal_a.jsonl"
    snapshot_file_a = tmp_path / "snapshot_a.json"
    service_a = PaperService(state_path=state_file_a, journal_path=journal_file_a, snapshot_path=snapshot_file_a)
    service_a.run_once(external_candles=candles)
    final_state_a = PortfolioPaperState.load(state_file_a)

    # Run B: Step-by-step with new service instance created for each bar step
    state_file_b = tmp_path / "state_b.json"
    journal_file_b = tmp_path / "journal_b.jsonl"
    snapshot_file_b = tmp_path / "snapshot_b.json"

    # Step through from warmup to end
    for k in range(200, len(df) + 1):
        sub_candles = {"ETHUSDT": df.iloc[:k], "SOLUSDT": df.iloc[:k]}
        service_b = PaperService(state_path=state_file_b, journal_path=journal_file_b, snapshot_path=snapshot_file_b)
        service_b.run_once(external_candles=sub_candles)

    final_state_b = PortfolioPaperState.load(state_file_b)

    # Verify complete state equivalence
    assert final_state_a.portfolio_equity == pytest.approx(final_state_b.portfolio_equity, rel=1e-9)
    assert final_state_a.eth_state.position == final_state_b.eth_state.position
    assert final_state_a.eth_state.trailing_stop_price == pytest.approx(final_state_b.eth_state.trailing_stop_price, rel=1e-9)
    assert final_state_a.sol_state.position == final_state_b.sol_state.position
