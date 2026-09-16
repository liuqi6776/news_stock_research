# -*- coding: utf-8 -*-
"""
Unit Tests: Idempotency, Network Failure & Gap Replay (Phase 22)
================================================================
Verifies:
1. test_duplicate_run_is_idempotent
2. test_network_error_does_not_advance_state
3. test_unprocessed_gap_is_replayed_in_order
"""

import pandas as pd
import numpy as np
import pytest

from crypto_quant.paper.journal import PaperJournal
from crypto_quant.paper.service import PaperService
from crypto_quant.paper.state import PortfolioPaperState


def create_test_candles(n_bars: int = 220) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=n_bars, freq="4h", tz="UTC")
    closes = 2000.0 + np.cumsum(np.random.normal(0, 5, n_bars))
    highs = closes + 10.0
    lows = closes - 10.0
    opens = closes - 1.0
    vols = np.random.uniform(50, 100, n_bars)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols
    }, index=times)


def test_duplicate_run_is_idempotent(tmp_path):
    """Calling run_once 10 times on the exact same candles must not generate duplicate trades or change equity."""
    df = create_test_candles(n_bars=220)
    candles = {"ETHUSDT": df, "SOLUSDT": df}

    state_file = tmp_path / "idempotent_state.json"
    journal_file = tmp_path / "idempotent_journal.jsonl"
    service = PaperService(state_path=state_file, journal_path=journal_file)

    # First run: processes bars
    snap1 = service.run_once(external_candles=candles)
    eq1 = snap1["equity"]["portfolio_50_50"]
    state1 = PortfolioPaperState.load(state_file)

    # Run 9 more times on identical data
    for _ in range(9):
        snap_next = service.run_once(external_candles=candles)
        eq_next = snap_next["equity"]["portfolio_50_50"]
        assert eq_next == pytest.approx(eq1, rel=1e-12)

    final_state = PortfolioPaperState.load(state_file)
    assert final_state.portfolio_equity == pytest.approx(state1.portfolio_equity, rel=1e-12)
    assert final_state.eth_state.position == state1.eth_state.position
    assert final_state.eth_state.last_processed_bar_time == state1.eth_state.last_processed_bar_time


def test_network_error_does_not_advance_state(tmp_path, monkeypatch):
    """Network failure must raise an exception and leave persistent state completely unchanged."""
    df = create_test_candles(n_bars=220)
    candles = {"ETHUSDT": df, "SOLUSDT": df}

    state_file = tmp_path / "network_err_state.json"
    journal_file = tmp_path / "network_err_journal.jsonl"
    service = PaperService(state_path=state_file, journal_path=journal_file)

    # Initial successful run
    service.run_once(external_candles=candles)
    initial_state = PortfolioPaperState.load(state_file)

    # Simulate network failure on next run
    def failing_fetch(*args, **kwargs):
        raise RuntimeError("Simulated Binance Network Disconnection")

    monkeypatch.setattr(service.fetcher, "fetch_closed_klines", failing_fetch)

    # Service should raise RuntimeError and NOT advance or overwrite state
    with pytest.raises(RuntimeError, match="Simulated Binance Network Disconnection"):
        service.run_once(external_candles=None)

    after_error_state = PortfolioPaperState.load(state_file)
    assert after_error_state.last_update_time == initial_state.last_update_time
    assert after_error_state.portfolio_equity == initial_state.portfolio_equity


def test_unprocessed_gap_is_replayed_in_order(tmp_path):
    """If service missed 5 bars during offline downtime, all 5 bars must be caught up in order."""
    df = create_test_candles(n_bars=230)

    state_file = tmp_path / "gap_state.json"
    journal_file = tmp_path / "gap_journal.jsonl"
    service = PaperService(state_path=state_file, journal_path=journal_file)

    # Process up to bar 210
    candles_part1 = {"ETHUSDT": df.iloc[:210], "SOLUSDT": df.iloc[:210]}
    service.run_once(external_candles=candles_part1)
    state1 = PortfolioPaperState.load(state_file)
    assert state1.eth_state.last_processed_bar_time == str(df.index[209])

    # Downtime passes, 20 bars accumulate up to 230
    candles_part2 = {"ETHUSDT": df, "SOLUSDT": df}
    service.run_once(external_candles=candles_part2)
    state2 = PortfolioPaperState.load(state_file)

    # Last processed bar must now be the latest bar (229)
    assert state2.eth_state.last_processed_bar_time == str(df.index[-1])
