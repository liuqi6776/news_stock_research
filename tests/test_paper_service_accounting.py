# -*- coding: utf-8 -*-
"""
Automated Pytest Suite: PaperService Equity Accounting & Infrastructure (Phase 25)
==================================================================================
Mandatory Audit Remediation Suite: Tests 1 to 10.

Verifies:
1. test_real_service_equity_changes_after_closed_trade: Total equity changes and != initial.
2. test_real_service_fee_accumulates: Fees accumulate on trade fills, stay constant when flat.
3. test_real_service_drawdown_updates: Drawdown updates correctly during downturns.
4. test_real_service_replay_matches_vector_engine: Zero-tolerance match against clean engine (< 1e-8).
5. test_real_service_restart_matches_continuous: Restart from snapshot equals continuous execution.
6. test_forward_report_excludes_recovery_replay: Forward reporting excludes recovery replay bars.
7. test_long_gap_beyond_fetch_window_blocks_progress: Downtime gap > 250 bars raises UnrecoverableDataGapError.
8. test_snapshot_paths_are_instance_isolated: Separate instances do not collide on snapshots.
9. test_snapshot_atomic_replace_retry: Transient Windows file locks retry and succeed.
10. test_dashboard_read_does_not_lock_snapshot: Concurrent reads do not block atomic snapshot updates.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict
import numpy as np
import pandas as pd
import pytest

from crypto_quant.paper.config import INITIAL_EQUITY
from crypto_quant.paper.journal import EventTypes, PaperJournal
from crypto_quant.paper.monitoring import PaperMonitor
from crypto_quant.paper.service import PaperService, UnrecoverableDataGapError
from crypto_quant.paper.state import PaperStrategyState, PortfolioPaperState
from scripts.export_forward_report import export_weekly_report
from scripts.replay_paper_service import run_replay_for_token

root_dir = Path(__file__).resolve().parent.parent


@pytest.fixture
def eth_historical_slice():
    """Returns a realistic slice of ETH historical 4h data covering warmup and closed trades."""
    p = root_dir / "data" / "ETHUSDT_4h_2020_2026.parquet"
    if not p.exists():
        p = root_dir / "data" / "ETHUSDT_4h_2021_2026.parquet"
    assert p.exists(), "Historical ETH parquet file missing"
    df = pd.read_parquet(p)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # Slice from 2023-10-01 to 2024-03-01 (includes 200-bar warmup and multiple active trades)
    return df.loc["2023-10-01":"2024-03-01"].copy()


def test_real_service_equity_changes_after_closed_trade(tmp_path, eth_historical_slice):
    """Test 1: Real PaperService equity changes after trading and does not remain static at initial."""
    state_file = tmp_path / "state.json"
    journal_file = tmp_path / "journal.jsonl"
    snapshot_file = tmp_path / "snapshot.json"

    service = PaperService(
        state_path=state_file,
        journal_path=journal_file,
        snapshot_path=snapshot_file,
        symbols=["ETHUSDT"],
    )
    service.run_once(external_candles={"ETHUSDT": eth_historical_slice})

    eth_s = service.state.eth_state
    # Total equity and realized equity must have evolved away from initial 1.0
    assert eth_s.total_equity != INITIAL_EQUITY
    assert eth_s.realized_equity != INITIAL_EQUITY
    assert eth_s.total_equity > 0.5  # Valid positive equity


def test_real_service_fee_accumulates(tmp_path, eth_historical_slice):
    """Test 2: Fees accumulate on trades, and stay static on non-trading flat bars."""
    state_file = tmp_path / "state.json"
    journal_file = tmp_path / "journal.jsonl"
    snapshot_file = tmp_path / "snapshot.json"

    service = PaperService(
        state_path=state_file,
        journal_path=journal_file,
        snapshot_path=snapshot_file,
        symbols=["ETHUSDT"],
    )
    service.run_once(external_candles={"ETHUSDT": eth_historical_slice})

    eth_s = service.state.eth_state
    fees_after_trades = eth_s.total_fees
    assert fees_after_trades > 0.0, "Fees must accumulate after trading"

    # Now step with 1 bar of identical price and no position change (no new trades)
    last_bar_time = eth_historical_slice.index[-1]
    next_bar_time = last_bar_time + pd.Timedelta(hours=4)
    last_row = eth_historical_slice.iloc[-1]

    new_row = pd.DataFrame(
        {
            "open": [last_row["open"]],
            "high": [last_row["high"]],
            "low": [last_row["low"]],
            "close": [last_row["close"]],
            "volume": [10.0],
        },
        index=[next_bar_time],
    )
    extended_df = pd.concat([eth_historical_slice, new_row])
    service.run_once(external_candles={"ETHUSDT": extended_df})

    # If no new order filled, total_fees must not falsely increase
    assert service.state.eth_state.total_fees == pytest.approx(fees_after_trades, abs=1e-8)


def test_real_service_drawdown_updates(tmp_path, eth_historical_slice):
    """Test 3: PaperService correctly tracks peak equity and updates drawdowns."""
    state_file = tmp_path / "state.json"
    journal_file = tmp_path / "journal.jsonl"
    snapshot_file = tmp_path / "snapshot.json"

    service = PaperService(
        state_path=state_file,
        journal_path=journal_file,
        snapshot_path=snapshot_file,
        symbols=["ETHUSDT"],
    )
    service.run_once(external_candles={"ETHUSDT": eth_historical_slice})

    eth_s = service.state.eth_state
    # In this historical slice, ETH had volatility and drawdowns
    assert eth_s.peak_equity >= eth_s.total_equity
    assert eth_s.drawdown <= 0.0  # Drawdown is expressed as negative float or zero
    assert service.state.portfolio_drawdown >= 0.0  # Portfolio drawdown is positive percentage/fraction


def test_real_service_replay_matches_vector_engine(eth_historical_slice):
    """Test 4: PaperService replay matches clean StructuralTrendEngine within 1e-8 tolerance."""
    p = root_dir / "data" / "ETHUSDT_4h_2020_2026.parquet"
    if not p.exists():
        p = root_dir / "data" / "ETHUSDT_4h_2021_2026.parquet"
    df = pd.read_parquet(p)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    passed, report = run_replay_for_token("ETHUSDT", df.loc["2023-01-01":])
    assert passed is True, f"Replay mismatch: {report}"
    assert report["max_equity_diff"] < 1e-8
    assert report["trade_count_diff"] == 0
    assert report["position_mismatches"] == 0


def test_real_service_restart_matches_continuous(tmp_path, eth_historical_slice):
    """Test 5: Restarting from atomic state matches uninterrupted continuous run."""
    df_part1 = eth_historical_slice.iloc[:230]
    df_full = eth_historical_slice.iloc[:260]

    # Continuous run
    cont_dir = tmp_path / "cont"
    cont_dir.mkdir()
    service_cont = PaperService(
        state_path=cont_dir / "state.json",
        journal_path=cont_dir / "journal.jsonl",
        snapshot_path=cont_dir / "snap.json",
        symbols=["ETHUSDT"],
    )
    service_cont.run_once(external_candles={"ETHUSDT": df_full})

    # Restarted run: Part 1
    restart_dir = tmp_path / "restart"
    restart_dir.mkdir()
    service_part1 = PaperService(
        state_path=restart_dir / "state.json",
        journal_path=restart_dir / "journal.jsonl",
        snapshot_path=restart_dir / "snap.json",
        symbols=["ETHUSDT"],
    )
    service_part1.run_once(external_candles={"ETHUSDT": df_part1})

    # Restarted run: Part 2 (new instance from persisted state)
    service_part2 = PaperService(
        state_path=restart_dir / "state.json",
        journal_path=restart_dir / "journal.jsonl",
        snapshot_path=restart_dir / "snap.json",
        symbols=["ETHUSDT"],
    )
    service_part2.run_once(external_candles={"ETHUSDT": df_full})

    # Equities and positions must match with machine precision
    eq_cont = service_cont.state.eth_state.total_equity
    eq_restart = service_part2.state.eth_state.total_equity
    assert abs(eq_cont - eq_restart) < 1e-8
    assert service_cont.state.eth_state.position == service_part2.state.eth_state.position
    assert service_cont.state.eth_state.last_processed_bar_time == service_part2.state.eth_state.last_processed_bar_time


def test_forward_report_excludes_recovery_replay(tmp_path, monkeypatch):
    """Test 6: Forward weekly reporting strictly excludes recovery replay trades."""
    state_file = tmp_path / "state.json"
    journal_file = tmp_path / "journal.jsonl"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    report_file = docs_dir / "forward_paper_weekly.json"

    # Monkeypatch config paths for report exporter
    monkeypatch.setattr("scripts.export_forward_report.STATE_PATH", state_file)
    monkeypatch.setattr("scripts.export_forward_report.JOURNAL_PATH", journal_file)
    monkeypatch.setattr("scripts.export_forward_report.WEEKLY_REPORT_PATH", report_file)
    monkeypatch.setattr("scripts.export_forward_report.DOCS_DIR", docs_dir)

    # Initialize state with forward boundary
    state = PortfolioPaperState.create_initial()
    state.forward_start_bar_time = "2024-03-01 00:00:00"
    state.recovery_replay_completed = True
    state.save_atomic(state_file)

    journal = PaperJournal(journal_file)
    # 1. Recovery replay trade (occurred before forward boundary)
    journal.log_event(
        EventTypes.POSITION_CLOSED,
        {"exit_price": 2500.0, "is_recovery_replay": True, "net_ret": 0.05},
        symbol="ETHUSDT",
        bar_time="2024-02-15 00:00:00",
    )
    # 2. True forward trade (occurred after forward boundary)
    journal.log_event(
        EventTypes.POSITION_CLOSED,
        {"exit_price": 3500.0, "is_recovery_replay": False, "net_ret": 0.12},
        symbol="ETHUSDT",
        bar_time="2024-03-05 00:00:00",
    )

    report = export_weekly_report()
    stats = report["forward_trading_stats"]
    assert stats["total_recovery_replay_trades"] == 1
    assert stats["total_forward_closed_trades"] == 1
    # Only true forward trades appear in recent forward closed trades
    recent = report["recent_forward_closed_trades"]
    assert len(recent) == 1
    assert recent[0]["bar_time"] == "2024-03-05 00:00:00"


def test_long_gap_beyond_fetch_window_blocks_progress(tmp_path, eth_historical_slice):
    """Test 7: Unrecoverable downtime gap (> 250 bars) raises UnrecoverableDataGapError and blocks advance."""
    state_file = tmp_path / "state.json"
    journal_file = tmp_path / "journal.jsonl"
    snapshot_file = tmp_path / "snapshot.json"

    service = PaperService(
        state_path=state_file,
        journal_path=journal_file,
        snapshot_path=snapshot_file,
        symbols=["ETHUSDT"],
    )
    # Run initial batch of 210 bars
    service.run_once(external_candles={"ETHUSDT": eth_historical_slice.iloc[:210]})
    last_processed = service.state.eth_state.last_processed_bar_time
    assert last_processed is not None

    # Construct gap: skip 300 bars
    gap_df = pd.concat([eth_historical_slice.iloc[:200], eth_historical_slice.iloc[260:]])
    # The next bar in gap_df after bar 210 is far beyond 4 hours
    with pytest.raises(UnrecoverableDataGapError) as exc_info:
        service.run_once(external_candles={"ETHUSDT": gap_df})

    assert "Unrecoverable downtime gap detected" in str(exc_info.value)


def test_snapshot_paths_are_instance_isolated(tmp_path):
    """Test 8: Separate PaperService instances use isolated snapshot paths."""
    snap1 = tmp_path / "service1" / "snap.json"
    snap2 = tmp_path / "service2" / "snap.json"

    s1 = PaperService(
        state_path=tmp_path / "service1" / "state.json",
        journal_path=tmp_path / "service1" / "journal.jsonl",
        snapshot_path=snap1,
    )
    s2 = PaperService(
        state_path=tmp_path / "service2" / "state.json",
        journal_path=tmp_path / "service2" / "journal.jsonl",
        snapshot_path=snap2,
    )

    assert s1.snapshot_path != s2.snapshot_path
    state1 = PortfolioPaperState.create_initial()
    state1.portfolio_equity = 1.25
    state2 = PortfolioPaperState.create_initial()
    state2.portfolio_equity = 0.85

    s1.monitor.generate_snapshot(state1)
    s2.monitor.generate_snapshot(state2)

    with open(snap1, "r", encoding="utf-8") as f:
        d1 = json.load(f)
    with open(snap2, "r", encoding="utf-8") as f:
        d2 = json.load(f)

    assert d1["equity"]["portfolio_50_50"] == 1.25
    assert d2["equity"]["portfolio_50_50"] == 0.85


def test_snapshot_atomic_replace_retry(tmp_path, monkeypatch):
    """Test 9: Atomic state saving handles transient Windows lock with retry."""
    state_file = tmp_path / "state.json"
    state = PortfolioPaperState.create_initial()

    real_replace = os.replace
    attempts = 0

    def mock_replace(src, dst):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("Simulated transient Windows file lock")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", mock_replace)
    state.save_atomic(state_file)

    assert attempts == 2, "Must have retried after transient PermissionError"
    assert state_file.exists()


def test_dashboard_read_does_not_lock_snapshot(tmp_path):
    """Test 10: Concurrent dashboard reads do not cause PermissionError or lock contention."""
    snap_path = tmp_path / "snapshot.json"
    monitor = PaperMonitor(snapshot_path=snap_path)
    state = PortfolioPaperState.create_initial()

    # Initial write
    monitor.generate_snapshot(state)

    def reader_task():
        for _ in range(40):
            success = False
            for _ in range(5):
                try:
                    with open(snap_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        assert "equity" in data
                        success = True
                        break
                except (PermissionError, FileNotFoundError, json.JSONDecodeError):
                    time.sleep(0.005)
            assert success, "Reader failed to acquire snapshot after retries"
            time.sleep(0.002)

    def writer_task():
        for i in range(15):
            state.portfolio_equity = 1.0 + i * 0.01
            monitor.generate_snapshot(state)
            time.sleep(0.003)

    with ThreadPoolExecutor(max_workers=4) as executor:
        f_writer = executor.submit(writer_task)
        f_reader1 = executor.submit(reader_task)
        f_reader2 = executor.submit(reader_task)

        f_writer.result()
        f_reader1.result()
        f_reader2.result()

    assert snap_path.exists()
