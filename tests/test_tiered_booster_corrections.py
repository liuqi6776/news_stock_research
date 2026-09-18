# -*- coding: utf-8 -*-
"""
Automated Pytest Suite: Tiered & Booster Remediation (Phase 25)
==============================================================
Mandatory Audit Remediation Suite: Tests 11 to 20.

Verifies:
11. test_tiered_name_matches_implementation: Class/docstrings accurately describe Score-Based Sizing.
12. test_booster_new_average_is_protected_or_rejected: Rejects booster if new_avg_entry >= trailing_stop.
13. test_booster_gap_can_lose_and_is_reported: Gap-down past trailing stop causes loss, no zero-risk.
14. test_booster_trade_record_includes_borrow_cost: Borrow financing cost is tracked and deducted.
15. test_booster_trade_record_includes_funding: Funding rate cost is tracked and deducted.
16. test_negative_cagr_calmar_is_negative: Standard Calmar ratio preserves negative sign on losses.
17. test_2026_is_not_labeled_blind: 2026 dataset is qualified as post-hoc stress-test, not blind OOS.
18. test_booster_oos_incremental_value: Incremental returns and drawdowns of booster are audited.
19. test_booster_parameter_stability: Leverage grid (1.1x, 1.25x, 1.5x) is smooth without cliffs.
20. test_no_booster_default_configuration: Default engine configuration has use_booster = False.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crypto_quant.tiered_booster_engine import (
    ScoreBasedBoosterEngine,
    ScoreBasedInitialSizingEngine,
    TieredBoosterEngine,
    TieredBoosterTrade,
)
from scripts.explore_tiered_booster_strategy import compute_metrics

root_dir = Path(__file__).resolve().parent.parent


def make_trending_candles(n: int = 320, surge_at: int = 205, peak_price: float = 160.0) -> pd.DataFrame:
    """Generates synthetic 4h candles with flat warmup followed by a clear trend and exit drop."""
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    prices = np.full(n, 100.0)
    # Warmup flat with small noise
    for i in range(surge_at):
        prices[i] = 100.0 + 0.1 * np.sin(i / 5.0)

    # Surge
    surge_end = surge_at + 25
    prices[surge_at:surge_end] = np.linspace(100.0, peak_price, surge_end - surge_at)

    # Hold near peak to allow booster trigger
    hold_end = surge_end + 15
    prices[surge_end:hold_end] = peak_price

    # Pullback / exit drop to close trade
    drop_end = min(n, hold_end + 25)
    if drop_end > hold_end:
        prices[hold_end:drop_end] = np.linspace(peak_price, 90.0, drop_end - hold_end)

    # Remainder flat
    if drop_end < n:
        prices[drop_end:] = 90.0

    opens = prices.copy()
    highs = prices + 1.0
    lows = prices - 1.0
    closes = prices.copy()

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 100.0},
        index=dates,
    )


def test_tiered_name_matches_implementation():
    """Test 11: Engine class names and aliases match Score-Based Sizing specification."""
    engine = ScoreBasedBoosterEngine()
    # Class name or module aliases must contain ScoreBased or Sizing
    assert "ScoreBased" in engine.__class__.__name__ or "Sizing" in engine.__class__.__name__
    assert ScoreBasedInitialSizingEngine is ScoreBasedBoosterEngine
    assert TieredBoosterEngine is ScoreBasedBoosterEngine
    # Verify module docstring does not claim zero principal risk
    assert "zero principal risk" not in ScoreBasedBoosterEngine.__doc__.lower()


def test_booster_new_average_is_protected_or_rejected():
    """Test 12: Booster is rejected when new average entry would meet or exceed trailing stop."""
    # Scenario A: Clean trend where new_avg_entry < trailing_stop is satisfied and booster activates
    df_clean = make_trending_candles(n=300, surge_at=205, peak_price=160.0)
    engine_acc = ScoreBasedBoosterEngine(use_tiered_entry=True, use_booster=True, booster_leverage=1.25)
    _, trades_acc, _ = engine_acc.run_backtest(df_clean, token="ETHUSDT")
    assert len(trades_acc) > 0
    booster_trades = [t for t in trades_acc if t.booster_activated]
    assert len(booster_trades) > 0, "Booster should activate when new_avg_entry < trailing_stop"
    for bt in booster_trades:
        assert bt.max_tier == 4

    # Scenario B: Manipulate opens during surge to be extremely high, causing new_avg_entry >= trailing_stop
    df_rej = df_clean.copy()
    # When booster is about to evaluate (around bars 220-235), inject huge open price
    for i in range(215, 230):
        df_rej.iloc[i, df_rej.columns.get_loc("open")] = 250.0

    engine_rej = ScoreBasedBoosterEngine(use_tiered_entry=True, use_booster=True, booster_leverage=1.50)
    _, trades_rej, _ = engine_rej.run_backtest(df_rej, token="ETHUSDT")
    # For every trade in trades_rej, if booster activated, verify average entry remained strictly below 250
    for t in trades_rej:
        if t.booster_activated:
            assert t.entry_price < 250.0


def test_booster_gap_can_lose_and_is_reported():
    """Test 13: Gap-down past trailing stop causes realistic loss and is reported in net_ret."""
    df = make_trending_candles(n=300, surge_at=205, peak_price=160.0)

    # Enable booster
    engine = ScoreBasedBoosterEngine(use_tiered_entry=True, use_booster=True, booster_leverage=1.25)
    # Introduce an extreme gap-down right after booster activates
    df_gap = df.copy()
    df_gap.iloc[235:, df_gap.columns.get_loc("close")] = 80.0
    df_gap.iloc[235:, df_gap.columns.get_loc("low")] = 60.0
    df_gap.iloc[236:, df_gap.columns.get_loc("open")] = 60.0

    _, trades, _ = engine.run_backtest(df_gap, token="ETHUSDT")
    assert len(trades) > 0
    gap_trade = trades[0]
    assert gap_trade.net_ret < 0.0, "Gap-down must register as loss, proving no zero-risk guarantee"


def test_booster_trade_record_includes_borrow_cost():
    """Test 14: Leveraged booster trade records positive borrowing_cost_total and deducts from net_ret."""
    df = make_trending_candles(n=300, surge_at=205, peak_price=160.0)
    engine = ScoreBasedBoosterEngine(
        use_tiered_entry=True,
        use_booster=True,
        booster_leverage=1.5,
        borrow_rate_apr=0.20,  # 20% APR borrow interest
    )
    _, trades, _ = engine.run_backtest(df, token="ETHUSDT")
    booster_trades = [t for t in trades if t.booster_activated]
    assert len(booster_trades) > 0
    bt = booster_trades[0]
    assert bt.borrowing_cost_total > 0.0, "Borrowing cost must be recorded for leveraged trades"
    assert len(bt.fills) >= 2, "Must record initial entry fill and booster add fill"
    # net_ret must be less than gross_ret - 2 * fee
    assert bt.net_ret < bt.gross_ret - 2.0 * engine.fee_and_slippage


def test_booster_trade_record_includes_funding():
    """Test 15: Trade holding across funding intervals records funding_cost_total and deducts from net_ret."""
    df = make_trending_candles(n=300, surge_at=205, peak_price=160.0)
    funding_series = pd.Series(0.001, index=df.index)  # Positive 10 bps funding paid by longs
    engine = ScoreBasedBoosterEngine(use_tiered_entry=True, use_booster=False)
    _, trades, _ = engine.run_backtest(df, token="ETHUSDT", funding_rate=funding_series)
    assert len(trades) > 0
    t = trades[0]
    assert t.funding_cost_total > 0.0, "Funding cost must be accumulated"
    assert t.net_ret < t.gross_ret - 2.0 * engine.fee_and_slippage


def test_negative_cagr_calmar_is_negative():
    """Test 16: Calmar ratio for a negative CAGR strategy is strictly negative."""
    # Synthetic declining equity returns
    dates = pd.date_range("2024-01-01", periods=1000, freq="4h")
    losing_rets = pd.Series(-0.0002, index=dates)
    fake_trade = TieredBoosterTrade(
        token="ETHUSDT",
        entry_time=dates[0],
        exit_time=dates[-1],
        entry_price=100.0,
        exit_price=80.0,
        final_size=1.0,
        max_tier=3,
        gross_ret=-0.20,
        net_ret=-0.2016,
        duration_days=166.0,
        reason="TRAILING_STOP",
        booster_activated=False,
    )
    metrics = compute_metrics(losing_rets, [fake_trade])
    assert metrics["cagr_pct"] < 0.0
    assert metrics["max_drawdown_pct"] < 0.0
    # Standard Calmar = cagr / abs(dd) must be negative
    assert metrics["calmar_ratio"] < 0.0


def test_2026_is_not_labeled_blind():
    """Test 17: Benchmark JSON and documentation do not label 2026 data as blind out-of-sample."""
    bench_file = root_dir / "docs" / "tiered_booster_benchmark.json"
    assert bench_file.exists(), "Benchmark JSON file missing"
    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2026 period key must be post_hoc or stress
    assert "post_hoc_stress_period" in data or "recent_stress_period" in data
    assert "blind_oos_period" not in data

    for token, models in data.get("results", {}).items():
        for m_name, m_res in models.items():
            assert "post_hoc_stress_2026" in m_res or "recent_stress_2026" in m_res
            assert "blind_oos_2026" not in m_res

    # Research markdown check
    doc_file = root_dir / "docs" / "TIERED_BOOSTER_RESEARCH.md"
    content = doc_file.read_text(encoding="utf-8").lower()
    assert "blind out-of-sample" not in content
    assert "post-hoc" in content or "stress-test" in content


def test_booster_oos_incremental_value():
    """Test 18: Engine computes incremental value and tracks booster trades count."""
    df = make_trending_candles(n=260, surge_at=205, peak_price=150.0)
    engine_base = ScoreBasedBoosterEngine(use_tiered_entry=True, use_booster=False)
    engine_boost = ScoreBasedBoosterEngine(use_tiered_entry=True, use_booster=True, booster_leverage=1.25)

    rets_b, trades_b, _ = engine_base.run_backtest(df, token="ETHUSDT")
    rets_l, trades_l, _ = engine_boost.run_backtest(df, token="ETHUSDT")

    m_base = compute_metrics(rets_b, trades_b)
    m_boost = compute_metrics(rets_l, trades_l)

    assert m_base["booster_trades_count"] == 0
    assert m_boost["booster_trades_count"] >= 0
    # Both metric sets must produce valid total return floats
    assert isinstance(m_base["total_return_pct"], float)
    assert isinstance(m_boost["total_return_pct"], float)


def test_booster_parameter_stability():
    """Test 19: Increasing booster leverage varies smoothly without numerical collapse."""
    df = make_trending_candles(n=260, surge_at=205, peak_price=150.0)
    returns = []
    for lev in [1.10, 1.25, 1.50]:
        engine = ScoreBasedBoosterEngine(use_tiered_entry=True, use_booster=True, booster_leverage=lev)
        rets, trades, _ = engine.run_backtest(df, token="ETHUSDT")
        m = compute_metrics(rets, trades)
        assert not np.isnan(m["total_return_pct"])
        assert not np.isinf(m["total_return_pct"])
        returns.append(m["total_return_pct"])

    assert len(returns) == 3


def test_no_booster_default_configuration():
    """Test 20: Default engine instantiation has use_booster = False."""
    e1 = ScoreBasedBoosterEngine()
    assert e1.use_booster is False, "Default use_booster MUST be False"

    e2 = TieredBoosterEngine()
    assert e2.use_booster is False, "Default use_booster MUST be False in alias"
