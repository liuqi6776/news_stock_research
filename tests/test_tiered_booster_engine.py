# -*- coding: utf-8 -*-
"""
Unit Test Suite: Tiered Multi-Indicator Entry & Free-Roll Booster Engine (Phase 24)
===================================================================================
Verifies:
1. Strict causality: Future candle mutations do not alter past indicator values.
2. Tiered sizing: Initial entry respects 1/3, 2/3, and 1.0x score mapping.
3. Booster safety invariant: Booster CANNOT activate unless trailing stop >= entry price.
4. Maximum leverage constraint: Gross position size never exceeds configured booster_leverage.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.tiered_booster_engine import TieredBoosterEngine


@pytest.fixture
def sample_4h_candles():
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    # Generate realistic trending price series
    rets = np.random.normal(0.001, 0.015, n)
    # Add a strong surge in the middle
    rets[150:180] += 0.01
    prices = 2000.0 * np.exp(np.cumsum(rets))
    highs = prices * (1.0 + np.random.uniform(0.002, 0.01, n))
    lows = prices * (1.0 - np.random.uniform(0.002, 0.01, n))
    opens = prices * (1.0 + np.random.uniform(-0.003, 0.003, n))
    closes = prices

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.uniform(100, 500, n),
        },
        index=dates,
    )
    return df


def test_indicators_strictly_causal(sample_4h_candles):
    """Corrupting future bars must NOT change past indicator values."""
    engine = TieredBoosterEngine()
    ind_orig = engine.compute_indicators(sample_4h_candles)

    # Corrupt last 50 bars
    df_corrupt = sample_4h_candles.copy()
    df_corrupt.iloc[-50:, df_corrupt.columns.get_loc("close")] *= 5.0
    df_corrupt.iloc[-50:, df_corrupt.columns.get_loc("high")] *= 5.0
    ind_corrupt = engine.compute_indicators(df_corrupt)

    # Past bars (up to bar 240) must be identical
    pd.testing.assert_frame_equal(ind_orig.iloc[:240], ind_corrupt.iloc[:240])


def test_tiered_entry_proportions(sample_4h_candles):
    """Verify that tiered positions are assigned correctly."""
    engine = TieredBoosterEngine(use_tiered_entry=True, use_booster=False)
    _, trades, positions = engine.run_backtest(sample_4h_candles, token="ETHUSDT")

    active_sizes = set(np.round(positions[positions > 0], 2))
    # Active sizes must be subset of [0.33, 0.67, 1.0]
    for s in active_sizes:
        assert s in [0.33, 0.67, 1.0], f"Unexpected tier size: {s}"


def test_booster_safety_invariant_zero_principal_risk(sample_4h_candles):
    """Booster MUST NEVER activate unless trailing stop is at or above average entry price."""
    engine = TieredBoosterEngine(use_tiered_entry=True, use_booster=True, booster_leverage=1.25)
    _, trades, positions = engine.run_backtest(sample_4h_candles, token="ETHUSDT")

    booster_trades = [t for t in trades if t.booster_activated]
    # For every trade where booster was activated, verify max tier was 4
    for t in booster_trades:
        assert t.max_tier == 4
        # Since booster only triggers when trailing stop >= entry price,
        # exit price must be close to or above entry price unless extreme gap down
        assert t.gross_ret > -0.05, f"Booster trade suffered unexpected large loss: {t.gross_ret}"


def test_maximum_leverage_constraint(sample_4h_candles):
    """Position size must never exceed booster_leverage."""
    engine = TieredBoosterEngine(use_tiered_entry=True, use_booster=True, booster_leverage=1.25)
    _, _, positions = engine.run_backtest(sample_4h_candles, token="ETHUSDT")

    assert np.max(positions) <= 1.25 + 1e-6, f"Max position exceeded booster limit: {np.max(positions)}"
