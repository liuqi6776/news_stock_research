# -*- coding: utf-8 -*-
"""
Unit Tests for Macro Structural Trend & Wave Engine (Phase 19)
=============================================================
Verifies:
1. Strict indicator causality (no lookahead).
2. Trailing stop ratcheting monotonicity.
3. Holding duration expansion (multi-day to multi-week).
4. Macro regime multiplier sizing.
5. Transaction friction & funding carry causality.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crypto_quant.structural_trend_engine import StructuralTrendEngine, TradeRecord


@pytest.fixture
def synthetic_ohlcv():
    """Generates 300 bars of synthetic 4h OHLCV with an initial consolidation, a bull breakout, and a pullback."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=300, freq='4h')
    
    # Prices: flat around 100 for 120 bars, then ramp to 200, then drop to 150
    closes = np.ones(300) * 100.0
    closes[:120] += np.random.normal(0, 1.0, 120)
    closes[120:220] = np.linspace(100.0, 200.0, 100) + np.random.normal(0, 1.5, 100)
    closes[220:] = np.linspace(200.0, 150.0, 80) + np.random.normal(0, 2.0, 80)

    highs = closes + np.random.uniform(0.5, 2.5, 300)
    lows = closes - np.random.uniform(0.5, 2.5, 300)
    opens = (closes + np.roll(closes, 1)) / 2.0
    opens[0] = closes[0]

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.uniform(1000, 5000, 300)
    }, index=dates)
    return df


def test_indicator_causality(synthetic_ohlcv):
    engine = StructuralTrendEngine(mode='bollinger', lookback_bars=60)
    ind = engine.compute_indicators(synthetic_ohlcv)

    # Shifting check: modifying the last bar of raw close must NOT affect indicators at bar -1
    modified_df = synthetic_ohlcv.copy()
    modified_df.iloc[-1, modified_df.columns.get_loc('close')] += 1000.0
    modified_ind = engine.compute_indicators(modified_df)

    # Indicator at bar -1 (using shifted values) must remain IDENTICAL
    pd.testing.assert_series_equal(ind['bb_mid'].iloc[:-1], modified_ind['bb_mid'].iloc[:-1])
    pd.testing.assert_series_equal(ind['bb_upper'].iloc[:-1], modified_ind['bb_upper'].iloc[:-1])
    pd.testing.assert_series_equal(ind['donchian_high'].iloc[:-1], modified_ind['donchian_high'].iloc[:-1])


def test_trailing_stop_monotonicity():
    """Verify that during a sustained rally, trailing stop prices ratchet upward and lock in profits on pullback."""
    dates = pd.date_range('2024-01-01', periods=150, freq='4h')
    closes = np.ones(150) * 100.0
    closes[40:110] = np.linspace(100.0, 220.0, 70)  # Sharp bull surge
    closes[110:] = np.linspace(220.0, 160.0, 40)    # Pullback triggering trailing stop
    highs = closes + 2.0
    lows = closes - 2.0
    opens = closes - 0.5
    df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes}, index=dates)

    engine = StructuralTrendEngine(mode='bollinger', lookback_bars=30, atr_trailing_mult=2.5)
    rets, trades, pos = engine.run_backtest(df, token='TEST')

    assert len(trades) >= 1
    t = trades[0]
    # Trade entered around bar 41 and rode the trend
    assert t.gross_ret > 0.80  # Captured major bull expansion (>80% gain)
    assert t.duration_days >= 10.0  # Multi-week holding duration
    assert t.exit_reason == 'TRAILING_STOP'


def test_holding_duration_expansion(synthetic_ohlcv):
    """Verify that the engine holds through minor pullbacks, with average duration > 3 days."""
    engine = StructuralTrendEngine(mode='bollinger', lookback_bars=60, exit_lookback_bars=30)
    rets, trades, pos = engine.run_backtest(synthetic_ohlcv, token='SYNTH')

    assert len(trades) > 0
    avg_dur_days = np.mean([t.duration_days for t in trades])
    assert avg_dur_days >= 3.0  # Multi-day holding horizon


def test_macro_regime_multiplier_sizing(synthetic_ohlcv):
    """Verify that macro regime multipliers scale position size correctly."""
    engine = StructuralTrendEngine(mode='bollinger', lookback_bars=60)
    
    # Multiplier at 0.5x
    half_mult = pd.Series(0.5, index=synthetic_ohlcv.index)
    _, trades_half, _ = engine.run_backtest(synthetic_ohlcv, token='SYNTH', macro_multipliers=half_mult)

    # Multiplier at 1.0x
    full_mult = pd.Series(1.0, index=synthetic_ohlcv.index)
    _, trades_full, _ = engine.run_backtest(synthetic_ohlcv, token='SYNTH', macro_multipliers=full_mult)

    assert len(trades_half) == len(trades_full)
    for th, tf in zip(trades_half, trades_full):
        assert pytest.approx(th.size, rel=1e-3) == 0.5 * tf.size


def test_friction_and_funding_carry(synthetic_ohlcv):
    """Verify that fees are deducted and funding carry is accurately aggregated."""
    engine = StructuralTrendEngine(mode='bollinger', lookback_bars=60, fee_and_slippage=0.001)  # 10 bps
    
    # Constant funding of 0.01% per 8h
    funding = pd.Series(0.0001, index=synthetic_ohlcv.index)
    rets, trades, pos = engine.run_backtest(synthetic_ohlcv, token='SYNTH', funding_rate=funding)

    for t in trades:
        # Long pays positive funding: funding carry should be negative
        assert t.funding_carry < 0.0
        # Net return = gross_ret - 2*0.001 + funding_carry
        expected_net = t.gross_ret - 0.002 + t.funding_carry
        assert pytest.approx(t.net_ret, abs=1e-6) == expected_net
