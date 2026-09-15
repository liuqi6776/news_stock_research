# -*- coding: utf-8 -*-
"""
Unit Tests for Causal Chan-Lun Features & Wave Transformer (Phase 20)
====================================================================
Verifies:
1. Strict causality of Chan-Lun indicators (no lookahead leakage).
2. Central hub coordinate boundaries and distance scaling.
3. Multi-horizon wave tensor shapes and forward execution.
4. ChanWaveCombinedLoss numerical stability.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from crypto_quant.chan_features import compute_chan_features
from crypto_quant.crypto_transformer import ChanWaveCombinedLoss, CryptoSTChanTransformer


@pytest.fixture
def synthetic_candles():
    """Generates 200 bars of synthetic 4h OHLCV with distinct swing fractals."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='4h')
    closes = 100.0 + np.cumsum(np.random.normal(0, 2.0, 200))
    highs = closes + np.random.uniform(1.0, 3.0, 200)
    lows = closes - np.random.uniform(1.0, 3.0, 200)
    opens = (closes + np.roll(closes, 1)) / 2.0
    opens[0] = closes[0]

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.uniform(1000, 5000, 200)
    }, index=dates)
    return df


def test_chan_features_causality(synthetic_candles):
    """Verify that modifying bar t does not alter Chan-Lun features at bar t-1 or earlier."""
    base_feats = compute_chan_features(synthetic_candles)

    # Corrupt the last bar with a massive spike
    modified = synthetic_candles.copy()
    modified.iloc[-1, modified.columns.get_loc('close')] += 5000.0
    modified.iloc[-1, modified.columns.get_loc('high')] += 5000.0
    mod_feats = compute_chan_features(modified)

    # Indicators up to bar -2 must remain identical
    cols_to_check = ['chan_fractal_type', 'chan_bi_dir', 'chan_hub_dist', 'chan_hub_width']
    for c in cols_to_check:
        pd.testing.assert_series_equal(base_feats[c].iloc[:-2], mod_feats[c].iloc[:-2])


def test_hub_distance_bounds(synthetic_candles):
    """Verify that hub distance and width are bounded and contain zero NaNs."""
    feats = compute_chan_features(synthetic_candles)
    assert feats.isna().sum().sum() == 0, "Chan-Lun features must contain 0 NaNs"
    assert feats['chan_hub_width'].min() >= 0.01
    assert feats['chan_bi_bars'].min() >= 0.0


def test_chan_transformer_forward():
    """Verify tensor shape compatibility across multi-scale wave prediction heads."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CryptoSTChanTransformer(num_assets=4, in_features=41, lookback=18, d_model=32).to(device)

    dummy_x = torch.randn(8, 4, 18, 41).to(device)
    out = model(dummy_x)

    assert out['pred_wave_3d'].shape == (8, 4)
    assert out['pred_wave_6d'].shape == (8, 4)
    assert out['pred_wave_12d'].shape == (8, 4)
    assert out['prob_expansion'].shape == (8, 4)
    assert (out['prob_expansion'] >= 0.0).all() and (out['prob_expansion'] <= 1.0).all()


def test_chan_wave_combined_loss():
    """Verify combined multi-task loss calculation and backward pass stability."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CryptoSTChanTransformer(num_assets=4, in_features=41, lookback=18, d_model=32).to(device)
    criterion = ChanWaveCombinedLoss()

    dummy_x = torch.randn(8, 4, 18, 41).to(device)
    out = model(dummy_x)

    y_3d = torch.randn(8, 4).to(device) * 0.05
    y_6d = torch.randn(8, 4).to(device) * 0.08
    y_12d = torch.randn(8, 4).to(device) * 0.12
    y_exp = (y_12d > 0.02).float()

    loss, loss_dict = criterion(out, y_3d, y_6d, y_12d, y_exp)
    assert loss.item() > 0.0
    loss.backward()

    # Verify gradients exist for parameters
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_chan_transformer_hybrid_engine(synthetic_candles):
    """Verify ChanTransformerHybridEngine backtest execution and metric integrity."""
    from crypto_quant.chan_transformer_engine import ChanTransformerHybridEngine

    n = len(synthetic_candles)
    dates = synthetic_candles.index
    chan_feats = compute_chan_features(synthetic_candles)

    # Mock predictions
    mock_preds = pd.DataFrame({
        'ETHUSDT_pred_3d': np.random.normal(0.01, 0.05, n),
        'ETHUSDT_pred_6d': np.random.normal(0.02, 0.08, n),
        'ETHUSDT_pred_12d': np.random.normal(0.03, 0.10, n),
        'ETHUSDT_prob_exp': np.random.uniform(0.40, 0.60, n),
    }, index=dates)

    engine = ChanTransformerHybridEngine(
        token='ETHUSDT',
        atr_trailing_mult=3.0,
        exp_pct=50.0,
        pred_12d_threshold=0.0,
    )
    res = engine.backtest(synthetic_candles, chan_feats, mock_preds)

    assert 'metrics' in res
    assert 'bar_rets' in res
    assert 'equity_curve' in res
    assert res['equity_curve'].isna().sum() == 0
    assert len(res['bar_rets']) == n

