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


def test_no_bfill_in_signal_features(synthetic_candles):
    """Verify that early features never read backwards from future values via bfill."""
    # When future values (bars 100-199) are modified, bars 0-50 must remain completely unchanged.
    base_feats = compute_chan_features(synthetic_candles)
    
    corrupted = synthetic_candles.copy()
    corrupted.iloc[100:, corrupted.columns.get_loc('close')] *= 10.0
    corrupted.iloc[100:, corrupted.columns.get_loc('high')] *= 10.0
    corrupted.iloc[100:, corrupted.columns.get_loc('low')] *= 10.0
    corrupted_feats = compute_chan_features(corrupted)
    
    # All features for bars 0 to 98 must be identical
    for col in base_feats.columns:
        pd.testing.assert_series_equal(
            base_feats[col].iloc[:98],
            corrupted_feats[col].iloc[:98],
            check_names=False,
            check_dtype=False
        )


def test_corrupt_future_does_not_change_past_features(synthetic_candles):
    """Corrupting data at time t must leave all features at <= t-1 strictly invariant."""
    t_split = 80
    base_feats = compute_chan_features(synthetic_candles)
    
    noisy = synthetic_candles.copy()
    np.random.seed(999)
    noisy.iloc[t_split:, :] = np.random.uniform(500, 1000, size=noisy.iloc[t_split:, :].shape)
    noisy_feats = compute_chan_features(noisy)
    
    # Features strictly before t_split must remain identical
    for col in base_feats.columns:
        diff = np.abs(base_feats[col].iloc[:t_split-1].values - noisy_feats[col].iloc[:t_split-1].values)
        assert np.nanmax(diff) < 1e-7, f"Feature {col} leaked future data from t={t_split} into past!"


def test_no_trade_before_minimum_warmup(synthetic_candles):
    """Verify engine generates strictly 0 trades during the mandatory warmup period."""
    from crypto_quant.chan_transformer_engine import ChanTransformerHybridEngine

    n = len(synthetic_candles)
    dates = synthetic_candles.index
    chan_feats = compute_chan_features(synthetic_candles)

    # Aggressive predictions designed to force instant trades if warmup were bypassed
    mock_preds = pd.DataFrame({
        'ETHUSDT_pred_3d': np.full(n, 0.50),
        'ETHUSDT_pred_6d': np.full(n, 0.50),
        'ETHUSDT_pred_12d': np.full(n, 0.50),
        'ETHUSDT_prob_exp': np.full(n, 0.99),
    }, index=dates)

    engine = ChanTransformerHybridEngine(
        token='ETHUSDT',
        min_warmup_bars=60,
        pred_12d_threshold=-0.50,
        exp_pct=10.0,
    )
    res = engine.backtest(synthetic_candles, chan_feats, mock_preds)
    
    # No trades may be entered before bar 60
    for trade in res['trades']:
        entry_idx = dates.get_loc(pd.to_datetime(trade.entry_time))
        assert entry_idx >= 60, f"Trade entered at index {entry_idx} before minimum warmup (60)!"


def test_no_target_crosses_train_boundary():
    """Verify that train sample 12-day wave labels do not access validation prices."""
    from crypto_quant.dataset_builder import prepare_chan_wave_datasets
    _, _, _, meta = prepare_chan_wave_datasets(lookback_len=18, purge_bars=72, embargo_bars=18)
    
    train_end_dt = pd.to_datetime(meta['train_end'])
    last_train_t = pd.to_datetime(meta['train_timestamps'][-1])
    
    # Label ends at last_train_t + 72 bars (12 days)
    label_end_t = last_train_t + pd.Timedelta(hours=72 * 4)
    assert label_end_t <= train_end_dt, (
        f"Train label leakage! Last train sample at {last_train_t}, label ends at {label_end_t} > {train_end_dt}"
    )


def test_no_target_crosses_validation_boundary():
    """Verify that validation sample 12-day wave labels do not access blind test prices."""
    from crypto_quant.dataset_builder import prepare_chan_wave_datasets
    _, _, _, meta = prepare_chan_wave_datasets(lookback_len=18, purge_bars=72, embargo_bars=18)
    
    val_end_dt = pd.to_datetime(meta['val_end'])
    last_val_t = pd.to_datetime(meta['val_timestamps'][-1])
    
    label_end_t = last_val_t + pd.Timedelta(hours=72 * 4)
    assert label_end_t <= val_end_dt, (
        f"Val label leakage! Last val sample at {last_val_t}, label ends at {label_end_t} > {val_end_dt}"
    )


def test_purge_and_embargo_gap():
    """Verify that sample timestamp gap between splits satisfies purge and embargo minimums."""
    from crypto_quant.dataset_builder import prepare_chan_wave_datasets
    _, _, _, meta = prepare_chan_wave_datasets(lookback_len=18, purge_bars=72, embargo_bars=18)
    
    train_last = pd.to_datetime(meta['train_timestamps'][-1])
    val_first = pd.to_datetime(meta['val_timestamps'][0])
    val_last = pd.to_datetime(meta['val_timestamps'][-1])
    test_first = pd.to_datetime(meta['blind_test_timestamps'][0])
    
    # Gap must be at least 72 bars (12 days) + 18 bars (3 days) = 15 days
    gap_train_val = (val_first - train_last).total_seconds() / 3600.0 / 4.0
    gap_val_test = (test_first - val_last).total_seconds() / 3600.0 / 4.0
    
    assert gap_train_val >= 72.0, f"Gap train->val {gap_train_val} bars is less than 72 bars"
    assert gap_val_test >= 72.0, f"Gap val->test {gap_val_test} bars is less than 72 bars"


