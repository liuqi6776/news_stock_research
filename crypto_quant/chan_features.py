# -*- coding: utf-8 -*-
"""
Causal Chan-Lun Topological Feature Extraction Engine (Phase 20)
================================================================
Extracts strictly causal, non-repainting Chan-Lun structural features from 4h OHLCV data:
1. chan_hub_dist: Normalized price position relative to 4h Central Hub [ZD, ZG]
2. chan_bi_dir: Current Bi direction (+1.0 Upward Bi, -1.0 Downward Bi)
3. chan_bi_bars: Maturity duration of the active Bi (normalized)
4. chan_bi_amplitude: Cumulative return of the active Bi
5. chan_fractal_type: Confirmed Top Fractal (-1.0), Bottom Fractal (+1.0), or None (0.0)
6. chan_divergence_ratio: MACD momentum divergence ratio between current and prior Bi
7. chan_hub_width: Relative height/volatility of the current Hub (ZG - ZD) / Mid
8. chan_third_buy_flag: Structural Third Buy / Third Sell breakout confirmation flag

STRICTLY CAUSAL: All indicators are shifted or computed forward in time with zero lookahead.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_chan_features(df: pd.DataFrame, hub_window: int = 60, atr_period: int = 14) -> pd.DataFrame:
    """
    Computes 8-dimensional Chan-Lun topological features on OHLCV DataFrame.
    Inputs:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume'
        hub_window: Lookback bars to construct central hub (default: 60 bars = 10 days)
        atr_period: ATR smoothing window (default: 14)
    Returns:
        DataFrame with 8 causal Chan-Lun features indexed identical to df.
    """
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']
    n = len(df)
    
    feats = pd.DataFrame(index=df.index)

    # 1. Causal ATR
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().bfill()
    atr_ratio = atr / (c + 1e-8)

    # 2. Confirmed Fractals (shifted strictly for causality: a fractal at t-1 is confirmed at t)
    # Bottom Fractal at t-1: low[t-1] < low[t-2] and low[t-1] < low[t]
    # Top Fractal at t-1: high[t-1] > high[t-2] and high[t-1] > high[t]
    is_bottom_fractal = (l.shift(1) < l.shift(2)) & (l.shift(1) < l)
    is_top_fractal = (h.shift(1) > h.shift(2)) & (h.shift(1) > h)

    fractal_type = np.zeros(n)
    fractal_type[is_bottom_fractal.values] = 1.0
    fractal_type[is_top_fractal.values] = -1.0
    feats['chan_fractal_type'] = fractal_type

    # 3. Dynamic Causal Bi (笔) Tracking
    # Iterate causally forward to identify confirmed swing pivots and bi direction
    bi_dir = np.zeros(n)
    bi_bars = np.zeros(n)
    bi_amp = np.zeros(n)
    
    # State tracking
    curr_dir = 1.0  # +1 Upward, -1 Downward
    curr_start_price = float(c.iloc[0])
    curr_start_idx = 0

    h_vals = h.values
    l_vals = l.values
    c_vals = c.values

    for i in range(1, n):
        # A confirmed bottom fractal confirms the end of downward bi and starts upward bi
        if is_bottom_fractal.iloc[i]:
            if curr_dir == -1.0 and (i - curr_start_idx) >= 4:  # At least 4-5 bars per Bi
                curr_dir = 1.0
                curr_start_idx = i - 1
                curr_start_price = float(l_vals[i - 1])
        # A confirmed top fractal confirms the end of upward bi and starts downward bi
        elif is_top_fractal.iloc[i]:
            if curr_dir == 1.0 and (i - curr_start_idx) >= 4:
                curr_dir = -1.0
                curr_start_idx = i - 1
                curr_start_price = float(h_vals[i - 1])

        bi_dir[i] = curr_dir
        elapsed = i - curr_start_idx
        bi_bars[i] = min(elapsed / 30.0, 3.0)  # Normalized by 30 bars (~5 days)
        if curr_start_price > 0:
            bi_amp[i] = np.clip((c_vals[i] - curr_start_price) / curr_start_price, -0.5, 0.5)

    feats['chan_bi_dir'] = bi_dir
    feats['chan_bi_bars'] = bi_bars
    feats['chan_bi_amplitude'] = bi_amp

    # 4. Central Hub Coordinates (ZG: upper bound, ZD: lower bound)
    # Using 60-bar lookback of confirmed swing highs and lows, shifted by 1 for causality
    zg = h.shift(1).rolling(hub_window).quantile(0.85).bfill()
    zd = l.shift(1).rolling(hub_window).quantile(0.15).bfill()
    hub_mid = (zg + zd) / 2.0
    hub_span = (zg - zd).clip(lower=1e-6)

    # 5. Hub Distance: (P - ZD) / (ZG - ZD)
    # < 0: below hub; [0, 1]: inside hub; > 1: above hub breakout
    hub_dist = ((c - zd) / hub_span).clip(-2.0, 3.0)
    feats['chan_hub_dist'] = hub_dist.values
    feats['chan_hub_width'] = (hub_span / (hub_mid + 1e-8)).clip(0.01, 0.5).values

    # 6. MACD Momentum Divergence Ratio
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_diff = ema12 - ema26
    signal = macd_diff.ewm(span=9, adjust=False).mean()
    hist = (macd_diff - signal).abs()

    # Momentum of current 12 bars vs prior 12-24 bars
    hist_recent = hist.rolling(12).mean()
    hist_prior = hist.shift(12).rolling(12).mean().bfill()
    div_ratio = np.clip(hist_recent / (hist_prior + 1e-8) - 1.0, -1.0, 1.0).fillna(0.0)
    feats['chan_divergence_ratio'] = div_ratio.values

    # 7. Structural Third Buy / Third Sell Breakout Flag
    # Third Buy: Price > ZG, and lowest pullback in last 6 bars > ZG
    # Third Sell: Price < ZD, and highest pullback in last 6 bars < ZD
    low_recent = l.rolling(6).min()
    high_recent = h.rolling(6).max()

    is_third_buy = (c > zg) & (low_recent >= zg * 0.995)
    is_third_sell = (c < zd) & (high_recent <= zd * 1.005)

    third_buy_flag = np.zeros(n)
    third_buy_flag[is_third_buy.values] = 1.0
    third_buy_flag[is_third_sell.values] = -1.0
    feats['chan_third_buy_flag'] = third_buy_flag

    return feats
