"""
Crypto Quantitative Strategies / 加密货币经典量化策略库
Implements trend following, volatility breakout, RSI momentum, and SuperTrend.
涵盖双均线趋势跟踪、布林带波动率突破、RSI动量以及 SuperTrend 经典加密量化策略。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from crypto_quant.factors import (
    add_moving_averages,
    add_bollinger_bands,
    add_rsi,
    add_supertrend,
    add_atr,
    add_crypto_market_factors
)

def dual_ema_trend_strategy(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    atr_period: int = 14,
    atr_stop_multiplier: float = 2.5
) -> pd.Series:
    """
    Dual EMA Trend Following Strategy with ATR Trailing Stop.
    双均线趋势跟踪策略（结合 ATR 动态跟踪止损）。
    
    Logic:
    - Long when Fast EMA crosses above Slow EMA.
    - Flat when Fast EMA crosses below Slow EMA or trailing stop is breached.
    """
    data = add_moving_averages(df, fast_period=fast_period, slow_period=slow_period)
    data = add_atr(data, period=atr_period)

    fast_col = f"ema_{fast_period}"
    slow_col = f"ema_{slow_period}"
    atr_col = f"atr_{atr_period}"

    signals = pd.Series(0, index=data.index)
    in_position = 0
    stop_price = 0.0

    for i in range(1, len(data)):
        cur_close = data["close"].iloc[i]
        cur_fast = data[fast_col].iloc[i]
        cur_slow = data[slow_col].iloc[i]
        prev_fast = data[fast_col].iloc[i - 1]
        prev_slow = data[slow_col].iloc[i - 1]
        cur_atr = data[atr_col].iloc[i]

        # Check entry: Golden Cross
        if in_position == 0:
            if cur_fast > cur_slow and prev_fast <= prev_slow:
                in_position = 1
                stop_price = cur_close - (atr_stop_multiplier * cur_atr)
        elif in_position == 1:
            # Trailing stop update
            new_stop = cur_close - (atr_stop_multiplier * cur_atr)
            if new_stop > stop_price:
                stop_price = new_stop

            # Check exit: Death Cross or Stop Loss breached
            if cur_close < stop_price or (cur_fast < cur_slow and prev_fast >= prev_slow):
                in_position = 0

        signals.iloc[i] = in_position

    return signals

def bollinger_breakout_strategy(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    bandwidth_threshold: float = 0.03
) -> pd.Series:
    """
    Bollinger Bands Volatility Squeeze Breakout Strategy.
    布林带波动率收缩突破策略。

    Logic:
    - Identifies squeeze when bandwidth is expanding.
    - Buy when Close breaks above Upper Band.
    - Exit when Close falls back below Middle Band.
    """
    data = add_bollinger_bands(df, period=period, std_dev=std_dev)
    signals = pd.Series(0, index=data.index)
    in_position = 0

    for i in range(1, len(data)):
        cur_close = data["close"].iloc[i]
        prev_close = data["close"].iloc[i - 1]
        cur_upper = data["bb_upper"].iloc[i]
        prev_upper = data["bb_upper"].iloc[i - 1]
        cur_mid = data["bb_mid"].iloc[i]
        cur_bw = data["bb_bandwidth"].iloc[i]

        if in_position == 0:
            # Breakout with sufficient bandwidth / volatility
            if cur_close > cur_upper and prev_close <= prev_upper and cur_bw > bandwidth_threshold:
                in_position = 1
        elif in_position == 1:
            if cur_close < cur_mid:
                in_position = 0

        signals.iloc[i] = in_position

    return signals

def rsi_momentum_strategy(
    df: pd.DataFrame,
    rsi_period: int = 14,
    oversold_entry: float = 35.0,
    overbought_exit: float = 70.0,
    use_volume_filter: bool = True
) -> pd.Series:
    """
    RSI Adaptive Momentum & Reversion Strategy.
    RSI 动量修复与反转策略（结合主动买盘比例过滤）。

    Logic:
    - Long when RSI crosses back above oversold threshold (indicating rebound momentum).
    - If volume filter is enabled, require taker buy ratio > 0.48.
    - Exit when RSI touches overbought territory or momentum falters.
    """
    data = add_rsi(df, period=rsi_period)
    data = add_crypto_market_factors(data)

    rsi_col = f"rsi_{rsi_period}"
    signals = pd.Series(0, index=data.index)
    in_position = 0

    for i in range(1, len(data)):
        cur_rsi = data[rsi_col].iloc[i]
        prev_rsi = data[rsi_col].iloc[i - 1]
        taker_ratio = data["taker_buy_ratio"].iloc[i]

        if in_position == 0:
            cond_rebound = prev_rsi < oversold_entry and cur_rsi >= oversold_entry
            cond_volume = (taker_ratio > 0.48) if use_volume_filter else True
            if cond_rebound and cond_volume:
                in_position = 1
        elif in_position == 1:
            if cur_rsi >= overbought_exit or cur_rsi < 30.0:
                in_position = 0

        signals.iloc[i] = in_position

    return signals

def supertrend_strategy(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0
) -> pd.Series:
    """
    SuperTrend Trend-Following Strategy.
    超级趋势指标策略。

    Logic:
    - Long when SuperTrend direction is +1 (Bullish).
    - Flat when SuperTrend direction is -1 (Bearish).
    """
    data = add_supertrend(df, period=period, multiplier=multiplier)
    # 1 for bullish, 0 for flat
    signals = (data["supertrend_dir"] == 1).astype(int)
    return signals
