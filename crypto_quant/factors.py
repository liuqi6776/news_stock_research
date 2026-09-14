"""
Crypto Technical Factors & Indicators / 加密资产技术指标与因子计算模块
Tailored for 24/7 continuous cryptocurrency markets.
专门针对 24/7 不间断加密市场设计的技术因子与指标库。
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

def add_moving_averages(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26) -> pd.DataFrame:
    """Add Exponential Moving Averages (EMA)."""
    df = df.copy()
    df[f"ema_{fast_period}"] = df["close"].ewm(span=fast_period, adjust=False).mean()
    df[f"ema_{slow_period}"] = df["close"].ewm(span=slow_period, adjust=False).mean()
    df[f"sma_{fast_period}"] = df["close"].rolling(window=fast_period).mean()
    df[f"sma_{slow_period}"] = df["close"].rolling(window=slow_period).mean()
    return df

def add_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    df = df.copy()
    fast_ema = df["close"].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow_period, adjust=False).mean()
    df["macd"] = fast_ema - slow_ema
    df["macd_signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index (RSI)."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df[f"rsi_{period}"] = 100.0 - (100.0 / (1.0 + rs))
    return df

def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.DataFrame:
    """Calculate Bollinger Bands, Bandwidth, and %B."""
    df = df.copy()
    mid = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()

    df["bb_mid"] = mid
    df["bb_upper"] = mid + (std * std_dev)
    df["bb_lower"] = mid - (std * std_dev)
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / (mid + 1e-10)
    df["bb_percent_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    return df

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Average True Range (ATR)."""
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["tr"] = tr
    df[f"atr_{period}"] = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return df

def add_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Calculate SuperTrend indicator.
    Returns trend direction (+1 for Bullish, -1 for Bearish) and supertrend line.
    """
    df = add_atr(df, period=period)
    atr_col = f"atr_{period}"

    hl2 = (df["high"] + df["low"]) / 2.0
    basic_upper = hl2 + (multiplier * df[atr_col])
    basic_lower = hl2 - (multiplier * df[atr_col])

    upper_band = basic_upper.copy()
    lower_band = basic_lower.copy()
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)

    close_arr = df["close"].values
    upper_arr = basic_upper.values
    lower_arr = basic_lower.values
    dir_arr = np.ones(len(df), dtype=int)
    st_arr = np.zeros(len(df), dtype=float)

    for i in range(1, len(df)):
        # Final lower band
        if lower_arr[i] > lower_arr[i - 1] or close_arr[i - 1] < lower_arr[i - 1]:
            pass
        else:
            lower_arr[i] = lower_arr[i - 1]

        # Final upper band
        if upper_arr[i] < upper_arr[i - 1] or close_arr[i - 1] > upper_arr[i - 1]:
            pass
        else:
            upper_arr[i] = upper_arr[i - 1]

        # Trend direction
        if dir_arr[i - 1] == 1:
            if close_arr[i] < lower_arr[i]:
                dir_arr[i] = -1
                st_arr[i] = upper_arr[i]
            else:
                dir_arr[i] = 1
                st_arr[i] = lower_arr[i]
        else:
            if close_arr[i] > upper_arr[i]:
                dir_arr[i] = 1
                st_arr[i] = lower_arr[i]
            else:
                dir_arr[i] = -1
                st_arr[i] = upper_arr[i]

    df["supertrend"] = st_arr
    df["supertrend_dir"] = dir_arr
    return df

def add_crypto_market_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate crypto-specific market microstructure factors:
    - Taker Buy Ratio (Active Buying Pressure)
    - Volume Surge Ratio
    - Rolling Realized Volatility
    """
    df = df.copy()
    if "taker_buy_volume" in df.columns and "volume" in df.columns:
        df["taker_buy_ratio"] = df["taker_buy_volume"] / (df["volume"] + 1e-10)
    else:
        df["taker_buy_ratio"] = 0.5

    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["vol_sma_20"] + 1e-10)

    # Log returns and annualized realized volatility (assuming 1h data: 365 * 24 periods/yr)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol_24h"] = df["log_ret"].rolling(24).std() * np.sqrt(365 * 24)
    return df

def compute_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full suite of technical indicators and crypto factors."""
    df = add_moving_averages(df, fast_period=12, slow_period=26)
    df = add_macd(df)
    df = add_rsi(df, period=14)
    df = add_bollinger_bands(df, period=20, std_dev=2.0)
    df = add_atr(df, period=14)
    df = add_supertrend(df, period=10, multiplier=3.0)
    df = add_crypto_market_factors(df)
    return df
