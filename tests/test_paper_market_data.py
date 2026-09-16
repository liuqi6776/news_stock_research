# -*- coding: utf-8 -*-
"""
Unit Tests: Market Data Validation & Candle Hygiene (Phase 22)
==============================================================
Verifies:
1. test_only_closed_candle_is_used
2. test_missing_bar_is_detected
3. test_duplicate_bar_is_rejected
4. test_data_age_limit
5. test_warmup_requires_200_bars
6. test_no_bfill
"""

from datetime import datetime, timezone, timedelta
import inspect
import numpy as np
import pandas as pd
import pytest

from crypto_quant.paper.market_data import DataValidationError, MarketDataFetcher
import crypto_quant.paper.market_data as md_module
import crypto_quant.paper.strategy as strat_module


def create_synthetic_4h_candles(n_bars: int = 250, start_time: str = "2024-01-01 00:00:00") -> pd.DataFrame:
    """Helper creating clean synthetic 4h OHLCV data."""
    start_dt = pd.to_datetime(start_time, utc=True)
    times = [start_dt + pd.Timedelta(hours=4 * i) for i in range(n_bars)]
    closes = 2000.0 + np.cumsum(np.random.normal(0, 10, n_bars))
    highs = closes + 15.0
    lows = closes - 15.0
    opens = closes - 2.0
    vols = np.random.uniform(100, 500, n_bars)

    df = pd.DataFrame({
        "open_time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
        "close_time": [t + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1) for t in times],
    })
    return df


def test_only_closed_candle_is_used():
    """Unclosed forming candle (close_time > now_utc) must be strictly discarded."""
    df = create_synthetic_4h_candles(n_bars=250)
    fetcher = MarketDataFetcher()

    # Set reference_time such that the last bar has NOT closed yet
    last_bar_open = df.iloc[-1]["open_time"]
    ref_time = last_bar_open + timedelta(hours=2)  # 2 hours into 4h bar -> bar is NOT closed

    cleaned = fetcher.validate_and_format_klines(
        df,
        symbol="ETHUSDT",
        reference_time_utc=ref_time,
    )
    # The last unclosed bar must be dropped
    assert len(cleaned) == 249
    assert cleaned.index[-1] == df.iloc[-2]["open_time"]


def test_missing_bar_is_detected():
    """Missing or skipped 4h bar must raise DataValidationError."""
    df = create_synthetic_4h_candles(n_bars=250)
    # Drop index 150 to create an 8-hour gap
    df_missing = df.drop(index=[150]).reset_index(drop=True)

    fetcher = MarketDataFetcher()
    with pytest.raises(DataValidationError, match="Missing or non-contiguous bars detected"):
        fetcher.validate_and_format_klines(df_missing, symbol="ETHUSDT")


def test_duplicate_bar_is_rejected():
    """Duplicate timestamp rows must be deduplicated cleanly."""
    df = create_synthetic_4h_candles(n_bars=250)
    # Duplicate row 50
    dup_row = df.iloc[[50]]
    df_dup = pd.concat([df, dup_row]).sort_values("open_time").reset_index(drop=True)

    fetcher = MarketDataFetcher()
    cleaned = fetcher.validate_and_format_klines(df_dup, symbol="ETHUSDT")
    # Must deduplicate and have exactly 250 bars
    assert len(cleaned) == 250
    assert not cleaned.index.duplicated().any()


def test_data_age_limit():
    """Data where latest closed bar is older than MAX_DATA_AGE_SECONDS must raise error."""
    df = create_synthetic_4h_candles(n_bars=250)
    fetcher = MarketDataFetcher(max_data_age_seconds=300)

    # Reference time is 2 days in the future
    ref_future = df.iloc[-1]["close_time"] + timedelta(days=2)

    with pytest.raises(DataValidationError, match="is stale"):
        fetcher.validate_and_format_klines(
            df,
            symbol="ETHUSDT",
            check_freshness=True,
            reference_time_utc=ref_future,
        )


def test_warmup_requires_200_bars():
    """Fewer than 200 bars must raise DataValidationError."""
    df = create_synthetic_4h_candles(n_bars=150)
    fetcher = MarketDataFetcher()

    with pytest.raises(DataValidationError, match="Available closed bars count 150 is less than required 200"):
        fetcher.validate_and_format_klines(df, symbol="ETHUSDT")


def test_no_bfill():
    """Verifies that no bfill is present in market data or paper strategy modules."""
    for mod in [md_module, strat_module]:
        source = inspect.getsource(mod)
        assert "bfill" not in source, f"Forbidden 'bfill' detected in {mod.__name__}"
