# -*- coding: utf-8 -*-
"""
Market Data Module: Binance Public REST 4h K-Line Pipeline (Phase 22)
====================================================================
Retrieves, cleans, strictly validates, and verifies 4-hour closed candles
from Binance Public REST endpoints without requiring any API keys.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
import requests
import pandas as pd
import numpy as np

from crypto_quant.paper.config import (
    BINANCE_PUBLIC_API_URLS,
    MAX_DATA_AGE_SECONDS,
    MINIMUM_WARMUP_BARS,
    TIMEFRAME,
)


class DataValidationError(Exception):
    """Raised when market data violates continuity, causality, or physical price constraints."""
    pass


class MarketDataFetcher:
    """
    Public market data fetcher and rigorous validator for 4h closed candles.
    """

    def __init__(
        self,
        base_urls: Optional[List[str]] = None,
        timeout_seconds: int = 10,
        max_data_age_seconds: int = MAX_DATA_AGE_SECONDS,
    ):
        self.base_urls = base_urls or BINANCE_PUBLIC_API_URLS
        self.timeout_seconds = timeout_seconds
        self.max_data_age_seconds = max_data_age_seconds

    def fetch_closed_klines(
        self,
        symbol: str,
        interval: str = "4h",
        limit: int = 250,
        check_freshness: bool = True,
    ) -> pd.DataFrame:
        """
        Fetches at least `limit` klines from Binance Public API, verifies closure,
        validates continuity and OHLC integrity, and returns cleaned DataFrame.

        Parameters:
            symbol (str): e.g. 'ETHUSDT' or 'SOLUSDT'
            interval (str): '4h'
            limit (int): Number of bars to fetch (>= 200)
            check_freshness (bool): Whether to enforce MAX_DATA_AGE_SECONDS

        Returns:
            pd.DataFrame: Index = open_time (pd.DatetimeIndex, UTC),
                          Columns = ['open', 'high', 'low', 'close', 'volume', 'close_time']
        """
        if limit < MINIMUM_WARMUP_BARS:
            raise DataValidationError(
                f"Requested limit {limit} is less than required minimum warmup {MINIMUM_WARMUP_BARS}"
            )

        raw_data = None
        last_error = None

        # Rotate through public Binance REST endpoints
        for base_url in self.base_urls:
            url = f"{base_url}/api/v3/klines"
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": limit,
            }
            try:
                resp = requests.get(url, params=params, timeout=self.timeout_seconds)
                if resp.status_code == 200:
                    raw_data = resp.json()
                    break
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
            except Exception as e:
                last_error = str(e)

        if raw_data is None:
            raise RuntimeError(
                f"FATAL: Failed to fetch {interval} klines for {symbol} from all Binance public mirrors. "
                f"Last error: {last_error}"
            )

        if not isinstance(raw_data, list) or len(raw_data) == 0:
            raise DataValidationError(f"Invalid empty response received from Binance API for {symbol}")

        # Parse raw Binance kline array:
        # [0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume, 6: close_time, ...]
        rows = []
        for item in raw_data:
            rows.append({
                "open_time": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "close_time": pd.to_datetime(int(item[6]), unit="ms", utc=True),
            })

        df = pd.DataFrame(rows)
        return self.validate_and_format_klines(
            df,
            symbol=symbol,
            interval=interval,
            check_freshness=check_freshness,
        )

    def validate_and_format_klines(
        self,
        df: pd.DataFrame,
        symbol: str = "ASSET",
        interval: str = "4h",
        check_freshness: bool = False,
        reference_time_utc: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Applies exhaustive institutional data integrity and continuity checks.
        Can be used both for live API responses and offline replay data.
        """
        if df.empty:
            raise DataValidationError(f"Empty candle DataFrame received for {symbol}")

        now_utc = reference_time_utc or datetime.now(timezone.utc)
        if not hasattr(now_utc, "tzinfo") or now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        # 1. Deduplicate by open_time
        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

        # 2. Check and drop unclosed live forming bar
        # A bar is closed if and only if now_utc >= close_time (or open_time + interval)
        if "close_time" in df.columns:
            # Drop bars whose close_time is still in the future relative to now_utc
            last_close_time = df.iloc[-1]["close_time"]
            if hasattr(last_close_time, "tzinfo") and last_close_time.tzinfo is None:
                last_close_time = last_close_time.replace(tzinfo=timezone.utc)
            if last_close_time > now_utc:
                df = df.iloc[:-1].copy()

        if len(df) < MINIMUM_WARMUP_BARS:
            raise DataValidationError(
                f"{symbol}: Available closed bars count {len(df)} is less than required {MINIMUM_WARMUP_BARS}"
            )

        # 3. Check for monotonic strictly increasing timestamps and missing intervals
        df = df.set_index("open_time").sort_index()
        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Verify exact 4-hour spacing
        expected_step = pd.Timedelta(hours=4) if interval == "4h" else pd.Timedelta(hours=1)
        time_diffs = df.index.to_series().diff().dropna()
        if not (time_diffs == expected_step).all():
            bad_diffs = time_diffs[time_diffs != expected_step]
            raise DataValidationError(
                f"{symbol}: Missing or non-contiguous bars detected at timestamps: {bad_diffs.index.tolist()[:3]}"
            )

        # 4. Check OHLC physical consistency
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        if (closes <= 0).any() or (opens <= 0).any() or (highs <= 0).any() or (lows <= 0).any():
            raise DataValidationError(f"{symbol}: Price contains non-positive values (close <= 0)")

        if (highs < np.maximum(opens, closes)).any():
            raise DataValidationError(f"{symbol}: Physical violation: High is strictly less than max(Open, Close)")

        if (lows > np.minimum(opens, closes)).any():
            raise DataValidationError(f"{symbol}: Physical violation: Low is strictly greater than min(Open, Close)")

        # 5. Data freshness check
        if check_freshness:
            # A 4h bar closed at T remains the valid latest closed bar until T + 4h.
            # Data is stale if now_utc exceeds T + 4h (when the NEXT bar should have closed) + max_data_age_seconds.
            last_closed_bar_close_time = df.index[-1] + expected_step
            next_expected_close_time = last_closed_bar_close_time + expected_step
            if now_utc > next_expected_close_time + timedelta(seconds=self.max_data_age_seconds):
                data_age_seconds = (now_utc - last_closed_bar_close_time).total_seconds()
                raise DataValidationError(
                    f"{symbol}: Market data feed is stale! "
                    f"Latest closed bar was {df.index[-1]} (closed at {last_closed_bar_close_time}), "
                    f"next bar should have closed by {next_expected_close_time}. "
                    f"Overdue age: {data_age_seconds:.1f}s exceeds limit {self.max_data_age_seconds}s"
                )

        return df[["open", "high", "low", "close", "volume"]]


# Standalone canonical helper
def fetch_closed_klines(
    symbol: str,
    interval: str = TIMEFRAME,
    limit: int = 250,
) -> pd.DataFrame:
    """Canonical function required by task specification."""
    fetcher = MarketDataFetcher()
    return fetcher.fetch_closed_klines(symbol=symbol, interval=interval, limit=limit, check_freshness=False)
