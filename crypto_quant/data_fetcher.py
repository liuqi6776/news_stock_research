"""
Binance Public Data Fetcher / 币安公共行情数据抓取模块
Supports Spot & USDS-M Futures without requiring API credentials.
支持无需 API Key 的币安现货与 U 本位合约行情与历史 K 线拉取。
"""

import json
import time
import os
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"

# Default request headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def _http_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15, retries: int = 3) -> Any:
    """Execute HTTP GET request with retries and timeout."""
    if params:
        query_string = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        full_url = f"{url}?{query_string}"
    else:
        full_url = url

    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(full_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read().decode("utf-8")
                return json.loads(content)
        except Exception as e:
            last_error = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {full_url} after {retries} attempts: {last_error}")

def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    market_type: str = "spot"
) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data from Binance public API.
    从币安公共接口获取 OHLCV K线数据。

    Parameters:
    -----------
    symbol: str
        Trading pair symbol, e.g., 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
    interval: str
        Kline interval: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w'
    limit: int
        Number of candles per request (default 500, max 1000)
    start_time: int, optional
        Start timestamp in milliseconds
    end_time: int, optional
        End timestamp in milliseconds
    market_type: str
        'spot' for spot market, 'futures' for USDS-M perpetual futures

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by open_time with numeric columns: open, high, low, close, volume, quote_volume, count, etc.
    """
    base_url = SPOT_BASE_URL if market_type.lower() == "spot" else FUTURES_BASE_URL
    endpoint = f"{base_url}/api/v3/klines" if market_type.lower() == "spot" else f"{base_url}/fapi/v1/klines"

    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, 1000)
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    raw_data = _http_get(endpoint, params=params)

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ]

    df = pd.DataFrame(raw_data, columns=columns)
    if df.empty:
        return df

    # Type conversions
    numeric_cols = [
        "open", "high", "low", "close", "volume", "quote_volume",
        "taker_buy_volume", "taker_buy_quote_volume"
    ]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    df["count"] = df["count"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)

    return df

def fetch_history_klines_paginated(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    total_candles: int = 1500,
    market_type: str = "spot"
) -> pd.DataFrame:
    """
    Fetch more than 1000 historical candles by backward pagination.
    通过向后分页拉取大量历史 K 线数据。
    """
    all_dfs = []
    end_time = None
    remaining = total_candles

    while remaining > 0:
        batch_size = min(remaining, 1000)
        batch_df = fetch_klines(
            symbol=symbol,
            interval=interval,
            limit=batch_size,
            end_time=end_time,
            market_type=market_type
        )
        if batch_df.empty:
            break

        all_dfs.append(batch_df)
        remaining -= len(batch_df)

        first_timestamp_ms = int(batch_df.index[0].timestamp() * 1000)
        end_time = first_timestamp_ms - 1

        if len(batch_df) < batch_size:
            break
        time.sleep(0.1)

    if not all_dfs:
        return pd.DataFrame()

    full_df = pd.concat(reversed(all_dfs)).drop_duplicates()
    full_df.sort_index(inplace=True)
    return full_df.tail(total_candles)

def fetch_24h_ticker(symbol: str = "BTCUSDT", market_type: str = "spot") -> Dict[str, Any]:
    """Fetch 24-hour price change statistics."""
    base_url = SPOT_BASE_URL if market_type.lower() == "spot" else FUTURES_BASE_URL
    endpoint = f"{base_url}/api/v3/ticker/24hr" if market_type.lower() == "spot" else f"{base_url}/fapi/v1/ticker/24hr"
    return _http_get(endpoint, params={"symbol": symbol.upper()})

def fetch_funding_rate_history(symbol: str = "BTCUSDT", limit: int = 100) -> pd.DataFrame:
    """Fetch funding rate history for USDS-M Futures."""
    endpoint = f"{FUTURES_BASE_URL}/fapi/v1/fundingRate"
    data = _http_get(endpoint, params={"symbol": symbol.upper(), "limit": limit})
    df = pd.DataFrame(data)
    if not df.empty:
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["fundingRate"] = df["fundingRate"].astype(float)
        df.set_index("fundingTime", inplace=True)
        df.sort_index(inplace=True)
    return df

def fetch_full_history(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    start_date: str = "2017-08-17",
    use_cache: bool = True,
    cache_dir: str = "data/crypto_cache"
) -> pd.DataFrame:
    """
    Fetch multi-year historical K-lines from Binance since inception, with parquet caching.
    拉取币安上线以来的多年跨周期历史 K 线数据，支持本地 Parquet 极速缓存。
    """
    os.makedirs(cache_dir, exist_ok=True)
    clean_symbol = symbol.upper()
    cache_file = os.path.join(cache_dir, f"{clean_symbol}_{interval}_full.parquet")

    if use_cache and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception:
            pass

    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    all_data = []
    cur_ts = start_ts

    while True:
        url = f"{SPOT_BASE_URL}/api/v3/klines?symbol={clean_symbol}&interval={interval}&startTime={cur_ts}&limit=1000"
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                batch = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            time.sleep(1.0)
            break

        if not batch:
            break

        all_data.extend(batch)
        last_ts = batch[-1][0]
        if len(batch) < 1000:
            break

        interval_ms = 86400000 if interval == "1d" else 3600000
        cur_ts = last_ts + interval_ms
        time.sleep(0.15)

    if not all_data:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(all_data, columns=cols)
    num_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]
    for c in num_cols:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    if use_cache:
        df.to_parquet(cache_file)

    return df
