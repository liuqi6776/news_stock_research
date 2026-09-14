# -*- coding: utf-8 -*-
"""
Point-in-Time Data Aligner (Phase 17)
Enforces strict causal Point-in-Time (PIT) data alignment for external daily features:
1. Alternative.me Fear & Greed Index (FNG)
2. DefiLlama on-chain TVL & stablecoin capital flows
3. US stock macro indices (Nasdaq/S&P 500)

Rule:
Daily data published for day D is not knowable at 00:00:00 UTC of day D.
Applying lag_days=1 (shift(1).ffill()) ensures that bar t only reads data
finalized before bar t's timestamp, eliminating look-ahead leakage.
"""

from typing import Dict, Union, Optional
import pandas as pd
import numpy as np


def get_lag_policy() -> Dict[str, str]:
    """
    Returns the immutable system-wide Point-in-Time lag policy dictionary.
    """
    return {
        "fng": "1d_lag_ffill",
        "macro": "1d_lag_ffill",
        "onchain": "1d_lag_ffill",
        "funding": "1_bar_lag_ffill",
    }


def align_daily_features_to_4h(
    df_or_series: Union[pd.DataFrame, pd.Series],
    target_4h_index: pd.DatetimeIndex,
    lag_days: int = 1,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Aligns daily features onto a target 4h DatetimeIndex with causal lag.
    
    Parameters
    ----------
    df_or_series : pd.DataFrame or pd.Series
        Daily data indexed by datetime (typically 00:00:00 UTC).
    target_4h_index : pd.DatetimeIndex
        Target 4h bar timestamps.
    lag_days : int
        Number of days to shift forward before forward-filling (default=1).
        
    Returns
    -------
    pd.DataFrame or pd.Series
        Reindexed and causally forward-filled data on target_4h_index.
    """
    if df_or_series is None or len(df_or_series) == 0:
        return df_or_series

    data = df_or_series.copy()

    # Ensure tz-naive for consistent index matching
    if hasattr(data.index, "tz") and data.index.tz is not None:
        data.index = data.index.tz_convert(None)
    if hasattr(target_4h_index, "tz") and target_4h_index.tz is not None:
        target_4h_index = target_4h_index.tz_convert(None)

    # Shift by lag_days if positive
    if lag_days > 0:
        data = data.shift(lag_days)

    # Reindex onto union of existing and target timestamps, then forward-fill
    union_idx = data.index.union(target_4h_index).sort_values()
    reindexed = data.reindex(union_idx).ffill()

    # Slice strictly to target_4h_index
    aligned = reindexed.loc[target_4h_index]
    return aligned
