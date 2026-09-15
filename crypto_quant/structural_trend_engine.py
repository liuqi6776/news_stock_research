# -*- coding: utf-8 -*-
"""
Structural Trend & Wave Tracking Engine (Phase 19)
===================================================
Institutional Macro Wave & Chan Lun Structural Trend Following Architecture.

Key Principles:
1. Long Holding Horizon: 10 to 45 days (capturing multi-week macro expansions).
2. Low Trade Frequency: 10 to 25 trades per year (drastically cutting transaction friction).
3. Asymmetric Payoff: High win/loss ratio (3:1 to 5:1) by letting profits run and cutting structural losses.
4. Monotonic Trailing Stop: Trailing stop level ratchets up with highest highs and ATR bands.
5. Transformer Macro Multiplier: Repurposes Transformer predictions as macro regime multipliers
   rather than high-frequency scalping triggers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    token: str
    direction: str  # 'LONG' or 'SHORT'
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    size: float
    duration_hours: float
    duration_days: float
    gross_ret: float
    net_ret: float
    funding_carry: float
    exit_reason: str  # 'TRAILING_STOP', 'CHANNEL_EXIT', 'REGIME_REVERSAL', 'SWING_BREAK'


class StructuralTrendEngine:
    """
    Macro Wave Trend & Chan Lun Structural Trading Engine.
    
    Supports:
    - Channel modes: 'donchian', 'bollinger', 'chan_swing', 'dual_ema'
    - Trailing stop: ATR trailing stop, structural swing low trailing stop
    - Macro regime multiplier: scaling position size based on Transformer z-scores and on-chain sentiment
    """

    def __init__(
        self,
        mode: str = 'chan_swing',
        lookback_bars: int = 120,       # 120 bars of 4h = 20 days
        exit_lookback_bars: int = 60,   # 60 bars of 4h = 10 days
        atr_period: int = 14,
        atr_trailing_mult: float = 3.0, # 3.0x ATR trailing stop
        fee_and_slippage: float = 0.0008, # 8 bps one-way (16 bps roundtrip)
        use_short: bool = False,
        base_size: float = 1.0,
        min_warmup_bars: Optional[int] = None,
    ):
        self.mode = mode
        self.lookback_bars = lookback_bars
        self.exit_lookback_bars = exit_lookback_bars
        self.atr_period = atr_period
        self.atr_trailing_mult = atr_trailing_mult
        self.fee_and_slippage = fee_and_slippage
        self.use_short = use_short
        self.base_size = base_size
        self.min_warmup_bars = min_warmup_bars if min_warmup_bars is not None else self.lookback_bars

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all trend, channel, and trailing stop indicators.
        STRICTLY CAUSAL: all indicators are computed and shifted so that bar t
        only sees information up to bar t (or bar t-1 as appropriate).
        Zero bfill: early bars use expanding averages or remain NaN.
        """
        res = pd.DataFrame(index=df.index)
        closes = df['close']
        highs = df['high']
        lows = df['low']

        # 1. ATR (Average True Range - strictly causal, no bfill)
        tr1 = highs - lows
        tr2 = (highs - closes.shift(1)).abs()
        tr3 = (lows - closes.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        res['atr'] = tr.rolling(self.atr_period, min_periods=1).mean()
        res['atr_ratio'] = res['atr'] / (closes + 1e-8)

        # 2. Donchian Channels (shifted by 1 for causality)
        res['donchian_high'] = highs.shift(1).rolling(self.lookback_bars).max()
        res['donchian_low'] = lows.shift(1).rolling(self.exit_lookback_bars).min()
        res['donchian_break_short'] = lows.shift(1).rolling(self.lookback_bars).min()

        # 3. Bollinger Bands (shifted by 1 for causality)
        bb_mid = closes.shift(1).rolling(self.lookback_bars).mean()
        bb_std = closes.shift(1).rolling(self.lookback_bars).std()
        res['bb_mid'] = bb_mid
        res['bb_upper'] = bb_mid + 2.0 * bb_std
        res['bb_lower'] = bb_mid - 2.0 * bb_std

        # 4. Chan Lun Structural Swing Pivots (顶底分型与中枢结构)
        pivot_win = max(15, self.lookback_bars // 4)  # e.g. 30 bars (~5 days)
        res['swing_high'] = highs.shift(1).rolling(pivot_win).max()
        res['swing_low'] = lows.shift(1).rolling(pivot_win).min()
        res['swing_mid'] = (res['swing_high'] + res['swing_low']) / 2.0

        # 5. Moving Averages for Macro Trend
        res['ema20'] = closes.shift(1).ewm(span=20).mean()
        res['ema120'] = closes.shift(1).ewm(span=120).mean()

        return res

    def run_backtest(
        self,
        df: pd.DataFrame,
        token: str = 'ASSET',
        funding_rate: Optional[pd.Series] = None,
        macro_multipliers: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, List[TradeRecord], pd.Series]:
        """
        Executes causal backtest on historical OHLCV data.
        - Signal generated at close of bar t.
        - Filled at open of bar t+1.
        - Deducts fee_and_slippage on every turnover.
        - Accrues continuous 8h funding rate carry.
        
        Returns:
            bar_returns (pd.Series): Mark-to-market portfolio bar returns.
            trades (List[TradeRecord]): Comprehensive trade execution records.
            positions (pd.Series): Historical signed position time series [-1.0, 1.0].
        """
        ind = self.compute_indicators(df)
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        idx = df.index
        n = len(df)

        fund_vals = funding_rate.reindex(df.index).fillna(0.0).values if funding_rate is not None else np.zeros(n)
        macro_mults = macro_multipliers.reindex(df.index).fillna(1.0).values if macro_multipliers is not None else np.ones(n)

        # active_pos[k] tracks the position held from open[k] to open[k+1]
        active_pos = np.zeros(n)
        trades: List[TradeRecord] = []

        in_pos = 0  # +1 Long, -1 Short, 0 Cash
        entry_idx = 0
        current_size = 0.0
        peak_price_since_entry = 0.0
        trough_price_since_entry = 1e9
        trailing_stop_price = 0.0

        for i in range(1, n - 1):
            if i < self.min_warmup_bars:
                continue

            curr_c = closes[i]
            curr_h = highs[i]
            curr_l = lows[i]
            curr_atr = ind['atr'].iloc[i]
            if np.isnan(curr_atr):
                continue
            curr_macro_mult = macro_mults[i]

            if in_pos == 0:
                enter_long = False
                enter_short = False

                if self.mode == 'donchian':
                    if curr_c > ind['donchian_high'].iloc[i]:
                        enter_long = True
                    elif self.use_short and curr_c < ind['donchian_break_short'].iloc[i]:
                        enter_short = True

                elif self.mode == 'bollinger':
                    if curr_c > ind['bb_upper'].iloc[i]:
                        enter_long = True
                    elif self.use_short and curr_c < ind['bb_lower'].iloc[i]:
                        enter_short = True

                elif self.mode == 'chan_swing':
                    # Break above prior swing high and above EMA120
                    if curr_c > ind['swing_high'].iloc[i] and curr_c > ind['ema120'].iloc[i]:
                        enter_long = True
                    elif self.use_short and curr_c < ind['swing_low'].iloc[i] and curr_c < ind['ema120'].iloc[i]:
                        enter_short = True

                elif self.mode == 'dual_ema':
                    if ind['ema20'].iloc[i] > ind['ema120'].iloc[i]:
                        enter_long = True
                    elif self.use_short and ind['ema20'].iloc[i] < ind['ema120'].iloc[i]:
                        enter_short = True

                if enter_long and curr_macro_mult > 0.1:
                    in_pos = 1
                    entry_idx = i + 1  # will enter at open[i+1]
                    current_size = self.base_size * curr_macro_mult
                    peak_price_since_entry = curr_c
                    initial_stop = max(ind['swing_low'].iloc[i], curr_c - self.atr_trailing_mult * curr_atr)
                    trailing_stop_price = initial_stop

                elif enter_short and curr_macro_mult > 0.1:
                    in_pos = -1
                    entry_idx = i + 1
                    current_size = self.base_size * curr_macro_mult
                    trough_price_since_entry = curr_c
                    initial_stop = min(ind['swing_high'].iloc[i], curr_c + self.atr_trailing_mult * curr_atr)
                    trailing_stop_price = initial_stop

            elif in_pos == 1:
                peak_price_since_entry = max(peak_price_since_entry, curr_h)
                new_trailing_stop = peak_price_since_entry - self.atr_trailing_mult * curr_atr
                structural_floor = ind['swing_low'].iloc[i]
                active_stop_candidate = max(new_trailing_stop, structural_floor)
                trailing_stop_price = max(trailing_stop_price, active_stop_candidate)

                exit_long = False
                exit_reason = ''

                if curr_c < trailing_stop_price:
                    exit_long = True
                    exit_reason = 'TRAILING_STOP'
                elif self.mode == 'donchian' and curr_c < ind['donchian_low'].iloc[i]:
                    exit_long = True
                    exit_reason = 'CHANNEL_EXIT'
                elif self.mode == 'bollinger' and curr_c < ind['bb_mid'].iloc[i]:
                    exit_long = True
                    exit_reason = 'CHANNEL_EXIT'
                elif self.mode == 'dual_ema' and ind['ema20'].iloc[i] < ind['ema120'].iloc[i]:
                    exit_long = True
                    exit_reason = 'EMA_CROSS'
                elif curr_macro_mult == 0.0:
                    exit_long = True
                    exit_reason = 'REGIME_REVERSAL'

                if exit_long:
                    exit_idx = i + 1
                    entry_p = opens[entry_idx]
                    exit_p = opens[exit_idx]
                    gross_ret = (exit_p / entry_p) - 1.0
                    roundtrip_cost = 2.0 * self.fee_and_slippage
                    funding_carry = -np.sum(fund_vals[entry_idx:exit_idx] * 0.5)
                    net_ret = gross_ret - roundtrip_cost + funding_carry
                    dur_hours = (exit_idx - entry_idx) * 4.0

                    # Mark active_pos from entry_idx up to exit_idx - 1
                    active_pos[entry_idx:exit_idx] = current_size

                    trades.append(
                        TradeRecord(
                            token=token,
                            direction='LONG',
                            entry_idx=entry_idx,
                            entry_time=idx[entry_idx],
                            entry_price=entry_p,
                            exit_idx=exit_idx,
                            exit_time=idx[exit_idx],
                            exit_price=exit_p,
                            size=current_size,
                            duration_hours=dur_hours,
                            duration_days=dur_hours / 24.0,
                            gross_ret=gross_ret,
                            net_ret=net_ret,
                            funding_carry=funding_carry,
                            exit_reason=exit_reason,
                        )
                    )
                    in_pos = 0

            elif in_pos == -1:
                trough_price_since_entry = min(trough_price_since_entry, curr_l)
                new_trailing_stop = trough_price_since_entry + self.atr_trailing_mult * curr_atr
                structural_ceiling = ind['swing_high'].iloc[i]
                active_stop_candidate = min(new_trailing_stop, structural_ceiling)
                trailing_stop_price = min(trailing_stop_price, active_stop_candidate)

                exit_short = False
                exit_reason = ''

                if curr_c > trailing_stop_price:
                    exit_short = True
                    exit_reason = 'TRAILING_STOP'
                elif self.mode == 'donchian' and curr_c > ind['donchian_high'].iloc[i]:
                    exit_short = True
                    exit_reason = 'CHANNEL_EXIT'
                elif self.mode == 'bollinger' and curr_c > ind['bb_mid'].iloc[i]:
                    exit_short = True
                    exit_reason = 'CHANNEL_EXIT'
                elif self.mode == 'dual_ema' and ind['ema20'].iloc[i] > ind['ema120'].iloc[i]:
                    exit_short = True
                    exit_reason = 'EMA_CROSS'

                if exit_short:
                    exit_idx = i + 1
                    entry_p = opens[entry_idx]
                    exit_p = opens[exit_idx]
                    gross_ret = (entry_p / exit_p) - 1.0
                    roundtrip_cost = 2.0 * self.fee_and_slippage
                    funding_carry = np.sum(fund_vals[entry_idx:exit_idx] * 0.5)
                    net_ret = gross_ret - roundtrip_cost + funding_carry
                    dur_hours = (exit_idx - entry_idx) * 4.0

                    active_pos[entry_idx:exit_idx] = -current_size

                    trades.append(
                        TradeRecord(
                            token=token,
                            direction='SHORT',
                            entry_idx=entry_idx,
                            entry_time=idx[entry_idx],
                            entry_price=entry_p,
                            exit_idx=exit_idx,
                            exit_time=idx[exit_idx],
                            exit_price=exit_p,
                            size=current_size,
                            duration_hours=dur_hours,
                            duration_days=dur_hours / 24.0,
                            gross_ret=gross_ret,
                            net_ret=net_ret,
                            funding_carry=funding_carry,
                            exit_reason=exit_reason,
                        )
                    )
                    in_pos = 0

        # If still in position at the end of the data, mark active_pos
        if in_pos != 0:
            if in_pos == 1:
                active_pos[entry_idx:n-1] = current_size
            else:
                active_pos[entry_idx:n-1] = -current_size

        # Causal mark-to-market bar return calculation:
        # active_pos[k] is held from open[k] to open[k+1]
        open_to_open_rets = np.zeros(n)
        open_to_open_rets[:-1] = opens[1:] / (opens[:-1] + 1e-8) - 1.0

        # Turnover fees applied on transitions:
        # Entry turnover at entry_idx: size * fee
        # Exit turnover at exit_idx: size * fee
        turnover = np.abs(active_pos - np.roll(active_pos, 1))
        turnover[0] = np.abs(active_pos[0])

        funding_pnl = np.where(active_pos > 0, -fund_vals * 0.5 * active_pos, np.where(active_pos < 0, fund_vals * 0.5 * np.abs(active_pos), 0.0))
        bar_rets = active_pos * open_to_open_rets - turnover * self.fee_and_slippage + funding_pnl

        pos_series = pd.Series(active_pos, index=idx)
        bar_ret_series = pd.Series(bar_rets, index=idx)

        return bar_ret_series, trades, pos_series
