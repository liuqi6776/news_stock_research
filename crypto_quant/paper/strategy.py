# -*- coding: utf-8 -*-
"""
Strategy Module: Incremental Pure Structural Trend Engine (Phase 22)
===================================================================
Executes causal bar-by-bar incremental indicator calculation and signal
generation for the frozen Pure Structural Trend strategy (120-bar Bollinger).
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from crypto_quant.paper.config import (
    ATR_PERIOD,
    ATR_TRAILING_MULT,
    BOLLINGER_STD,
    EXIT_LOOKBACK_BARS,
    LOOKBACK_BARS,
    MACRO_BEAR_SIZE,
    MACRO_BULL_SIZE,
    MACRO_EMA_SPAN,
    MINIMUM_WARMUP_BARS,
    MODE,
    STRATEGY_NAME,
)
from crypto_quant.paper.state import PaperStrategyState


class StructuralTrendPaperStrategy:
    """
    Incremental stateful implementation of Pure Structural Trend Strategy.
    Guaranteed to produce identical decisions as StructuralTrendEngine on closed bars.
    """

    def __init__(
        self,
        lookback_bars: int = LOOKBACK_BARS,
        exit_lookback_bars: int = EXIT_LOOKBACK_BARS,
        bollinger_std: float = BOLLINGER_STD,
        atr_period: int = ATR_PERIOD,
        atr_trailing_mult: float = ATR_TRAILING_MULT,
        macro_ema_span: int = MACRO_EMA_SPAN,
        macro_bull_size: float = MACRO_BULL_SIZE,
        macro_bear_size: float = MACRO_BEAR_SIZE,
        minimum_warmup_bars: int = MINIMUM_WARMUP_BARS,
    ):
        self.lookback_bars = lookback_bars
        self.exit_lookback_bars = exit_lookback_bars
        self.bollinger_std = bollinger_std
        self.atr_period = atr_period
        self.atr_trailing_mult = atr_trailing_mult
        self.macro_ema_span = macro_ema_span
        self.macro_bull_size = macro_bull_size
        self.macro_bear_size = macro_bear_size
        self.minimum_warmup_bars = minimum_warmup_bars
        self.pivot_win = max(15, self.lookback_bars // 4)  # 30 bars

    def compute_indicators(self, history_df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes the latest causal indicators at the close of history_df.iloc[-1].
        History must contain at least minimum_warmup_bars.
        STRICT CAUSALITY: all rolling and ewm channels are shifted by 1 relative to current close.
        """
        if len(history_df) < self.minimum_warmup_bars:
            raise ValueError(
                f"Insufficient warmup history: {len(history_df)} bars (requires >= {self.minimum_warmup_bars})"
            )

        closes = history_df["close"]
        highs = history_df["high"]
        lows = history_df["low"]

        # 1. ATR (strictly causal, rolling mean of true range)
        tr1 = highs - lows
        tr2 = (highs - closes.shift(1)).abs()
        tr3 = (lows - closes.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.rolling(self.atr_period, min_periods=1).mean()
        curr_atr = float(atr_series.iloc[-1])

        # 2. Bollinger Bands 120 (shifted 1 bar so current bar is evaluated against prior 120 bars)
        bb_mid_series = closes.shift(1).rolling(self.lookback_bars).mean()
        bb_std_series = closes.shift(1).rolling(self.lookback_bars).std()
        curr_bb_mid = float(bb_mid_series.iloc[-1])
        curr_bb_std = float(bb_std_series.iloc[-1])
        curr_bb_upper = curr_bb_mid + self.bollinger_std * curr_bb_std
        curr_bb_lower = curr_bb_mid - self.bollinger_std * curr_bb_std

        # 3. Structural Swing Low (shifted 1 bar)
        swing_low_series = lows.shift(1).rolling(self.pivot_win).min()
        curr_swing_low = float(swing_low_series.iloc[-1])

        # 4. Macro EMA 200 (shifted 1 bar)
        ema200_series = closes.shift(1).ewm(span=self.macro_ema_span).mean()
        curr_ema200 = float(ema200_series.iloc[-1])

        curr_close = float(closes.iloc[-1])
        curr_high = float(highs.iloc[-1])
        curr_low = float(lows.iloc[-1])

        # Macro sizing multiplier
        macro_mult = self.macro_bull_size if curr_close > curr_ema200 else self.macro_bear_size

        return {
            "close": curr_close,
            "high": curr_high,
            "low": curr_low,
            "atr": curr_atr,
            "bb_mid": curr_bb_mid,
            "bb_upper": curr_bb_upper,
            "bb_lower": curr_bb_lower,
            "swing_low": curr_swing_low,
            "ema200": curr_ema200,
            "macro_mult": macro_mult,
        }

    def evaluate_bar(
        self,
        history_df: pd.DataFrame,
        state: PaperStrategyState,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Dict[str, float]]:
        """
        Evaluates strategy rules at the close of the latest bar in history_df.

        Returns:
            signal (str or None): 'BUY', 'EXIT', or None
            order_details (dict or None): Parameters for the resulting paper order
            indicators (dict): Diagnostic dictionary of computed indicators
        """
        ind = self.compute_indicators(history_df)
        curr_close = ind["close"]
        curr_high = ind["high"]
        curr_atr = ind["atr"]
        curr_bb_upper = ind["bb_upper"]
        curr_bb_mid = ind["bb_mid"]
        curr_swing_low = ind["swing_low"]
        curr_macro_mult = ind["macro_mult"]

        bar_time = str(history_df.index[-1])

        # Case 1: Currently FLAT -> Evaluate ENTRY
        if state.position == 0:
            if curr_close > curr_bb_upper and curr_macro_mult > 0.1:
                initial_stop = max(curr_swing_low, curr_close - self.atr_trailing_mult * curr_atr)
                order_details = {
                    "side": "BUY",
                    "quantity_fraction": curr_macro_mult,
                    "signal_time": bar_time,
                    "reason": "BOLLINGER_BREAKOUT",
                    "initial_stop": initial_stop,
                    "macro_mult": curr_macro_mult,
                }
                return "BUY", order_details, ind
            return None, None, ind

        # Case 2: Currently IN LONG -> Update Ratchet Stop & Evaluate EXIT
        elif state.position == 1:
            # Update peak price since entry
            peak_price = max(state.highest_price_since_entry, curr_high)
            new_trailing_stop = peak_price - self.atr_trailing_mult * curr_atr
            structural_floor = curr_swing_low
            active_stop_candidate = max(new_trailing_stop, structural_floor)
            ratcheted_stop = max(state.trailing_stop_price, active_stop_candidate)

            # Check exit conditions (Scheme A: close < stop or close < bb_mid)
            exit_reason = None
            if curr_close < ratcheted_stop:
                exit_reason = "TRAILING_STOP"
            elif curr_close < curr_bb_mid:
                exit_reason = "CHANNEL_EXIT"
            elif curr_macro_mult == 0.0:
                exit_reason = "REGIME_REVERSAL"

            if exit_reason is not None:
                order_details = {
                    "side": "SELL",
                    "quantity_fraction": state.position_size,
                    "signal_time": bar_time,
                    "reason": exit_reason,
                    "ratcheted_stop": ratcheted_stop,
                    "peak_price": peak_price,
                }
                return "EXIT", order_details, ind
            else:
                # Update holding state with newly ratcheted stop
                state.highest_price_since_entry = peak_price
                state.trailing_stop_price = ratcheted_stop
                return None, None, ind

        return None, None, ind
