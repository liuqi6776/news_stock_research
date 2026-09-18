# -*- coding: utf-8 -*-
"""
Score-Based Initial Position Sizing & Floating-Profit Booster Engine (Phase 24/25)
================================================================================
A systematic quantitative execution engine implementing score-based initial position
sizing and conditional floating-profit leverage expansion.

Key Architecture:
- Confluence Score 1 (0.33x / 1/3 Position): Initial breakout probe (Close > 120-bar BB Upper).
  Minimizes capital exposure on false breakouts.
- Confluence Score 2 (0.67x / 2/3 Position): Macro trend confirmation (Close > EMA200).
- Confluence Score 3 (1.00x / Full Position): Multi-indicator resonance (EMA20 > EMA60 & RSI > 50).
- Booster (1.25x - 1.50x): Conditional leverage expansion on established trends.

CRITICAL RISK DISCLOSURE & INVARIANTS:
- There is NO "zero principal risk" in leveraged trading. Adding to a position raises
  the average entry price.
- In rapid reversals, flash crashes, gap downs, or liquidity voids, prices can blow
  through stop levels, resulting in loss of floating profits or principal.
- Protection Invariant: The booster is STRICTLY REJECTED if the resulting new average
  entry price would be greater than or equal to the current trailing stop:
  `new_avg_entry < trailing_stop` is mandatory.
- Default configuration disables booster (`use_booster = False`).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class TieredBoosterTrade:
    token: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    final_size: float
    max_tier: int  # 1: 1/3, 2: 2/3, 3: 1.0x, 4: Booster (1.25x/1.5x)
    gross_ret: float
    net_ret: float
    duration_days: float
    reason: str  # 'TRAILING_STOP', 'MID_EXIT'
    booster_activated: bool
    fills: List[Dict[str, Any]] = field(default_factory=list)
    borrowing_cost_total: float = 0.0
    funding_cost_total: float = 0.0
    fees_total: float = 0.0


class ScoreBasedBoosterEngine:
    """
    Score-Based Initial Position Sizing & Conditional Floating-Profit Booster Engine.
    """

    def __init__(
        self,
        lookback_bars: int = 120,
        exit_lookback_bars: int = 60,
        atr_period: int = 14,
        atr_trailing_mult: float = 3.0,
        booster_leverage: float = 1.25,
        fee_and_slippage: float = 0.0008,
        borrow_rate_apr: float = 0.06,
        use_tiered_entry: bool = True,
        use_booster: bool = False,
        min_warmup_bars: int = 200,
    ):
        self.lookback_bars = lookback_bars
        self.exit_lookback_bars = exit_lookback_bars
        self.atr_period = atr_period
        self.atr_trailing_mult = atr_trailing_mult
        self.booster_leverage = booster_leverage
        self.fee_and_slippage = fee_and_slippage
        self.borrow_rate_apr = borrow_rate_apr
        self.use_tiered_entry = use_tiered_entry
        self.use_booster = use_booster
        self.min_warmup_bars = min_warmup_bars

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strictly causal indicator computation without bfill.
        All signals computed at close of bar t are shifted by 1 to execute at open of bar t+1.
        """
        res = pd.DataFrame(index=df.index)
        closes = df['close']
        highs = df['high']
        lows = df['low']
        s_c = closes.shift(1)

        # 1. ATR 14
        tr1 = highs - lows
        tr2 = (highs - s_c).abs()
        tr3 = (lows - s_c).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        res['atr'] = tr.rolling(self.atr_period, min_periods=1).mean()

        # 2. Bollinger Bands 120
        res['bb_mid'] = s_c.rolling(self.lookback_bars).mean()
        res['bb_std'] = s_c.rolling(self.lookback_bars).std()
        res['bb_upper'] = res['bb_mid'] + 2.0 * res['bb_std']
        res['bb_exit_mid'] = s_c.rolling(self.exit_lookback_bars).mean()

        # 3. Macro EMA 200
        res['ema200'] = s_c.ewm(span=200).mean()

        # 4. Moving Average Momentum Alignment
        res['ema20'] = s_c.ewm(span=20).mean()
        res['ema60'] = s_c.ewm(span=60).mean()

        # 5. RSI 14
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_g = gain.rolling(14, min_periods=1).mean()
        avg_l = loss.rolling(14, min_periods=1).mean()
        rs = avg_g / (avg_l + 1e-8)
        res['rsi'] = (100.0 - (100.0 / (1.0 + rs))).shift(1)

        return res

    def run_backtest(
        self,
        df: pd.DataFrame,
        token: str = 'ETHUSDT',
        funding_rate: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, List[TieredBoosterTrade], pd.Series]:
        """
        Executes causal backtest with discrete initial sizing tiers and financing costs.
        """
        ind = self.compute_indicators(df)
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        idx = df.index
        n = len(df)

        fund_vals = funding_rate.reindex(df.index).fillna(0.0).values if funding_rate is not None else np.zeros(n)
        borrow_rate_4h = self.borrow_rate_apr / 2190.0

        bar_rets = np.zeros(n)
        positions = np.zeros(n)
        trades: List[TieredBoosterTrade] = []

        cur_size = 0.0
        in_pos = 0  # 0: flat, 1: tier1 (0.33x), 2: tier2 (0.67x), 3: tier3 (1.0x), 4: booster
        entry_idx = 0
        avg_entry_price = 0.0
        highest_price = 0.0
        trailing_stop = 0.0
        bars_in_pos = 0
        booster_activated = False

        trade_fills: List[Dict[str, Any]] = []
        trade_borrow_cost = 0.0
        trade_funding_cost = 0.0
        trade_fees = 0.0

        for i in range(self.min_warmup_bars, n - 1):
            c = closes[i]
            h = highs[i]
            nxt_o = opens[i + 1]
            cur_atr = ind['atr'].iloc[i]
            cur_bb_u = ind['bb_upper'].iloc[i]
            cur_ema200 = ind['ema200'].iloc[i]
            cur_ema20 = ind['ema20'].iloc[i]
            cur_ema60 = ind['ema60'].iloc[i]
            cur_rsi = ind['rsi'].iloc[i]
            cur_exit_mid = ind['bb_exit_mid'].iloc[i]

            # Mark-to-market position carry
            if cur_size > 0:
                price_ret = (nxt_o - opens[i]) / opens[i]
                funding_ret = -fund_vals[i] if (idx[i].hour % 8 == 0) else 0.0
                borrow_cost = borrow_rate_4h * max(0.0, cur_size - 1.0)
                bar_rets[i] = cur_size * price_ret + cur_size * funding_ret - borrow_cost
                positions[i] = cur_size
                bars_in_pos += 1
                highest_price = max(highest_price, h)

                # Monotonic ratchet stop
                new_stop = highest_price - self.atr_trailing_mult * cur_atr
                trailing_stop = max(trailing_stop, new_stop)

                # Accumulate trade carry costs
                if idx[i].hour % 8 == 0:
                    trade_funding_cost += cur_size * fund_vals[i]
                trade_borrow_cost += borrow_cost

            # Check exit triggers
            exit_trade = False
            exit_reason = ''
            if cur_size > 0:
                if c < trailing_stop:
                    exit_trade = True
                    exit_reason = 'TRAILING_STOP'
                elif c < cur_exit_mid and bars_in_pos >= 6:
                    exit_trade = True
                    exit_reason = 'MID_EXIT'

            if exit_trade:
                # Deduct exit friction
                exit_fee = cur_size * self.fee_and_slippage
                bar_rets[i] -= exit_fee
                trade_fees += exit_fee
                trade_fills.append({
                    "time": idx[i + 1],
                    "price": float(nxt_o),
                    "size": float(cur_size),
                    "type": "EXIT",
                    "fee": float(exit_fee),
                })
                gross_ret = nxt_o / avg_entry_price - 1.0
                size_denom = cur_size if cur_size > 0 else 1.0
                net_ret = gross_ret - (trade_fees + trade_borrow_cost + trade_funding_cost) / size_denom
                trades.append(
                    TieredBoosterTrade(
                        token=token,
                        entry_time=idx[entry_idx],
                        exit_time=idx[i + 1],
                        entry_price=avg_entry_price,
                        exit_price=nxt_o,
                        final_size=cur_size,
                        max_tier=in_pos,
                        gross_ret=gross_ret,
                        net_ret=net_ret,
                        duration_days=bars_in_pos * 4.0 / 24.0,
                        reason=exit_reason,
                        booster_activated=booster_activated,
                        fills=trade_fills,
                        borrowing_cost_total=trade_borrow_cost,
                        funding_cost_total=trade_funding_cost,
                        fees_total=trade_fees,
                    )
                )
                cur_size = 0.0
                in_pos = 0
                bars_in_pos = 0
                booster_activated = False
                trade_fills = []
                trade_borrow_cost = 0.0
                trade_funding_cost = 0.0
                trade_fees = 0.0
                continue

            # Position sizing logic
            if cur_size == 0:
                # Initial entry check
                if c > cur_bb_u:
                    macro_bull = (c > cur_ema200)
                    momentum_bull = (cur_ema20 > cur_ema60) and (cur_rsi > 50)
                    score = 1 + int(macro_bull) + int(momentum_bull)

                    if not self.use_tiered_entry:
                        size = 1.0 if macro_bull else 0.5
                        tier = 3 if macro_bull else 2
                    else:
                        if score == 1:
                            size = 0.33
                            tier = 1
                        elif score == 2:
                            size = 0.67
                            tier = 2
                        else:
                            size = 1.00
                            tier = 3

                    cur_size = size
                    in_pos = tier
                    entry_idx = i + 1
                    avg_entry_price = nxt_o
                    highest_price = c
                    trailing_stop = c - self.atr_trailing_mult * cur_atr
                    bars_in_pos = 0
                    booster_activated = False
                    entry_fee = size * self.fee_and_slippage
                    bar_rets[i] -= entry_fee
                    trade_fees = entry_fee
                    trade_borrow_cost = 0.0
                    trade_funding_cost = 0.0
                    trade_fills = [{
                        "time": idx[entry_idx],
                        "price": float(nxt_o),
                        "size": float(size),
                        "type": "INITIAL_ENTRY",
                        "fee": float(entry_fee),
                    }]

            else:
                # In position: check for Booster activation
                if self.use_booster and not booster_activated and self.booster_leverage > cur_size:
                    stop_protects_principal = (trailing_stop >= avg_entry_price)
                    profit_cushion = (c - avg_entry_price) >= 1.2 * cur_atr
                    trend_healthy = (c > cur_ema200) and (cur_rsi > 50)

                    if stop_protects_principal and profit_cushion and trend_healthy:
                        target_size = self.booster_leverage
                        add_size = target_size - cur_size
                        if add_size > 0:
                            new_avg_entry = (avg_entry_price * cur_size + nxt_o * add_size) / target_size
                            # STRICT INVARIANT: new average entry price must remain strictly below trailing stop
                            if new_avg_entry < trailing_stop:
                                add_fee = add_size * self.fee_and_slippage
                                bar_rets[i] -= add_fee
                                trade_fees += add_fee
                                trade_fills.append({
                                    "time": idx[i + 1],
                                    "price": float(nxt_o),
                                    "size": float(add_size),
                                    "type": "BOOSTER_ADD",
                                    "fee": float(add_fee),
                                })
                                avg_entry_price = new_avg_entry
                                cur_size = target_size
                                in_pos = 4
                                booster_activated = True

        return pd.Series(bar_rets, index=idx), trades, pd.Series(positions, index=idx)


# Aliases for backwards compatibility and clarity
TieredBoosterEngine = ScoreBasedBoosterEngine
ScoreBasedInitialSizingEngine = ScoreBasedBoosterEngine

