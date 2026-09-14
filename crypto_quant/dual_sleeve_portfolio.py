# -*- coding: utf-8 -*-
"""
Dual-Sleeve Portfolio Engine for Spatio-Temporal Transformer Quantitative System (Phase 13)
Features:
- Symmetrical Long/Short True Alpha Execution (Decoupled from Market Beta)
- Multi-Factor Top/Bottom Risk Radar (EMA Stretch, Perpetual Funding Crowding, FNG Sentiment)
- Symmetrical Dynamic Position Sizing (w_long and w_short in [0.35, 1.0])
- Symmetrical Hard Stop-Loss (Long stop-loss and Short stop-loss)
- Sleeve 1 (70%): Symmetrical Adaptive Momentum with Dynamic Sizing & Stop-Loss
- Sleeve 2 (30%): 8h Symmetrical Micro-Momentum Basis Arbitrage (3x / Day)
"""

import os
import sys
import numpy as np
import pandas as pd


def compute_top_and_bottom_risk(closes, funding_rate, fng_score):
    """
    Computes both Top-Exhaustion Risk and Bottom-Exhaustion Risk indices in [0.0, 1.0]:
    1. 72-bar (12-day) EMA Price Extension / Stretch
    2. Binance 8h perpetual funding rate crowding
    3. Alternative.me Fear & Greed Index
    Returns:
    - size_long: Position size for Long trades [0.35, 1.0] (downsized at overheated peaks)
    - size_short: Position size for Short trades [0.35, 1.0] (downsized at capitulation bottoms)
    """
    ema72 = closes.shift(1).ewm(span=72).mean()
    stretch = ((closes.shift(1) - ema72) / (ema72 + 1e-8)).fillna(0.0).values

    stretch_risk = np.clip(stretch / 0.05, -1.0, 1.0)
    fund_risk = np.clip((funding_rate * 100.0) / 0.02, -1.0, 1.0)
    fng_risk = np.clip((fng_score - 50.0) / 30.0, -1.0, 1.0)

    top_risk = 0.45 * np.maximum(0.0, stretch_risk) + 0.35 * np.maximum(0.0, fund_risk) + 0.20 * np.maximum(0.0, fng_risk)
    bot_risk = 0.45 * np.maximum(0.0, -stretch_risk) + 0.35 * np.maximum(0.0, -fund_risk) + 0.20 * np.maximum(0.0, -fng_risk)

    size_long = np.clip(1.0 - 0.65 * np.maximum(0.0, top_risk - 0.25) / 0.75, 0.35, 1.0)
    size_short = np.clip(1.0 - 0.65 * np.maximum(0.0, bot_risk - 0.25) / 0.75, 0.35, 1.0)

    return top_risk, bot_risk, size_long, size_short


def compute_top_exhaustion_risk(closes, funding_rate, fng_score):
    """Backward compatibility wrapper for top exhaustion risk"""
    top_risk, _, size_long, _ = compute_top_and_bottom_risk(closes, funding_rate, fng_score)
    return top_risk, size_long


def compute_sleeve_adaptive(preds, opens, closes, lows=None, highs=None,
                            fng=None, stb=None, funding=None, basis=None,
                            stop_loss=0.035, deadband=0.20,
                            fee_and_slippage=0.0008,
                            use_dyn=True, use_top_derisking=True, use_short=True):
    """
    Sleeve 1: Symmetrical Adaptive Momentum with Dynamic Sizing & Hard Stop-Loss
    Supports both Long and Short positions:
    - Long: z > 1.0 and fng < 85
    - Short: z < -1.0 and fng > 15 (if use_short=True)
    - Exit: |z| < deadband or Stop-Loss
    - Incorporates realistic fee + slippage (default 0.08% per turnover)
    - Causal 8h perpetual funding cash flow (4h bar carries 0.5 * 8h funding rate)
    """
    p_series = pd.Series(preds.values, index=preds.index)
    prior_mean = p_series.shift(1).rolling(72).mean()
    prior_std = p_series.shift(1).rolling(72).std() + 1e-8
    z_vals = ((p_series - prior_mean) / prior_std).values

    funding_vals = funding.values if funding is not None else np.zeros(len(preds))
    fng_vals = fng if fng is not None else np.ones(len(preds)) * 50.0

    if use_top_derisking:
        _, _, size_long, size_short = compute_top_and_bottom_risk(closes, funding_vals, fng_vals)
    else:
        size_long = np.ones(len(preds))
        size_short = np.ones(len(preds))

    n = len(preds)
    pos = np.zeros(n)
    in_pos = 0  # +1 Long, -1 Short, 0 Cash
    entry_bar = 0
    in_waterfall = False  # Long stop-loss cooldown flag
    in_short_squeeze = False  # Short stop-loss cooldown flag
    trades = []

    for i in range(n - 2):
        # State-driven re-entry unlocking (zero clock freeze)
        if in_waterfall:
            if closes.iloc[i] >= opens.iloc[i]:  # Green candle stabilization
                in_waterfall = False
            else:
                pos[i] = 0.0
                continue

        if in_short_squeeze:
            if closes.iloc[i] <= opens.iloc[i]:  # Red candle top rejection
                in_short_squeeze = False
            else:
                pos[i] = 0.0
                continue

        curr_z = z_vals[i]
        curr_c = closes.iloc[i]

        if in_pos == 0:
            if curr_z > 1.0 and fng_vals[i] < 85:
                in_pos = 1
                entry_bar = i
                pos[i] = size_long[i]
            elif use_short and curr_z < -1.0 and fng_vals[i] > 15:
                in_pos = -1
                entry_bar = i
                pos[i] = -size_short[i]
            else:
                pos[i] = 0.0
        elif in_pos == 1:  # Currently in Long
            entry_p = opens.iloc[entry_bar + 1]
            gross_ret = curr_c / entry_p - 1.0
            is_stop = False
            if stop_loss is not None and (gross_ret <= -stop_loss):
                is_stop = True

            should_exit = (curr_z < deadband) or is_stop
            if should_exit:
                in_pos = 0
                pos[i] = 0.0
                entry_idx = entry_bar + 1
                exit_idx = i + 1
                exit_p = opens.iloc[exit_idx]
                actual_gross = exit_p / entry_p - 1.0
                duration_bars = exit_idx - entry_idx
                duration_hours = duration_bars * 4
                roundtrip_cost = 2 * fee_and_slippage
                # Long pays positive funding: cashflow = -funding
                trade_funding_carry = -np.sum(funding_vals[entry_idx:exit_idx] * 0.5)
                net_ret = actual_gross - roundtrip_cost + trade_funding_carry

                trades.append({
                    'type': 'LONG',
                    'entry_time': opens.index[entry_idx],
                    'exit_time': opens.index[exit_idx],
                    'entry_price': entry_p,
                    'exit_price': exit_p,
                    'size': size_long[entry_bar],
                    'duration_hours': duration_hours,
                    'gross_ret': actual_gross,
                    'net_ret': net_ret,
                    'funding_carry': trade_funding_carry,
                    'weighted_pnl': net_ret * size_long[entry_bar],
                    'is_stop_loss': is_stop
                })

                if is_stop:
                    in_waterfall = True
            else:
                pos[i] = size_long[entry_bar]
        elif in_pos == -1:  # Currently in Short
            entry_p = opens.iloc[entry_bar + 1]
            gross_ret = 1.0 - curr_c / entry_p
            is_stop = False
            if stop_loss is not None and (gross_ret <= -stop_loss):
                is_stop = True

            should_exit = (curr_z > -deadband) or is_stop
            if should_exit:
                in_pos = 0
                pos[i] = 0.0
                entry_idx = entry_bar + 1
                exit_idx = i + 1
                exit_p = opens.iloc[exit_idx]
                actual_gross = 1.0 - exit_p / entry_p
                duration_bars = exit_idx - entry_idx
                duration_hours = duration_bars * 4
                roundtrip_cost = 2 * fee_and_slippage
                # Short receives positive funding: cashflow = +funding
                trade_funding_carry = np.sum(funding_vals[entry_idx:exit_idx] * 0.5)
                net_ret = actual_gross - roundtrip_cost + trade_funding_carry

                trades.append({
                    'type': 'SHORT',
                    'entry_time': opens.index[entry_idx],
                    'exit_time': opens.index[exit_idx],
                    'entry_price': entry_p,
                    'exit_price': exit_p,
                    'size': size_short[entry_bar],
                    'duration_hours': duration_hours,
                    'gross_ret': actual_gross,
                    'net_ret': net_ret,
                    'funding_carry': trade_funding_carry,
                    'weighted_pnl': net_ret * size_short[entry_bar],
                    'is_stop_loss': is_stop
                })

                if is_stop:
                    in_short_squeeze = True
            else:
                pos[i] = -size_short[entry_bar]

    o_series = pd.Series(opens.values, index=opens.index)
    rets_oto = (o_series.shift(-2) / o_series.shift(-1) - 1).values
    trade_signals = pd.Series(pos).diff().abs().fillna(0).values
    cost_bar = trade_signals * fee_and_slippage
    # Causal continuous 8h funding carry: each 4h bar carries 4h/8h = 0.5 funding
    funding_carry = -pos * (funding_vals * 0.5)
    sleeve_rets = (pos * rets_oto - cost_bar + funding_carry)[:-2]

    return pd.Series(sleeve_rets, index=opens.index[:-2]), pd.DataFrame(trades), pd.Series(pos[:-2], index=opens.index[:-2])


def compute_sleeve_8h(preds, opens, fng=None, stb=None, funding=None, fee_and_slippage=0.0008, use_short=True):
    """
    Sleeve 2: 8h Symmetrical Fixed Horizon (3x / Day) Basis Momentum
    """
    p_series = pd.Series(preds.values, index=preds.index)
    prior_mean = p_series.shift(1).rolling(72).mean()
    prior_std = p_series.shift(1).rolling(72).std() + 1e-8
    z_vals = ((p_series - prior_mean) / prior_std).values

    funding_vals = funding.values if funding is not None else np.zeros(len(preds))
    fng_vals = fng if fng is not None else np.ones(len(preds)) * 50.0

    n = len(preds)
    pos = np.zeros(n)
    i = 0
    while i < n - 2:
        zi = z_vals[i]
        if zi > 1.0 and fng_vals[i] < 85:
            pos[i] = 1.0
            if i + 1 < n - 2:
                pos[i + 1] = 1.0
            i += 2
        elif use_short and zi < -1.0 and fng_vals[i] > 15:
            pos[i] = -1.0
            if i + 1 < n - 2:
                pos[i + 1] = -1.0
            i += 2
        else:
            pos[i] = 0.0
            i += 1

    o_series = pd.Series(opens.values, index=opens.index)
    rets_oto = (o_series.shift(-2) / o_series.shift(-1) - 1).values
    trade_signals = pd.Series(pos).diff().abs().fillna(0).values
    cost_bar = trade_signals * fee_and_slippage
    funding_carry = -pos * (funding_vals * 0.5)
    sleeve_rets = (pos * rets_oto - cost_bar + funding_carry)[:-2]

    return pd.Series(sleeve_rets, index=opens.index[:-2]), pd.Series(pos[:-2], index=opens.index[:-2])



def build_dual_sleeve_portfolio(sleeve1_rets, sleeve2_rets, w1=0.70, w2=0.30):
    """
    Linear capital-weighted combination of sleeves
    """
    combined_rets = w1 * sleeve1_rets + w2 * sleeve2_rets
    cum = (1 + combined_rets).cumprod()
    total_ret = cum.iloc[-1] - 1.0
    cagr = (1 + total_ret) ** (1 / (len(combined_rets) / 2190)) - 1.0 if total_ret > -1 else -1.0
    dd = (cum - cum.cummax()) / cum.cummax()
    mdd = dd.min()
    daily_equity = cum.resample('1D').last().ffill()
    daily_rets = daily_equity.pct_change().dropna()
    daily_sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * np.sqrt(365)
    calmar = cagr / abs(mdd) if abs(mdd) > 1e-6 else 0.0

    return {
        'returns': combined_rets,
        'cumulative_equity': cum,
        'total_return': total_ret,
        'cagr': cagr,
        'max_drawdown': mdd,
        'daily_sharpe': daily_sharpe,
        'calmar': calmar
    }
