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
                            use_dyn=True, use_top_derisking=True, use_short=True,
                            trial_mode=False):
    """
    Sleeve 1: Symmetrical Adaptive Momentum with Dynamic Sizing & Hard Stop-Loss
    Supports both Long and Short positions:
    - Long: z > 1.0 and fng < 85
    - Short: z < -1.0 and fng > 15 (if use_short=True)
    - Exit: |z| < deadband or Stop-Loss
    - Incorporates realistic fee + slippage (default 0.08% per turnover)
    - Causal 8h perpetual funding cash flow (4h bar carries 0.5 * 8h funding rate)
    - trial_mode (Phase 15): Activates 5-dimensional trial-trading dynamic downsizing:
      1. m_streak: Consecutive loss/stop-loss penalty sizing (1.0 -> 0.7 -> 0.5 -> 0.25)
      2. m_dd: Portfolio peak-drawdown throttle (1.0 -> 0.75 -> 0.50 -> 0.25)
      3. m_vol: Realized ATR target inverse volatility sizing [0.40, 1.10]
      4. m_trend: Macro 144 EMA regime filter (counter-trend capped at 0.40-0.50)
      5. m_conf: Prediction confidence gradient sizing (0.65 for 1.0 < |z| < 1.4, 1.0 for |z| >= 1.4)
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

    # Precompute trial-trading ATR volatility and macro trend if trial_mode is enabled
    if trial_mode:
        if highs is not None and lows is not None:
            tr1 = highs - lows
            tr2 = (highs - closes.shift(1)).abs()
            tr3 = (lows - closes.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        else:
            tr = (closes - closes.shift(1)).abs()
        atr14 = tr.rolling(14).mean().bfill()
        atr_ratio = (atr14 / (closes + 1e-8)).values
        m_vol = np.clip(0.025 / (atr_ratio + 1e-8), 0.40, 1.10)

        ema144 = closes.shift(1).ewm(span=144).mean()
        trend_bias = (closes.shift(1) - ema144).values

    n = len(preds)
    pos = np.zeros(n)
    in_pos = 0  # +1 Long, -1 Short, 0 Cash
    entry_bar = 0
    current_size = 0.0
    in_waterfall = False  # Long stop-loss cooldown flag
    in_short_squeeze = False  # Short stop-loss cooldown flag
    trades = []

    loss_streak = 0
    cum_equity = 1.0
    peak_equity = 1.0

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
            if trial_mode:
                # 1. Streak multiplier
                if loss_streak == 0:
                    m_str = 1.0
                elif loss_streak == 1:
                    m_str = 0.70
                elif loss_streak == 2:
                    m_str = 0.50
                else:
                    m_str = 0.25

                # 2. Drawdown throttle
                dd = max(0.0, (peak_equity - cum_equity) / (peak_equity + 1e-8))
                if dd <= 0.04:
                    m_dd = 1.0
                elif dd <= 0.08:
                    m_dd = 0.75
                elif dd <= 0.12:
                    m_dd = 0.50
                else:
                    m_dd = 0.25

                # 3. Volatility multiplier
                m_v = m_vol[i]

                # 4. Confidence gradient
                m_conf = 0.65 if abs(curr_z) < 1.4 else 1.0

                # 5. Macro trend constraint
                m_tr_long = 0.40 if trend_bias[i] < 0 else 1.0
                m_tr_short = 0.50 if trend_bias[i] > 0 else 1.0

                s_long = float(np.clip(size_long[i] * m_str * m_dd * m_v * m_tr_long * m_conf, 0.15, 1.0))
                s_short = float(np.clip(size_short[i] * m_str * m_dd * m_v * m_tr_short * m_conf, 0.15, 1.0))
            else:
                s_long = size_long[i]
                s_short = size_short[i]

            if curr_z > 1.0 and fng_vals[i] < 85:
                in_pos = 1
                entry_bar = i
                current_size = s_long
                pos[i] = current_size
            elif use_short and curr_z < -1.0 and fng_vals[i] > 15:
                in_pos = -1
                entry_bar = i
                current_size = s_short
                pos[i] = -current_size
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
                weighted_pnl = net_ret * current_size

                trades.append({
                    'type': 'LONG',
                    'entry_time': opens.index[entry_idx],
                    'exit_time': opens.index[exit_idx],
                    'entry_price': entry_p,
                    'exit_price': exit_p,
                    'size': current_size,
                    'duration_hours': duration_hours,
                    'gross_ret': actual_gross,
                    'net_ret': net_ret,
                    'funding_carry': trade_funding_carry,
                    'weighted_pnl': weighted_pnl,
                    'is_stop_loss': is_stop
                })

                if trial_mode:
                    cum_equity *= (1.0 + weighted_pnl)
                    peak_equity = max(peak_equity, cum_equity)
                    if is_stop or net_ret < 0:
                        loss_streak += 1
                    else:
                        loss_streak = 0

                if is_stop:
                    in_waterfall = True
            else:
                pos[i] = current_size
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
                weighted_pnl = net_ret * current_size

                trades.append({
                    'type': 'SHORT',
                    'entry_time': opens.index[entry_idx],
                    'exit_time': opens.index[exit_idx],
                    'entry_price': entry_p,
                    'exit_price': exit_p,
                    'size': current_size,
                    'duration_hours': duration_hours,
                    'gross_ret': actual_gross,
                    'net_ret': net_ret,
                    'funding_carry': trade_funding_carry,
                    'weighted_pnl': weighted_pnl,
                    'is_stop_loss': is_stop
                })

                if trial_mode:
                    cum_equity *= (1.0 + weighted_pnl)
                    peak_equity = max(peak_equity, cum_equity)
                    if is_stop or net_ret < 0:
                        loss_streak += 1
                    else:
                        loss_streak = 0

                if is_stop:
                    in_short_squeeze = True
            else:
                pos[i] = -current_size

    o_series = pd.Series(opens.values, index=opens.index)
    rets_oto = (o_series.shift(-2) / o_series.shift(-1) - 1).values
    trade_signals = pd.Series(pos).diff().abs().fillna(0).values
    cost_bar = trade_signals * fee_and_slippage
    # Causal continuous 8h funding carry: each 4h bar carries 4h/8h = 0.5 funding
    funding_carry = -pos * (funding_vals * 0.5)
    sleeve_rets = (pos * rets_oto - cost_bar + funding_carry)[:-2]

    return pd.Series(sleeve_rets, index=opens.index[:-2]), pd.DataFrame(trades), pd.Series(pos[:-2], index=opens.index[:-2])


def compute_sleeve_trial_trading(preds, opens, closes, lows=None, highs=None,
                                 fng=None, stb=None, funding=None, basis=None,
                                 stop_loss=0.035, deadband=0.20,
                                 fee_and_slippage=0.0008,
                                 use_dyn=True, use_top_derisking=True, use_short=True):
    """
    Convenience wrapper for Sleeve 1 under Institutional Trial Trading Mode (Phase 15).
    Enables multi-dimensional dynamic downsizing (loss streak, drawdown throttle, ATR vol, macro trend, confidence).
    """
    return compute_sleeve_adaptive(preds, opens, closes, lows=lows, highs=highs,
                                   fng=fng, stb=stb, funding=funding, basis=basis,
                                   stop_loss=stop_loss, deadband=deadband,
                                   fee_and_slippage=fee_and_slippage,
                                   use_dyn=use_dyn, use_top_derisking=use_top_derisking,
                                   use_short=use_short, trial_mode=True)


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
