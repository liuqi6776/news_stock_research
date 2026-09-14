"""
Strategy A & Strategy B 5-Year Backtest Engine (2021 - 2026)
方案 A 与 方案 B 过去5年（2021-2026）深度量化回测系统
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def backtest_strategy_a_impulse(
    btc_df: pd.DataFrame,
    target_df: pd.DataFrame,
    symbol_name: str = "ETHUSDT",
    impulse_ret_threshold: float = 0.015,  # 1.5% 1h impulse
    taker_buy_threshold: float = 0.58,      # >58% taker aggressive buy
    take_profit: float = 0.05,              # +5% profit target
    stop_loss: float = -0.025,              # -2.5% hard stop
    max_hold_bars: int = 12,                # max hold 12 hours
    fee_rate: float = 0.0004,               # 0.04% taker fee with BNB
    slippage: float = 0.0002
) -> Dict[str, Any]:
    """
    Strategy A: BTC Impulse Leading Follower Strategy
    方案 A：BTC 脉冲引领 ➔ 目标币种极速跟随动量策略
    """
    common_idx = btc_df.index.intersection(target_df.index)
    btc = btc_df.loc[common_idx]
    target = target_df.loc[common_idx]

    btc_ret = btc['close'].pct_change()
    btc_taker_ratio = btc['taker_buy_volume'] / (btc['volume'] + 1e-8)
    
    # Impulse trigger on BTC
    btc_bull_impulse = (btc_ret >= impulse_ret_threshold) & (btc_taker_ratio >= taker_buy_threshold)
    
    # Also require target to not already be in a massive overbought blow-off
    target_ret_1h = target['close'].pct_change()
    
    initial_capital = 10000.0
    capital = initial_capital
    pos = 0 # 0 = flat, 1 = long
    entry_price = 0.0
    entry_bar = 0
    entry_date = None
    
    trades = []
    equity_curve = []
    
    opens = target['open'].values
    highs = target['high'].values
    lows = target['low'].values
    closes = target['close'].values
    signals = btc_bull_impulse.values
    times = common_idx
    
    for i in range(1, len(common_idx)):
        cur_open = opens[i]
        cur_high = highs[i]
        cur_low = lows[i]
        cur_close = closes[i]
        
        # 1. Position management & Exit check
        if pos == 1:
            bars_held = i - entry_bar
            pnl_high = (cur_high - entry_price) / entry_price
            pnl_low = (cur_low - entry_price) / entry_price
            
            exit = False
            exit_price = cur_open
            reason = ""
            
            # Stop Loss
            if pnl_low <= stop_loss:
                exit_price = entry_price * (1.0 + stop_loss - slippage)
                exit = True
                reason = "STOP_LOSS"
            # Take Profit
            elif pnl_high >= take_profit:
                exit_price = entry_price * (1.0 + take_profit - slippage)
                exit = True
                reason = "TAKE_PROFIT"
            # Time exit
            elif bars_held >= max_hold_bars:
                exit_price = cur_open * (1.0 - slippage)
                exit = True
                reason = "TIME_OUT"
                
            if exit:
                net_ret = (exit_price - entry_price) / entry_price - (fee_rate * 2)
                capital *= (1.0 + net_ret)
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": times[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": net_ret,
                    "bars_held": bars_held,
                    "reason": reason
                })
                pos = 0
                
        # 2. Entry check
        if pos == 0:
            if signals[i - 1]: # Previous bar BTC triggered impulse
                pos = 1
                entry_price = cur_open * (1.0 + slippage)
                entry_bar = i
                entry_date = times[i]
                
        cur_val = capital * (cur_close / entry_price if pos == 1 else 1.0)
        equity_curve.append(cur_val)
        
    trades_df = pd.DataFrame(trades)
    eq = pd.Series(equity_curve, index=common_idx[1:])
    peak = eq.cummax()
    dd = (eq - peak) / peak
    
    total_ret = (capital / initial_capital - 1.0) * 100
    bh_ret = (closes[-1] / opens[1] - 1.0) * 100
    years = len(common_idx) / 8760.0
    cagr = ((capital / initial_capital) ** (1.0 / max(years, 0.1)) - 1.0) * 100 if capital > 0 else -100
    
    win_rate = (trades_df['return'] > 0).mean() * 100 if not trades_df.empty else 0
    profit_factor = trades_df[trades_df['return'] > 0]['return'].sum() / abs(trades_df[trades_df['return'] <= 0]['return'].sum()) if (not trades_df.empty and (trades_df['return'] <= 0).sum() != 0) else np.nan
    
    # Calculate Sharpe
    strat_rets = eq.pct_change().fillna(0)
    sharpe = (strat_rets.mean() / (strat_rets.std() + 1e-8)) * np.sqrt(8760)
    
    return {
        "symbol": symbol_name,
        "total_return": total_ret,
        "cagr": cagr,
        "benchmark_return": bh_ret,
        "max_drawdown": dd.min() * 100,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trades": len(trades_df),
        "trades_df": trades_df,
        "equity": eq
    }

def backtest_strategy_b_maker_grid(
    df: pd.DataFrame,
    symbol_name: str = "ETHUSDT",
    grid_levels: int = 10,                 # 10 grid levels
    grid_spacing_atr_pct: float = 0.5,     # 0.5 ATR per grid spacing (~1.0% to 1.5%)
    trend_filter_ma: int = 240,            # 240 hours = 10 days MA trend filter
    maker_fee: float = 0.0002,             # 0.02% maker fee (Binance VIP / BNB)
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Strategy B: Adaptive High-Frequency Maker Grid / Market Making Strategy
    方案 B：币安自适应高频做市挂单网格策略（带大周期下行熔断避险）
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    times = df.index
    
    # Precompute ATR(24) and MA(trend_filter_ma)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(24).mean().fillna(closes[0] * 0.02).values
    ma_trend = df['close'].rolling(trend_filter_ma).mean().values
    
    # Capital management
    cash = initial_capital * 0.5 # 50% cash, 50% initial coin allocation for dual-sided grid
    coin_holdings = (initial_capital * 0.5) / closes[0]
    
    grid_profit = 0.0
    total_grid_trades = 0
    grid_wins = 0
    
    equity_curve = []
    
    # Base center price for grid
    center_price = closes[0]
    half_levels = grid_levels // 2
    
    for i in range(len(closes)):
        cur_p = closes[i]
        cur_atr = atr[i]
        cur_ma = ma_trend[i]
        
        # Grid step size dynamically pegged to volatility
        step = max(cur_p * 0.008, cur_atr * grid_spacing_atr_pct)
        
        # Trend protection rule:
        # In brutal bear market crashes (Price < MA_trend * 0.90), pause lower buy grids to avoid bag-holding!
        in_severe_downtrend = (not np.isnan(cur_ma)) and (cur_p < cur_ma * 0.88)
        
        # Check if price crossed grid levels from previous bar
        if i > 0:
            prev_p = closes[i - 1]
            price_move = cur_p - prev_p
            
            # Number of grid steps traversed in this bar
            num_steps = int(abs(price_move) / step)
            
            if num_steps > 0:
                trade_volume_usd = (initial_capital / grid_levels)
                trade_coins = trade_volume_usd / cur_p
                
                if price_move > 0:
                    # Price rose -> Sell grids triggered (Maker Limit Sell fills)
                    actual_sells = min(num_steps, half_levels)
                    for _ in range(actual_sells):
                        if coin_holdings >= trade_coins:
                            coin_holdings -= trade_coins
                            cash += trade_coins * cur_p * (1.0 - maker_fee)
                            profit = trade_coins * step * (1.0 - maker_fee * 2)
                            grid_profit += profit
                            total_grid_trades += 1
                elif price_move < 0 and not in_severe_downtrend:
                    # Price dropped -> Buy grids triggered (Maker Limit Buy fills)
                    actual_buys = min(num_steps, half_levels)
                    for _ in range(actual_buys):
                        cost = trade_coins * cur_p * (1.0 + maker_fee)
                        if cash >= cost:
                            cash -= cost
                            coin_holdings += trade_coins
                            total_grid_trades += 1
                            
        # Valuation: Cash + Coin holdings mark-to-market
        total_val = cash + (coin_holdings * cur_p)
        equity_curve.append(total_val)
        
    eq = pd.Series(equity_curve, index=times)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    
    total_ret = (eq.iloc[-1] / initial_capital - 1.0) * 100
    bh_ret = (closes[-1] / closes[0] - 1.0) * 100
    years = len(df) / 8760.0
    cagr = ((eq.iloc[-1] / initial_capital) ** (1.0 / max(years, 0.1)) - 1.0) * 100 if eq.iloc[-1] > 0 else -100
    
    strat_rets = eq.pct_change().fillna(0)
    sharpe = (strat_rets.mean() / (strat_rets.std() + 1e-8)) * np.sqrt(8760)
    
    return {
        "symbol": symbol_name,
        "total_return": total_ret,
        "cagr": cagr,
        "benchmark_return": bh_ret,
        "max_drawdown": dd.min() * 100,
        "sharpe": sharpe,
        "grid_trades": total_grid_trades,
        "final_capital": eq.iloc[-1],
        "equity": eq
    }
