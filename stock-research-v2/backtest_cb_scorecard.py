# -*- coding: utf-8 -*-
"""
backtest_cb_scorecard.py - 评分卡与熔断机制驱动的可转债轮动策略回测
==================================================================

本文件实现以下风控机制的可转债等权轮动回测：
1. 双低多因子选股 (N=20, 2W 轮动)
2. 评分卡仓位把控 (Green/Yellow/Red 动态仓位系数)
3. 账户层净值 15% 熔断保护 (15% Drawdown Melt + 60d Cooldown)
"""

import os
import sys
import pandas as pd
import numpy as np

# 导入同目录的模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from backtest_cb_doublelow import load_data, compute_metrics
from cb_risk_scorecard import calculate_scorecard_signals

def run_scorecard_backtest(
    df_pit,
    N=20,
    rebalance_freq='2W',
    min_rating='A',            # 评级 >= A
    min_size=1.0,              # 规模 >= 1.0亿
    min_maturity=0.5,          # 期限 >= 6个月
    single_side_cost=0.0005,   # 0.05%
    initial_capital=1000000.0,
    use_multi_factor=True,     # 默认多因子打分
    use_risk_control=True,     # 个券止损与130强平
    use_scorecard=True,        # 使用风险评分卡
    use_portfolio_melt=True,   # 使用账户层 15% 熔断
    melt_threshold=-0.15,      # 15% 熔断阈值
    cooldown_days=60           # 60 交易日冷静期
):
    df_pit = df_pit.copy()
    
    # 评级映射
    rating_ranks = {'AAA': 6, 'AA+': 5, 'AA': 4, 'AA-': 3, 'A+': 2, 'A': 1}
    min_rating_rank = rating_ranks.get(min_rating, 1)
    
    def get_rating_rank(r):
        if pd.isna(r): return 0
        r_str = str(r).upper().strip()
        for key in rating_ranks:
            if r_str.startswith(key): return rating_ranks[key]
        return 0
        
    df_pit['rating_rank'] = df_pit['rating'].apply(get_rating_rank)
    
    # 1. 计算评分卡信号
    print("生成评分卡日频信号序列...")
    df_signals = calculate_scorecard_signals(df_pit)
    df_signals['trade_date'] = pd.to_datetime(df_signals['trade_date'])
    signals_dict = df_signals.set_index('trade_date')['signal_coef'].to_dict()
    reasons_dict = df_signals.set_index('trade_date')['reason'].to_dict()
    
    # 获取日期列表
    trading_dates = sorted(df_pit['trade_date'].unique())
    df_dates = pd.DataFrame({'trade_date': trading_dates})
    
    # 计算调仓期
    if rebalance_freq == 'W':
        df_dates['group'] = df_dates['trade_date'].dt.strftime('%Y-%U')
    elif rebalance_freq == '2W':
        weeks = df_dates['trade_date'].dt.isocalendar().week
        years = df_dates['trade_date'].dt.isocalendar().year
        df_dates['group'] = years.astype(str) + '_' + (weeks // 2).astype(str)
    elif rebalance_freq == 'M':
        df_dates['group'] = df_dates['trade_date'].dt.strftime('%Y-%m')
    else:
        raise ValueError(f"Invalid rebalance_freq: {rebalance_freq}")
        
    rebalance_signal_dates = set(df_dates.groupby('group')['trade_date'].first())
    
    # 账户状态
    cash = initial_capital
    shares = {}              # {code: shares}
    purchase_prices = {}     # {code: price}
    last_known_close = {}    # {code: price}
    
    nav_history = []
    holdings_history = []
    
    # 熔断控制状态
    max_nav = initial_capital
    melt_active = False
    cooldown_remaining = 0
    melt_log = []
    
    # 调仓缓存
    target_weights = None
    target_position_coef = 1.0
    
    # 优化速度：提前按交易日分组为字典，避免循环中进行 boolean slicing
    df_today_dict = {dt: grp.set_index('ts_code') for dt, grp in df_pit.groupby('trade_date')}
    
    for idx, dt in enumerate(trading_dates):
        if dt not in df_today_dict:
            continue
        df_today = df_today_dict[dt]
        sig_coef = signals_dict.get(dt, 1.0) if use_scorecard else 1.0
        reason = reasons_dict.get(dt, 'Green') if use_scorecard else 'Green'
        
        # 1. 熔断冷静期递减
        if melt_active:
            cooldown_remaining -= 1
            if cooldown_remaining <= 0:
                melt_active = False
                print(f"[{dt.date()}] Cooldown finished. Resuming strategy.")
                
        # 2. 计算今日持仓收盘价值 (剔除退市债)
        val_holdings_close = 0.0
        for code, sh in list(shares.items()):
            if code in df_today.index:
                close_price = df_today.loc[code, 'close']
                
                # A. 评分卡红灯 (sig_coef == 0.0) 触发紧急盘中/收盘清仓
                is_red_light = use_scorecard and (sig_coef == 0.0)
                
                # B. 个券止损 (Price <= 0.95 * Cost) 和 强赎预警 (Price >= 130)
                purch_price = purchase_prices.get(code, close_price)
                is_stop_loss = use_risk_control and (close_price <= 0.95 * purch_price)
                is_warning_line = use_risk_control and (close_price >= 130.0)
                
                if is_red_light or is_stop_loss or is_warning_line:
                    # 立即变现
                    cash += sh * close_price * (1.0 - single_side_cost)
                    del shares[code]
                    if code in purchase_prices:
                        del purchase_prices[code]
                else:
                    val_holdings_close += sh * close_price
                    last_known_close[code] = close_price
            else:
                # 退市平仓
                close_price = last_known_close.get(code, 100.0)
                cash += sh * close_price * (1.0 - single_side_cost)
                del shares[code]
                if code in purchase_prices:
                    del purchase_prices[code]
                    
        current_nav = val_holdings_close + cash
        
        # 3. 组合层熔断风控判定 (以当前 NAV 测算)
        max_nav = max(max_nav, current_nav)
        dd = (current_nav - max_nav) / max_nav
        
        if use_portfolio_melt and not melt_active and dd <= melt_threshold:
            print(f"[MELT] [{dt.date()}] 警告: 组合回撤达 {dd:.2%} 触发 15% 熔断红线！强制清仓并进入 {cooldown_days} 天冷静期。")
            melt_active = True
            cooldown_remaining = cooldown_days
            melt_log.append({'date': dt, 'nav': current_nav, 'drawdown': dd})
            
            # 立即清仓所有剩余转债
            for code, sh in list(shares.items()):
                if code in df_today.index:
                    price = df_today.loc[code, 'close']
                else:
                    price = last_known_close.get(code, 100.0)
                cash += sh * price * (1.0 - single_side_cost)
            shares = {}
            purchase_prices = {}
            val_holdings_close = 0.0
            current_nav = cash
            
        nav_history.append({
            'trade_date': dt,
            'nav': current_nav,
            'cash': cash,
            'signal_coef': sig_coef,
            'reason': reason,
            'drawdown': dd,
            'melt_active': int(melt_active)
        })
        
        # 保存收盘价
        for code in df_today.index:
            last_known_close[code] = df_today.loc[code, 'close']
            
        # 记录每日持仓比例
        holdings_history.append({
            'trade_date': dt,
            'holdings': {code: sh * df_today.loc[code, 'close'] / current_nav if code in df_today.index else 0.0 for code, sh in shares.items()}
        })
        
        # 4. 执行调仓信号 (隔天收盘成交)
        if target_weights is not None and not melt_active:
            # 清仓所有旧持仓
            cash_temp = cash
            for code, sh in list(shares.items()):
                if code in df_today.index:
                    price = df_today.loc[code, 'close']
                else:
                    price = last_known_close.get(code, 100.0)
                cash_temp += sh * price * (1.0 - single_side_cost)
                
            # 根据评分卡信号调整总仓位占比 (绿灯=100%, 黄灯=50%, 红灯=0%)
            cash_allocated = cash_temp * target_position_coef
            
            # 建仓新券
            shares = {}
            purchase_prices = {}
            for code, weight in target_weights.items():
                if code in df_today.index:
                    price = df_today.loc[code, 'close']
                    target_value = cash_allocated * weight
                    shares[code] = target_value / (price * (1.0 + single_side_cost))
                    purchase_prices[code] = price
                    
            cash = cash_temp - sum(sh * df_today.loc[code, 'close'] * (1.0 + single_side_cost) for code, sh in shares.items())
            target_weights = None
            
        # 5. 生成调仓信号 (T 日收盘出信号)
        if dt in rebalance_signal_dates and not melt_active:
            # 确定本次调仓的仓位系数 (使用今日评分卡信号)
            target_position_coef = sig_coef
            
            # 过滤股票池
            df_active = df_today.dropna(subset=['close', 'premium']).copy()
            # 剔除正股 ST
            df_active = df_active[~df_active['stock_name'].str.contains('ST', na=False)]
            # 规模限制
            if min_size > 0:
                df_active = df_active[df_active['issue_size'] >= min_size]
            # 剩余存续期限制
            df_active = df_active[df_active['years_to_maturity'] >= min_maturity]
            # 评级限制
            df_active = df_active[df_active['rating_rank'] >= min_rating_rank]
            
            if not df_active.empty and target_position_coef > 0:
                if use_multi_factor:
                    # 稳健多因子打分
                    r_dl = df_active['double_low'].rank(pct=True, ascending=True)
                    r_prem = df_active['premium'].rank(pct=True, ascending=True)
                    mom_filled = df_active['stock_mom_20'].fillna(df_active['stock_mom_20'].median())
                    vol_filled = df_active['stock_vol_20'].fillna(df_active['stock_vol_20'].median())
                    r_mom = mom_filled.rank(pct=True, ascending=False)
                    r_vol = vol_filled.rank(pct=True, ascending=True)
                    r_scale = df_active['issue_size'].rank(pct=True, ascending=True)
                    r_ytm = df_active['ytm'].rank(pct=True, ascending=False)
                    r_dist = df_active['dist_redempt'].rank(pct=True, ascending=False)
                    
                    df_active['score'] = (
                        0.30 * r_dl + 
                        0.30 * r_prem + 
                        0.10 * r_mom + 
                        0.10 * r_vol + 
                        0.10 * r_scale + 
                        0.05 * r_ytm + 
                        0.05 * r_dist
                    )
                    df_selected = df_active.sort_values('score').head(N)
                else:
                    # 纯双低打分
                    df_selected = df_active.sort_values('double_low').head(N)
                    
                target_codes = df_selected.index.tolist()
                target_weights = {code: 1.0 / len(target_codes) for code in target_codes}
            else:
                target_weights = {}
                
    df_nav = pd.DataFrame(nav_history).set_index('trade_date')
    return df_nav, holdings_history, melt_log
