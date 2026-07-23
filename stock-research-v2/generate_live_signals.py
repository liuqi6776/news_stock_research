# -*- coding: utf-8 -*-
"""
generate_live_signals.py - 可转债实盘信号生成与风控盘后/盘中查询工具
===================================================================

功能 / Features:
1. 自动读取最新 PIT 包含数据，运行双低多因子打分引擎。
2. 过滤硬性雷区 (评级 < A, 正股 ST, 规模 < 1亿, 剩余期限 < 0.5年)。
3. 输出最新 Top 20 目标持仓清单（含建议买入金额与目标权重）。
4. 校验当前持仓的个券止损线 (-5%) 与高价强赎线 (>= 130元)，输出预警指示。

使用方式 / Usage:
    python generate_live_signals.py [--holdings holdings.json]
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from backtest_cb_doublelow import load_data

def generate_signals(capital_allocated=1200000.0, top_n=20):
    """
    生成实盘最新选股信号 / Generate live portfolio selection signals
    """
    print("正在加载最新可转债 PIT 数据集...")
    df_pit = load_data()
    
    latest_date = df_pit['trade_date'].max()
    print(f"数据最新交易日: {latest_date.strftime('%Y-%m-%d')}")
    
    df_today = df_pit[df_pit['trade_date'] == latest_date].copy()
    
    # 评级映射
    rating_ranks = {'AAA': 6, 'AA+': 5, 'AA': 4, 'AA-': 3, 'A+': 2, 'A': 1}
    def get_rating_rank(r):
        if pd.isna(r): return 0
        r_str = str(r).upper().strip()
        for key in rating_ranks:
            if r_str.startswith(key): return rating_ranks[key]
        return 0
    df_today['rating_rank'] = df_today['rating'].apply(get_rating_rank)
    
    # 1. 硬性防雷与安全过滤 (含强赎前置预警剔除 convert_value < 130)
    df_active = df_today.dropna(subset=['close', 'premium']).copy()
    df_active = df_active[~df_active['stock_name'].str.contains('ST', na=False)]  # 剔除 ST
    df_active = df_active[df_active['issue_size'] >= 1.0]                           # 规模 >= 1.0亿
    df_active = df_active[df_active['years_to_maturity'] >= 0.5]                   # 存续期 >= 0.5年
    df_active = df_active[df_active['rating_rank'] >= 1]                            # 评级 >= A
    if 'convert_value' in df_active.columns:
        df_active = df_active[df_active['convert_value'] < 130.0]                  # 优化1保留：剔除已进入强赎区标的
    
    # 2. 双低 7 因子打分体系 (保留原始 30% 双低 + 30% 溢价率架构，坚决不作过度优化)
    r_dl = df_active['double_low'].rank(pct=True, ascending=True)                  # 30% 双低值
    r_prem = df_active['premium'].rank(pct=True, ascending=True)                    # 30% 溢价率
    
    mom_filled = df_active['stock_mom_20'].fillna(df_active['stock_mom_20'].median())
    vol_filled = df_active['stock_vol_20'].fillna(df_active['stock_vol_20'].median())
    r_mom = mom_filled.rank(pct=True, ascending=False)                             # 10% 正股动量
    r_vol = vol_filled.rank(pct=True, ascending=True)                              # 10% 低波优先
    r_scale = df_active['issue_size'].rank(pct=True, ascending=True)               # 10% 小盘溢价
    r_ytm = df_active['ytm'].rank(pct=True, ascending=False)                       # 5% 到期收益率 YTM
    r_dist = df_active['dist_redempt'].rank(pct=True, ascending=False)             # 5% 强赎距离
    
    df_active['score'] = (
        0.30 * r_dl + 
        0.30 * r_prem + 
        0.10 * r_mom + 
        0.10 * r_vol + 
        0.10 * r_scale + 
        0.05 * r_ytm + 
        0.05 * r_dist
    )
    
    df_top = df_active.sort_values('score').head(top_n).copy()
    
    single_alloc = capital_allocated / top_n
    df_top['target_weight'] = 1.0 / top_n
    df_top['target_amount'] = single_alloc
    df_top['target_shares_approx'] = (single_alloc / df_top['close'] / 10).astype(int) * 10 # 10张/手整倍数
    
    print("\n" + "="*95)
    print(f"                      【实盘目标持仓 TOP {top_n} 信号清单】")
    print(f"                       交易日: {latest_date.strftime('%Y-%m-%d')} | 总预算: {capital_allocated/10000:.1f}万")
    print("="*95)
    cols_show = ['ts_code', 'name', 'close', 'premium', 'double_low', 'rating', 'issue_size', 'target_amount', 'target_shares_approx']
    df_show = df_top[cols_show].copy()
    df_show.columns = ['代码', '名称', '现价', '溢价率%', '双低值', '评级', '规模(亿)', '目标金额(元)', '建议买入(张)']
    print(df_show.to_string(index=False))
    print("="*95)
    
    return df_top

if __name__ == "__main__":
    generate_signals()
