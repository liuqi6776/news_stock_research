# -*- coding: utf-8 -*-
"""
cb_risk_scorecard.py - 可转债风险评分卡与仓位决策引擎
=====================================================

本脚本实现对 A 股可转债市场日频宏观风险因子的测算，并输出仓位决策：
1. 市场双底均值 (Market Double-Low Mean)
2. 市场双底均值 20 交易日变化率 (20d Momentum)
3. 信用利差代理指标 (Credit Spread Proxy): Mean(AA- YTM) - Mean(AAA YTM)
   - YTM 估算公式同 backtest_cb_doublelow.py
4. 综合仓位决策系数 signal_coef:
   - 绿灯 (1.0): 估值与信用安全
   - 黄灯 (0.5): 风险有所积聚，仓位减半防守
   - 红灯 (0.0): 极端泡沫或重大违约信用风暴，空仓持币避险
"""

import os
import pandas as pd
import numpy as np

def calculate_scorecard_signals(df_pit=None, cache_dir=None):
    """
    计算每日转债风险评分卡指标与信号系数。
    无前视偏差设计：所有滚动指标、百分位数均仅基于 t 日及之前历史数据。
    """
    if df_pit is None:
        if cache_dir is None:
            # 自动定位 cache 目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "cache"))
        parquet_path = os.path.join(cache_dir, 'cb_pit_daily.parquet')
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"未找到可转债PIT数据: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        df = df_pit.copy()

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 1. 计算 YTM
    df['years_to_maturity'] = (df['list_date'] + pd.to_timedelta(2191.5, unit='D') - df['trade_date']).dt.days / 365.25
    df['years_to_maturity'] = df['years_to_maturity'].clip(lower=0.1)
    df['ytm'] = (118.0 - df['close']) / (df['close'] * df['years_to_maturity'])

    # 2. 计算双底值
    df['double_low'] = df['close'] + df['premium']

    # 3. 按交易日截面聚合计算宏观指标
    print("计算每日截面转债宏观风险指标...")
    
    # 评级标准化
    def clean_rating(r):
        if pd.isna(r):
            return "UNKNOWN"
        r_str = str(r).upper().strip()
        for key in ['AAA', 'AA+', 'AA-', 'AA', 'A+', 'A-', 'A']:
            if r_str.startswith(key):
                return key
        return "UNKNOWN"
    
    df['clean_rating'] = df['rating'].apply(clean_rating)
    
    # 每日双底均值与中位数
    g_date = df.groupby('trade_date')
    daily_stats = g_date['double_low'].agg(['mean', 'median']).rename(columns={'mean': 'market_dl_mean', 'median': 'market_dl_median'})
    
    # 过滤出债性较强的样本计算信用利差，避免股性期权价值扭曲 YTM
    df_debt = df[df['close'] < 110].copy()
    aaa_ytm = df_debt[df_debt['clean_rating'] == 'AAA'].groupby('trade_date')['ytm'].mean()
    aa_minus_ytm = df_debt[df_debt['clean_rating'] == 'AA-'].groupby('trade_date')['ytm'].mean()
    
    daily_stats['aaa_ytm'] = aaa_ytm
    daily_stats['aa_minus_ytm'] = aa_minus_ytm
    
    # 用前值填充缺失的评级 YTM (避免当天无此评级债导致的空值)
    daily_stats['aaa_ytm'] = daily_stats['aaa_ytm'].ffill().fillna(0.0)
    daily_stats['aa_minus_ytm'] = daily_stats['aa_minus_ytm'].ffill().fillna(0.0)
    
    # 信用利差代理：AA- 债与 AAA 债的 YTM 差值
    daily_stats['credit_spread'] = daily_stats['aa_minus_ytm'] - daily_stats['aaa_ytm']
    # 限制合理范围并平滑以过滤高频噪音
    daily_stats['credit_spread'] = daily_stats['credit_spread'].clip(lower=0).rolling(5, min_periods=1).mean()
    
    # 4. 计算特征与变化动量
    daily_stats['dl_mean_diff_20'] = daily_stats['market_dl_mean'] - daily_stats['market_dl_mean'].shift(20)
    
    # 滚动历史 250 天信用利差的 90% 分位数（作为信用危机警戒线）
    daily_stats['credit_spread_limit'] = daily_stats['credit_spread'].rolling(250, min_periods=60).quantile(0.90)
    daily_stats['credit_spread_limit'] = daily_stats['credit_spread_limit'].fillna(0.03)

    # 4. 计算滚动百分位数做动态估值阀值 (基于网格搜索寻优参数)
    daily_stats['mkt_dl_mean_low'] = daily_stats['market_dl_mean'].rolling(250, min_periods=60).quantile(0.20)
    daily_stats['mkt_dl_mean_high'] = daily_stats['market_dl_mean'].rolling(250, min_periods=60).quantile(0.80)
    daily_stats['mkt_dl_mean_low'] = daily_stats['mkt_dl_mean_low'].fillna(125.0)
    daily_stats['mkt_dl_mean_high'] = daily_stats['mkt_dl_mean_high'].fillna(140.0)

    # 5. 仓位控制信号逻辑 (Green / Yellow / Red)
    # coef 初始全部设为绿灯 1.0
    daily_stats['signal_coef'] = 1.0
    daily_stats['reason'] = 'Green (估值与信用安全)'
    
    # 条件判定
    for dt, row in daily_stats.iterrows():
        low_thresh = max(125.0, row['mkt_dl_mean_low'])
        high_thresh = max(140.0, row['mkt_dl_mean_high'])
        
        # (a) 黄灯阈值：估值偏高，或信用利差抬头，或估值开始负向动量
        is_yellow = (
            (row['market_dl_mean'] > low_thresh) or
            (row['credit_spread'] > 0.85 * row['credit_spread_limit']) or
            (-8.0 < row['dl_mean_diff_20'] <= -4.0)
        )
        
        # (b) 红灯阈值：估值严重泡沫，或爆发严重信用风暴，或估值雪崩
        is_red = (
            (row['market_dl_mean'] > high_thresh) or
            (row['credit_spread'] > 1.1 * row['credit_spread_limit']) or
            (row['dl_mean_diff_20'] <= -8.0)
        )
        
        if is_red:
            daily_stats.loc[dt, 'signal_coef'] = 0.0
            reasons = []
            if row['market_dl_mean'] > high_thresh: reasons.append("估值动态偏高")
            if row['credit_spread'] > 1.1 * row['credit_spread_limit']: reasons.append("信用风险红线")
            if row['dl_mean_diff_20'] <= -8.0: reasons.append("估值大跌动量")
            daily_stats.loc[dt, 'reason'] = f"Red ({'+'.join(reasons)})"
        elif is_yellow:
            daily_stats.loc[dt, 'signal_coef'] = 0.5
            reasons = []
            if row['market_dl_mean'] > low_thresh: reasons.append("估值动态中性")
            if row['credit_spread'] > 0.85 * row['credit_spread_limit']: reasons.append("信用利差走阔")
            if -8.0 < row['dl_mean_diff_20'] <= -4.0: reasons.append("估值下跌")
            daily_stats.loc[dt, 'reason'] = f"Yellow ({'+'.join(reasons)})"
            
    # 填充暖机期前置数据
    daily_stats['dl_mean_diff_20'] = daily_stats['dl_mean_diff_20'].fillna(0.0)
    
    return daily_stats.reset_index()

if __name__ == "__main__":
    try:
        daily_stats = calculate_scorecard_signals()
        print(f"成功运行评分卡，生成 {len(daily_stats)} 天数据。")
        print("\n最新 10 个交易日评分卡指标状态:")
        print(daily_stats[['trade_date', 'market_dl_mean', 'credit_spread', 'dl_mean_diff_20', 'signal_coef', 'reason']].tail(10).to_string(index=False))
    except Exception as e:
        print(f"评分卡测试失败: {e}")
