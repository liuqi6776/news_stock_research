"""
临时脚本：对比所有策略绩效 + 生成明日信号
"""
import pandas as pd
import numpy as np
import os

OUT_DIR = r'C:\Users\liuqi\quant_system_v2'

def calc_metrics(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    if len(df) < 2: return None
    df['ret'] = df['nav'].pct_change()
    
    total_ret = df['nav'].iloc[-1] / df['nav'].iloc[0] - 1
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365
    if years <= 0: years = 1
    annual_ret = (1 + total_ret) ** (1 / years) - 1
    annual_vol = df['ret'].std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    
    df['cummax'] = df['nav'].cummax()
    df['drawdown'] = (df['nav'] - df['cummax']) / df['cummax']
    mdd = df['drawdown'].min()
    
    # Win rate (daily)
    win_rate = (df['ret'] > 0).sum() / (df['ret'].notna().sum()) if df['ret'].notna().sum() > 0 else 0
    
    return {
        'Total Return': f"{total_ret:.2%}",
        'Annual Return': f"{annual_ret:.2%}",
        'Max Drawdown': f"{mdd:.2%}",
        'Sharpe': f"{sharpe:.2f}",
        'Win Rate': f"{win_rate:.1%}",
        'Last Date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
        'Start Date': df['date'].iloc[0].strftime('%Y-%m-%d'),
        'Days': len(df),
    }

strategy_paths = {
    'Weekly V15 (最新+市场择时+VIX+融资融券)': 'super_weekly_v15_equity.csv',
    'Weekly V13 (周频共振)': 'super_weekly_v13_equity.csv',
    'Weekly V7 (裸跑)': 'super_weekly_equity.csv',
    'Monthly V6.2 (稳健)': 'super_monthly_equity.csv',
    'Monthly Adaptive V82': 'super_monthly_adaptive_v82.csv',
    'Leading V12': 'super_leading_v12_equity.csv',
    'Leading V12.1 (改版)': 'super_leading_v12_1_equity.csv',
    'Micro Adaptive': 'super_micro_adaptive_nav.csv',
    'V6 Close-Close': 'trades_close_close.csv',
    'Weekly Optimized': 'weekly_optimized_equity.csv',
}

print("=" * 100)
print(f"{'策略':<45} | {'总收益':>10} | {'年化':>10} | {'最大回撤':>10} | {'夏普':>8} | {'日胜率':>8} | {'数据截止':>10}")
print("=" * 100)
results = []
for name, filename in strategy_paths.items():
    path = os.path.join(OUT_DIR, filename)
    m = calc_metrics(path)
    if m:
        print(f"{name:<45} | {m['Total Return']:>10} | {m['Annual Return']:>10} | {m['Max Drawdown']:>10} | {m['Sharpe']:>8} | {m['Win Rate']:>8} | {m['Last Date']:>10}")
        results.append((name, m))

print("=" * 100)
print(f"\n共 {len(results)} 个策略有回测数据")
