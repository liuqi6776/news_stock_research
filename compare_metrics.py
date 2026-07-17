
import pandas as pd
import numpy as np
import os

OUT_DIR = r'C:\Users\liuqi\quant_system_v2'

def calc_metrics(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['ret'] = df['nav'].pct_change()
    
    total_ret = (df['nav'].iloc[-1] / df['nav'].iloc[0] - 1)
    annual_ret = (1 + total_ret) ** (1/3.0) - 1
    annual_vol = df['ret'].std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if (annual_vol is not None and annual_vol > 0) else 0
    
    df['cummax'] = df['nav'].cummax()
    df['drawdown'] = (df['nav'] - df['cummax']) / df['cummax']
    mdd = df['drawdown'].min()
    
    return {
        'Total Return': f"{total_ret:.2%}",
        'Annual Return': f"{annual_ret:.2%}",
        'Max Drawdown': f"{mdd:.2%}",
        'Sharpe': f"{sharpe:.2f}"
    }

strategy_paths = {
    'Super-Weekly (V7) - 裸跑': os.path.join(OUT_DIR, 'super_weekly_equity.csv'),
    'Super-Monthly (V6.2) - 稳健': os.path.join(OUT_DIR, 'super_monthly_equity.csv'),
    'Weekly-Leading (V12.1) - 改版': os.path.join(OUT_DIR, 'super_leading_v12_1_equity.csv')
}

print(f"{'策略版本':<25} | {'总收益':<10} | {'最大回撤':<10} | {'夏普比率':<10}")
print("-" * 65)
for name, path in strategy_paths.items():
    m = calc_metrics(path)
    if m:
        print(f"{name:<25} | {m['Total Return']:<10} | {m['Max Drawdown']:<10} | {m['Sharpe']:<10}")
