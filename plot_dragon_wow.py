
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

OUT_DIR = r'C:\Users\liuqi\quant_system_v2'
csv_path = os.path.join(OUT_DIR, 'super_monthly_equity.csv')
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# 计算指标
df['ret'] = df['nav'].pct_change()
total_ret = (df['nav'].iloc[-1] / df['nav'].iloc[0] - 1)
# 假设回测时长为 3.0 年 (2023-2025)
annual_ret = (1 + total_ret) ** (1/3.0) - 1
sharpe = df['ret'].mean() / df['ret'].std() * np.sqrt(252)
df['cummax'] = df['nav'].cummax()
df['drawdown'] = (df['nav'] - df['cummax']) / df['cummax']
max_drawdown = df['drawdown'].min()

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(df.index, df['nav'], color='#1f77b4', linewidth=2.5, label='Super-Monthly (Dragon Focus)')
ax1.fill_between(df.index, df['nav'], 100000, color='#1f77b4', alpha=0.1)
ax1.set_title(f'Dragon-Leading Strategy Performance (Strict T+1)\nTotal Return: {total_ret*100:.1f}% | Annual: {annual_ret*100:.1f}% | Sharpe: {sharpe:.2f}', fontsize=14, fontweight='bold')
ax1.set_ylabel('NAV', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

ax2.fill_between(df.index, df['drawdown']*100, 0, color='#d62728', alpha=0.3, label='Drawdown')
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.set_ylim(-50, 5)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower left')

plt.tight_layout()
save_path = os.path.join(OUT_DIR, 'dragon_strategy_wow.png')
plt.savefig(save_path, dpi=300)
print(f"Metrics: Total={total_ret:.2%}, Annual={annual_ret:.2%}, MDD={max_drawdown:.2%}, Sharpe={sharpe:.2f}")
