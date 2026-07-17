"""
V16 vs V15 vs V7 全策略对比图
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = r'C:\Users\liuqi\quant_system_v2'

# 加载所有净值数据
files = {
    'Weekly V7 (Raw)': 'super_weekly_equity.csv',
    'Monthly V6.2': 'super_monthly_equity.csv',
    'Weekly V15': 'super_weekly_v15_equity.csv',
    'Monthly Adaptive V82': 'super_monthly_adaptive_v82.csv',
    'Micro Adaptive': 'super_micro_adaptive_nav.csv',
    'V16 Adaptive (NEW)': 'super_weekly_v16_equity.csv',
}

navs = {}
for name, f in files.items():
    try:
        df = pd.read_csv(f'{OUT_DIR}/{f}')
        df['date'] = pd.to_datetime(df['date'])
        navs[name] = df.set_index('date')['nav']
    except: continue

# 统一日期范围
all_dates = sorted(set().union(*[v.index for v in navs.values()]))
nav_df = pd.DataFrame({k: v.reindex(all_dates).ffill() for k, v in navs.items()})
nav_df = nav_df.dropna(how='all').fillna(method='ffill')

# 归一化
nav_norm = nav_df / nav_df.iloc[0] * 100

# 计算指标
metrics = {}
for col in nav_df.columns:
    s = nav_df[col].dropna()
    if len(s) < 2: continue
    total_ret = (s.iloc[-1] / s.iloc[0] - 1) * 100
    days = (s.index[-1] - s.index[0]).days
    annual_ret = ((s.iloc[-1] / s.iloc[0]) ** (365/days) - 1) * 100
    peak = s.cummax()
    dd = (s - peak) / peak
    max_dd = dd.min() * 100
    daily_ret = s.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
    
    # 年度收益
    yearly = s.resample('A').apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 1 else 0)
    
    metrics[col] = {
        'Total': total_ret, 'Annual': annual_ret, 'MaxDD': max_dd,
        'Sharpe': sharpe, 'Calmar': calmar, 'Yearly': yearly
    }

# ===== 绘图 =====
fig, axes = plt.subplots(2, 2, figsize=(18, 13))

# 1. 净值曲线
ax1 = axes[0, 0]
colors = {'Weekly V7 (Raw)': '#FF6B35', 'Monthly V6.2': '#4ECDC4',
          'Weekly V15': '#45B7D1', 'V16 Adaptive (NEW)': '#FF1744',
          'Monthly Adaptive V82': '#96CEB4', 'Micro Adaptive': '#FFEAA7'}
for col in nav_norm.columns:
    lw = 2.5 if 'NEW' in col else 1.2
    ax1.plot(nav_norm.index, nav_norm[col], label=col, linewidth=lw, color=colors.get(col, 'gray'))
ax1.set_title('All Strategies - Normalized NAV (100)', fontsize=13, fontweight='bold')
ax1.set_ylabel('NAV')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. V16 vs V15 vs V7 放大对比
ax2 = axes[0, 1]
for col in ['Weekly V7 (Raw)', 'Weekly V15', 'V16 Adaptive (NEW)']:
    if col in nav_norm.columns:
        ax2.plot(nav_norm.index, nav_norm[col], label=col, linewidth=2, color=colors.get(col, 'gray'))
ax2.set_title('V16 vs V15 vs V7 - Focus Comparison', fontsize=13, fontweight='bold')
ax2.set_ylabel('NAV')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. 关键指标柱状图
ax3 = axes[1, 0]
strat_names = list(metrics.keys())
total_rets = [metrics[s]['Total'] for s in strat_names]
max_dds = [metrics[s]['MaxDD'] for s in strat_names]
sharpes = [metrics[s]['Sharpe'] for s in strat_names]
calmars = [metrics[s]['Calmar'] for s in strat_names]

x = np.arange(len(strat_names))
w = 0.2
bars1 = ax3.bar(x - 1.5*w, total_rets, w, label='Total Return %', color='#4CAF50', alpha=0.8)
bars2 = ax3.bar(x - 0.5*w, max_dds, w, label='Max Drawdown %', color='#F44336', alpha=0.8)
bars3 = ax3.bar(x + 0.5*w, [s*20 for s in sharpes], w, label='Sharpe x20', color='#2196F3', alpha=0.8)
bars4 = ax3.bar(x + 1.5*w, [c*20 for c in calmars], w, label='Calmar x20', color='#FF9800', alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels([n.replace(' (NEW)', '\n(NEW)').replace('Weekly ', '').replace('Monthly ', '') for n in strat_names], fontsize=8)
ax3.set_title('Key Metrics Comparison', fontsize=13, fontweight='bold')
ax3.legend(fontsize=8)
ax3.axhline(y=0, color='black', linewidth=0.5)
ax3.grid(True, alpha=0.3, axis='y')

# 4. 年度收益热力图
ax4 = axes[1, 1]
years = sorted(set().union(*[m['Yearly'].index.year for m in metrics.values()]))
yearly_data = []
for name in strat_names:
    yd = metrics[name]['Yearly']
    row = []
    for y in years:
        mask = yd.index.year == y
        if mask.any():
            row.append(yd[mask].iloc[0])
        else:
            row.append(np.nan)
    yearly_data.append(row)
yearly_df = pd.DataFrame(yearly_data, index=[n.replace(' (NEW)', '*') for n in strat_names], columns=years)

im = ax4.imshow(yearly_df.values, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=60)
ax4.set_xticks(range(len(years)))
ax4.set_xticklabels(years)
ax4.set_yticks(range(len(yearly_df)))
short_names = [n.replace('Weekly ', 'Wk ').replace('Monthly ', 'Mo ').replace(' Adaptive (NEW)', ' V16*') for n in yearly_df.index]
ax4.set_yticklabels(short_names, fontsize=8)
# 添加数值标注
for i in range(len(yearly_df)):
    for j in range(len(years)):
        val = yearly_df.iloc[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 25 else 'black'
            ax4.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=7, color=color)
ax4.set_title('Yearly Returns (%) - Green=Bull, Red=Bear', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax4, label='Return %')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/v16_vs_all_strategies.png', dpi=150, bbox_inches='tight')
print("Chart saved: v16_vs_all_strategies.png")

# 打印指标表
print(f"\n{'='*90}")
print(f"{'Strategy':<30} | {'Total':>8} | {'Annual':>8} | {'Max DD':>8} | {'Sharpe':>7} | {'Calmar':>7}")
print(f"{'-'*90}")
for name in strat_names:
    m = metrics[name]
    marker = ' <-- NEW' if 'NEW' in name else ''
    print(f"{name:<30} | {m['Total']:>+7.1f}% | {m['Annual']:>+7.1f}% | {m['MaxDD']:>7.1f}% | {m['Sharpe']:>7.2f} | {m['Calmar']:>7.2f}{marker}")
print(f"{'='*90}")

# 年度收益
header = f"\n{'Strategy':<30} | "
for y in years: header += f"{y:>7} | "
print(header, end='')
print(f"\n{'-'*90}")
for name in strat_names:
    print(f"{name:<30} | ", end='')
    for y in years:
        yd = metrics[name]['Yearly']
        mask = yd.index.year == y
        if mask.any():
            print(f"{yd[mask].iloc[0]:>+6.1f}% |", end='')
        else:
            print(f"{'N/A':>7} |", end='')
    print()
