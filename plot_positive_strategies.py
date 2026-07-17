"""
正收益策略回测净值曲线对比图
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import os

OUT_DIR = r'C:\Users\liuqi\quant_system_v2'

# 尝试加载中文字体
font_paths = [
    r'C:\Windows\Fonts\msyh.ttc',
    r'C:\Windows\Fonts\simhei.ttf',
    r'C:\Windows\Fonts\simsun.ttc',
]
zh_font = None
for fp in font_paths:
    if os.path.exists(fp):
        zh_font = FontProperties(fname=fp)
        break

def load_eq(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    # 去除平台期（连续相同nav值的中间重复）
    df = df.drop_duplicates(subset=['date'], keep='last')
    df = df.sort_values('date').reset_index(drop=True)
    return df

def calc_metrics(df):
    total = df['nav'].iloc[-1] / df['nav'].iloc[0] - 1
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365
    if years <= 0: years = 1
    annual = (1 + total) ** (1/years) - 1
    df['ret'] = df['nav'].pct_change()
    vol = df['ret'].std() * np.sqrt(252) if len(df) > 2 else 0
    sharpe = annual / vol if vol > 0 else 0
    df['cummax'] = df['nav'].cummax()
    dd = (df['nav'] - df['cummax']) / df['cummax']
    mdd = dd.min()
    return total, annual, mdd, sharpe

# 加载所有正收益策略
strategies = {
    'Weekly V7 (Raw)': ('super_weekly_equity.csv', '#FF6B35'),
    'Monthly V6.2 (Steady)': ('super_monthly_equity.csv', '#4ECDC4'),
    'Weekly V15 (Timing+VIX+Margin)': ('super_weekly_v15_equity.csv', '#45B7D1'),
    'Monthly Adaptive V82': ('super_monthly_adaptive_v82.csv', '#96CEB4'),
    'Micro Adaptive': ('super_micro_adaptive_nav.csv', '#FFEAA7'),
}

data = {}
metrics = {}
for name, (fname, color) in strategies.items():
    path = os.path.join(OUT_DIR, fname)
    df = load_eq(path)
    if df is not None and len(df) > 2:
        data[name] = (df, color)
        m = calc_metrics(df)
        metrics[name] = m

# ============ 图1: 净值曲线对比 ============
fig, axes = plt.subplots(2, 2, figsize=(18, 14), gridspec_kw={'hspace': 0.3, 'wspace': 0.25})

# --- (0,0) 全局净值曲线 ---
ax = axes[0, 0]
for name, (df, color) in data.items():
    ax.plot(df['date'], df['nav'] / df['nav'].iloc[0], color=color, linewidth=1.5, label=name)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_title('All Positive Strategies - NAV (Normalized to 1.0)', fontsize=13, fontweight='bold')
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('NAV', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

# --- (0,1) 关键指标柱状图 ---
ax2 = axes[0, 1]
names = list(metrics.keys())
totals = [metrics[n][0]*100 for n in names]
annuals = [metrics[n][1]*100 for n in names]
mdds = [metrics[n][2]*100 for n in names]
sharpes = [metrics[n][3] for n in names]

short_names = [n.split('(')[0].strip() for n in names]
x = np.arange(len(names))
w = 0.2

bars1 = ax2.bar(x - 1.5*w, totals, w, label='Total Return %', color='#FF6B35', alpha=0.85)
bars2 = ax2.bar(x - 0.5*w, annuals, w, label='Annual Return %', color='#4ECDC4', alpha=0.85)
bars3 = ax2.bar(x + 0.5*w, [abs(m) for m in mdds], w, label='|Max DD| %', color='#E74C3C', alpha=0.85)
bars4 = ax2.bar(x + 1.5*w, sharpes, w, label='Sharpe', color='#3498DB', alpha=0.85)

ax2.set_xticks(x)
ax2.set_xticklabels(short_names, rotation=25, ha='right', fontsize=9)
ax2.set_title('Strategy Metrics Comparison', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='black', linewidth=0.5)

# --- (1,0) 去掉V7放大看其他策略 ---
ax3 = axes[1, 0]
for name, (df, color) in data.items():
    if 'V7' in name:
        continue
    nav_norm = df['nav'] / df['nav'].iloc[0]
    ax3.plot(df['date'], nav_norm, color=color, linewidth=1.8, label=name)
    # 标注最终收益
    final_ret = nav_norm.iloc[-1]
    final_date = df['date'].iloc[-1]
    ax3.annotate(f'{final_ret:.2%}', xy=(final_date, final_ret),
                fontsize=9, fontweight='bold', color=color,
                ha='left', va='bottom')

ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax3.set_title('Excluding V7 - Zoom In', fontsize=13, fontweight='bold')
ax3.set_xlabel('Date', fontsize=11)
ax3.set_ylabel('NAV', fontsize=11)
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')

# --- (1,1) 回撤图 ---
ax4 = axes[1, 1]
for name, (df, color) in data.items():
    dd_series = (df['nav'] - df['nav'].cummax()) / df['nav'].cummax()
    ax4.fill_between(df['date'], dd_series, alpha=0.15, color=color)
    ax4.plot(df['date'], dd_series, color=color, linewidth=1.0, label=name)

ax4.set_title('Drawdown Comparison', fontsize=13, fontweight='bold')
ax4.set_xlabel('Date', fontsize=11)
ax4.set_ylabel('Drawdown', fontsize=11)
ax4.legend(fontsize=9, loc='lower left')
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.suptitle('Quant System V2 - Positive Return Strategies Backtest Review',
             fontsize=16, fontweight='bold', y=0.98)

save_path = os.path.join(OUT_DIR, 'positive_strategies_review.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Chart saved: {save_path}')

# ============ 图2: 年度收益热力图 ============
fig2, ax5 = plt.subplots(figsize=(14, 5))

yearly_returns = {}
for name, (df, color) in data.items():
    df_y = df.copy()
    df_y['year'] = df_y['date'].dt.year
    # 每年最后一个交易日的nav vs 该年第一个
    yearly = []
    for year, g in df_y.groupby('year'):
        if len(g) >= 2:
            ret = g['nav'].iloc[-1] / g['nav'].iloc[0] - 1
            yearly.append((year, ret))
    yearly_returns[name] = dict(yearly)

# 表格展示
print("\n" + "="*90)
print(f"{'Strategy':<40} | {'Total':>10} | {'Annual':>10} | {'Max DD':>10} | {'Sharpe':>8}")
print("="*90)
for name in metrics:
    t, a, m, s = metrics[name]
    print(f"{name:<40} | {t:>9.2%} | {a:>9.2%} | {m:>9.2%} | {s:>8.2f}")
print("="*90)

# 年度收益表
all_years = sorted(set(y for d in yearly_returns.values() for y in d.keys()))
print(f"\n{'Strategy':<40} | " + " | ".join(f"{y}" for y in all_years))
print("-"*90)
for name in metrics:
    row = yearly_returns.get(name, {})
    vals = [f"{row.get(y, 0)*100:>6.1f}%" for y in all_years]
    print(f"{name:<40} | " + " | ".join(vals))
print("-"*90)

plt.close('all')
print("\nDone!")
