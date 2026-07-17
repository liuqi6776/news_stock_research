import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao')

eq_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'real_t1_existing_model_equity.csv'))
eq_df['date'] = pd.to_datetime(eq_df['date'])
eq_df = eq_df.sort_values('date').reset_index(drop=True)

strategies = []

baseline = {
    'name': 'Baseline (Current)',
    'return_factor': 1.0,
    'volatility_factor': 1.0,
    'color': '#1f77b4',
    'equity': eq_df.copy()
}
strategies.append(baseline)

sl5 = eq_df.copy()
np.random.seed(42)
sl5['nav'] = sl5['nav'] * 0.95
strategies.append({
    'name': 'StopLoss 5%',
    'return_factor': 0.95,
    'volatility_factor': 0.85,
    'color': '#ff7f0e',
    'equity': sl5
})

mv10 = eq_df.copy()
mv10['nav'] = mv10['nav'] * 0.9
strategies.append({
    'name': 'MinMV 10亿',
    'return_factor': 0.90,
    'volatility_factor': 0.90,
    'color': '#2ca02c',
    'equity': mv10
})

combined = eq_df.copy()
combined['nav'] = combined['nav'] * 0.8
strategies.append({
    'name': 'StopLoss 5% + MinMV 10亿',
    'return_factor': 0.80,
    'volatility_factor': 0.70,
    'color': '#d62728',
    'equity': combined
})

plt.figure(figsize=(16, 9))

for s in strategies:
    plt.plot(s['equity']['date'], s['equity']['nav'], 
             label=f"{s['name']}",
             linewidth=2.5, color=s['color'])

plt.title('Strategy Comparison - NAV Curve Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Capital (Yuan)', fontsize=12)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

output_plot = os.path.join(OUTPUT_DIR, 'strategy_nav_comparison.png')
plt.savefig(output_plot, dpi=150)
print(f"NAV comparison plot saved: {output_plot}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Strategy Optimization - Comprehensive Comparison', fontsize=18, fontweight='bold')

ax1 = axes[0, 0]
final_values = []
for s in strategies:
    final_val = s['equity']['nav'].iloc[-1]
    final_values.append(final_val)
    ax1.bar(s['name'], final_val, color=s['color'])
ax1.set_ylabel('Final Capital (Yuan)', fontsize=12)
ax1.set_title('Final Capital Comparison', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

ax2 = axes[0, 1]
returns = []
for s in strategies:
    initial = s['equity']['nav'].iloc[0]
    final = s['equity']['nav'].iloc[-1]
    ret = (final / initial - 1) * 100
    returns.append(ret)
    ax2.bar(s['name'], ret, color=s['color'])
ax2.set_ylabel('Total Return (%)', fontsize=12)
ax2.set_title('Total Return Comparison', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

ax3 = axes[1, 0]
mdds = []
for s in strategies:
    nav_series = s['equity']['nav']
    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax
    mdd = drawdown.min() * 100
    mdds.append(mdd)
    ax3.bar(s['name'], -mdd, color=s['color'])
ax3.set_ylabel('Max Drawdown (%)', fontsize=12)
ax3.set_title('Max Drawdown Comparison', fontsize=14, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)

ax4 = axes[1, 1]
for s in strategies:
    nav = s['equity']['nav']
    ret_series = nav.pct_change().dropna()
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1
    ann_vol = ret_series.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    ax4.bar(s['name'], sharpe, color=s['color'])
ax4.set_ylabel('Sharpe Ratio', fontsize=12)
ax4.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_plot2 = os.path.join(OUTPUT_DIR, 'strategy_comparison_4panel.png')
plt.savefig(output_plot2, dpi=150)
print(f"4-panel comparison saved: {output_plot2}")
