import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao')

strategies = [
    {
        'name': 'Baseline (Current)',
        'return': 1047.25,
        'sharpe': 6.38,
        'mdd': -30.64,
        'color': '#1f77b4'
    },
    {
        'name': 'StopLoss 5%',
        'return': 950.0,
        'sharpe': 6.5,
        'mdd': -25.0,
        'color': '#ff7f0e'
    },
    {
        'name': 'MinMV 10亿',
        'return': 900.0,
        'sharpe': 6.4,
        'mdd': -28.0,
        'color': '#2ca02c'
    },
    {
        'name': 'StopLoss 5% + MinMV 10亿',
        'return': 800.0,
        'sharpe': 7.0,
        'mdd': -22.0,
        'color': '#d62728'
    }
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Strategy Optimization - Expected Results', fontsize=18, fontweight='bold')

names = [s['name'] for s in strategies]
colors = [s['color'] for s in strategies]

ax1 = axes[0, 0]
returns = [s['return'] for s in strategies]
bars = ax1.bar(names, returns, color=colors)
ax1.set_ylabel('Total Return (%)', fontsize=12)
ax1.set_title('Total Return Comparison', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, returns):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{val:+.1f}%', ha='center', va='bottom')

ax2 = axes[0, 1]
sharpes = [s['sharpe'] for s in strategies]
bars = ax2.bar(names, sharpes, color=colors)
ax2.set_ylabel('Sharpe Ratio', fontsize=12)
ax2.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, sharpes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}', ha='center', va='bottom')

ax3 = axes[1, 0]
mdds = [s['mdd'] for s in strategies]
bars = ax3.bar(names, [-m for m in mdds], color=colors)
ax3.set_ylabel('Max Drawdown (%)', fontsize=12)
ax3.set_title('Max Drawdown Comparison', fontsize=14, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, mdds):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}%', ha='center', va='bottom')

ax4 = axes[1, 1]
x = [s['return'] for s in strategies]
y = [-s['mdd'] for s in strategies]
sizes = [s['sharpe'] * 100 for s in strategies]
scatter = ax4.scatter(x, y, s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
for i, s in enumerate(strategies):
    ax4.annotate(s['name'], (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
ax4.set_xlabel('Total Return (%)', fontsize=12)
ax4.set_ylabel('Max Drawdown (%, lower is better)', fontsize=12)
ax4.set_title('Risk-Reward Tradeoff', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_plot = os.path.join(OUTPUT_DIR, 'optimization_expected_results.png')
plt.savefig(output_plot, dpi=150)
print(f"Plot saved: {output_plot}")

summary_df = pd.DataFrame(strategies)
summary_df = summary_df[['name', 'return', 'sharpe', 'mdd']]
summary_df.columns = ['Strategy', 'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
summary_csv = os.path.join(OUTPUT_DIR, 'optimization_expected_summary.csv')
summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
print(f"Summary saved: {summary_csv}")
