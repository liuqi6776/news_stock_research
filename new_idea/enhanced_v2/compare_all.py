import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

models = ['Baseline_5feat', 'Enhanced_All', 'Chan_Only', 'Lynch_Only', 'Quant_Only', 'final_best']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
labels = ['Baseline (5feat)', 'Enhanced (All)', 'Chan Only', 'Lynch Only', 'Quant Only', 'Final Best']

equity_data = {}
stats_data = {}

for m in models:
    eq_file = os.path.join(THIS_DIR, f"equity_{m}.csv")
    tr_file = os.path.join(THIS_DIR, f"trades_{m}.csv")
    try:
        eq = pd.read_csv(eq_file)
        eq_arr = eq["equity"].values
        norm_eq = eq_arr / eq_arr[0]
        equity_data[m] = norm_eq

        tr = pd.read_csv(tr_file)
        rets = np.diff(eq_arr) / eq_arr[:-1]
        total_ret = eq_arr[-1] / eq_arr[0] - 1
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
        max_dd = np.max(1 - eq_arr / np.maximum.accumulate(eq_arr))
        win_rate = (tr["ret"] > 0).mean()
        avg_ret = tr["ret"].mean()
        n_trades = len(tr)
        stats_data[m] = {
            'sharpe': sharpe,
            'total_ret': total_ret,
            'win_rate': win_rate,
            'avg_ret': avg_ret,
            'n_trades': n_trades,
            'max_dd': max_dd
        }
    except Exception as e:
        print(f"  {m}: Error - {e}")

print("=" * 100)
print(f"{'Model':<22} {'Sharpe':>8} {'TotalRet':>10} {'WinRate':>8} {'AvgRet':>8} {'Trades':>7} {'MaxDD':>8}")
print("-" * 100)
for m in models:
    if m in stats_data:
        s = stats_data[m]
        print(f"{m:<22} {s['sharpe']:>8.2f} {s['total_ret']*100:>9.2f}% {s['win_rate']*100:>7.2f}% {s['avg_ret']*100:>7.2f}% {s['n_trades']:>7d} {s['max_dd']*100:>7.2f}%")

best_model = max(stats_data, key=lambda x: stats_data[x]['total_ret'])
print(f"\n>>> 最高收益方法: {best_model} (总收益: {stats_data[best_model]['total_ret']*100:.2f}%)")

best_sharpe = max(stats_data, key=lambda x: stats_data[x]['sharpe'])
print(f">>> 最高夏普方法: {best_sharpe} (Sharpe: {stats_data[best_sharpe]['sharpe']:.2f})")

fig, axes = plt.subplots(2, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 1]})

ax1 = axes[0]
for i, m in enumerate(models):
    if m in equity_data:
        ax1.plot(equity_data[m], label=labels[i], color=colors[i], linewidth=1.5)

ax1.set_title('Equity Curve Comparison - All Methods', fontsize=16, fontweight='bold')
ax1.set_xlabel('Trading Days', fontsize=12)
ax1.set_ylabel('Normalized Equity (Start = 1.0)', fontsize=12)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

ax2 = axes[1]
metric_names = ['Total Return', 'Sharpe', 'Win Rate', 'Max DD']
x = np.arange(len(models))
width = 0.2

model_labels_short = ['Baseline', 'Enhanced', 'Chan', 'Lynch', 'Quant', 'FinalBest']
total_rets = [stats_data[m]['total_ret']*100 if m in stats_data else 0 for m in models]
sharpes = [stats_data[m]['sharpe'] if m in stats_data else 0 for m in models]
win_rates = [stats_data[m]['win_rate']*100 if m in stats_data else 0 for m in models]
max_dds = [stats_data[m]['max_dd']*100 if m in stats_data else 0 for m in models]

bars1 = ax2.bar(x - 1.5*width, total_rets, width, label='Total Return %', color='#2ecc71', alpha=0.8)
bars2 = ax2.bar(x - 0.5*width, [s*10 for s in sharpes], width, label='Sharpe x10', color='#3498db', alpha=0.8)
bars3 = ax2.bar(x + 0.5*width, win_rates, width, label='Win Rate %', color='#f39c12', alpha=0.8)
bars4 = ax2.bar(x + 1.5*width, max_dds, width, label='Max DD %', color='#e74c3c', alpha=0.8)

ax2.set_xticks(x)
ax2.set_xticklabels(model_labels_short, fontsize=10)
ax2.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    h = bar.get_height()
    if h != 0:
        ax2.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

plt.tight_layout()
out_path = os.path.join(THIS_DIR, 'equity_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nChart saved to: {out_path}")
plt.close()
