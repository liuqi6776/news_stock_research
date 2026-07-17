import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# 读取原始和修正后的数据
eq_original = pd.read_csv(os.path.join(THIS_DIR, 'equity_Enhanced_All.csv'))
eq_fixed = pd.read_csv(os.path.join(THIS_DIR, 'equity_Enhanced_All_Fixed.csv'))
tr_original = pd.read_csv(os.path.join(THIS_DIR, 'trades_Enhanced_All.csv'))
tr_fixed = pd.read_csv(os.path.join(THIS_DIR, 'trades_Enhanced_All_Fixed.csv'))

# 归一化
eq_orig = eq_original['equity'].values / eq_original['equity'].values[0]
eq_fix = eq_fixed['equity'].values / eq_fixed['equity'].values[0]

# 计算指标
def calc_stats(eq, trades):
    eq_arr = eq
    rets = np.diff(eq_arr) / eq_arr[:-1]
    total_ret = eq_arr[-1] / eq_arr[0] - 1
    sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
    max_dd = np.max(1 - eq_arr / np.maximum.accumulate(eq_arr))
    win_rate = (trades['ret'] > 0).mean()
    avg_ret = trades['ret'].mean()
    n_trades = len(trades)
    return {
        'total_ret': total_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'avg_ret': avg_ret,
        'n_trades': n_trades
    }

stats_orig = calc_stats(eq_orig, tr_original)
stats_fix = calc_stats(eq_fix, tr_fixed)

print("=" * 80)
print("Enhanced_All 修正前后对比")
print("=" * 80)
print(f"{'指标':<20} {'原始回测':>15} {'修正后回测':>15} {'差异':>15}")
print("-" * 80)
print(f"{'交易次数':<20} {stats_orig['n_trades']:>15} {stats_fix['n_trades']:>15} {stats_fix['n_trades']-stats_orig['n_trades']:>15}")
print(f"{'总收益':<20} {stats_orig['total_ret']*100:>14.2f}% {stats_fix['total_ret']*100:>14.2f}% {(stats_fix['total_ret']-stats_orig['total_ret'])*100:>14.2f}%")
print(f"{'夏普比率':<20} {stats_orig['sharpe']:>15.2f} {stats_fix['sharpe']:>15.2f} {stats_fix['sharpe']-stats_orig['sharpe']:>15.2f}")
print(f"{'最大回撤':<20} {stats_orig['max_dd']*100:>14.2f}% {stats_fix['max_dd']*100:>14.2f}% {(stats_fix['max_dd']-stats_orig['max_dd'])*100:>14.2f}%")
print(f"{'胜率':<20} {stats_orig['win_rate']*100:>14.2f}% {stats_fix['win_rate']*100:>14.2f}% {(stats_fix['win_rate']-stats_orig['win_rate'])*100:>14.2f}%")
print(f"{'平均收益':<20} {stats_orig['avg_ret']*100:>14.2f}% {stats_fix['avg_ret']*100:>14.2f}% {(stats_fix['avg_ret']-stats_orig['avg_ret'])*100:>14.2f}%")

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})

ax1 = axes[0]
ax1.plot(eq_orig, label='Original (Unrealistic)', color='#e74c3c', linewidth=2)
ax1.plot(eq_fix, label='Fixed (A-Share Rules)', color='#2ecc71', linewidth=2)
ax1.set_title('Enhanced_All: Original vs Fixed Backtest', fontsize=16, fontweight='bold')
ax1.set_xlabel('Trading Days', fontsize=12)
ax1.set_ylabel('Normalized Equity (Start = 1.0)', fontsize=12)
ax1.legend(fontsize=12, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

# 添加注释
ax1.annotate(f'Original: {stats_orig["total_ret"]*100:.0f}%', xy=(len(eq_orig)-1, eq_orig[-1]),
             xytext=(10, 0), textcoords='offset points', fontsize=10, color='#e74c3c')
ax1.annotate(f'Fixed: {stats_fix["total_ret"]*100:.1f}%', xy=(len(eq_fix)-1, eq_fix[-1]),
             xytext=(10, 0), textcoords='offset points', fontsize=10, color='#2ecc71')

ax2 = axes[1]
metrics = ['Total Return', 'Sharpe', 'Win Rate', 'Max DD']
x = np.arange(len(metrics))
width = 0.35

orig_vals = [stats_orig['total_ret']*100, stats_orig['sharpe']*10, stats_orig['win_rate']*100, stats_orig['max_dd']*100]
fix_vals = [stats_fix['total_ret']*100, stats_fix['sharpe']*10, stats_fix['win_rate']*100, stats_fix['max_dd']*100]

bars1 = ax2.bar(x - width/2, orig_vals, width, label='Original', color='#e74c3c', alpha=0.8)
bars2 = ax2.bar(x + width/2, fix_vals, width, label='Fixed', color='#2ecc71', alpha=0.8)

ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=11)
ax2.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# 添加数值标签
for bar in bars1:
    h = bar.get_height()
    ax2.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    h = bar.get_height()
    ax2.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
out_path = os.path.join(THIS_DIR, 'enhanced_fixed_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n对比图已保存至: {out_path}")
plt.close()
