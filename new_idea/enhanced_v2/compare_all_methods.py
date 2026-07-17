import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

methods = ['Baseline_5feat', 'Chan_Only', 'Lynch_Only', 'Quant_Only', 'final_best', 'Enhanced_All']

print("=" * 120)
print("所有方法修正前后对比")
print("=" * 120)
print(f"{'方法':<15} {'版本':<8} {'交易':>5} {'胜率':>8} {'平均收益':>8} {'总收益':>10} {'夏普':>6} {'最大回撤':>8}")
print("-" * 120)

results = {}

for m in methods:
    # 原始结果
    orig_file = os.path.join(THIS_DIR, f'equity_{m}.csv')
    orig_trades_file = os.path.join(THIS_DIR, f'trades_{m}.csv')

    if os.path.exists(orig_file) and os.path.exists(orig_trades_file):
        eq_orig = pd.read_csv(orig_file)['equity'].values
        tr_orig = pd.read_csv(orig_trades_file)

        rets = np.diff(eq_orig) / eq_orig[:-1]
        total_ret = eq_orig[-1] / eq_orig[0] - 1
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
        max_dd = np.max(1 - eq_orig / np.maximum.accumulate(eq_orig))

        print(f"{m:<15} {'原始':<8} {len(tr_orig):>5} {tr_orig['ret'].gt(0).mean()*100:>7.1f}% {tr_orig['ret'].mean()*100:>7.2f}% {total_ret*100:>9.1f}% {sharpe:>6.2f} {max_dd*100:>7.1f}%")

        results[m] = {'orig': {'total_ret': total_ret, 'sharpe': sharpe, 'max_dd': max_dd, 'win_rate': tr_orig['ret'].gt(0).mean()}}

    # 修正结果
    fixed_file = os.path.join(THIS_DIR, f'equity_{m}_Fixed.csv')
    fixed_trades_file = os.path.join(THIS_DIR, f'trades_{m}_Fixed.csv')

    if os.path.exists(fixed_file) and os.path.exists(fixed_trades_file):
        eq_fix = pd.read_csv(fixed_file)['equity'].values
        tr_fix = pd.read_csv(fixed_trades_file)

        rets = np.diff(eq_fix) / eq_fix[:-1]
        total_ret = eq_fix[-1] / eq_fix[0] - 1
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
        max_dd = np.max(1 - eq_fix / np.maximum.accumulate(eq_fix))

        print(f"{m:<15} {'修正':<8} {len(tr_fix):>5} {tr_fix['ret'].gt(0).mean()*100:>7.1f}% {tr_fix['ret'].mean()*100:>7.2f}% {total_ret*100:>9.1f}% {sharpe:>6.2f} {max_dd*100:>7.1f}%")

        if m in results:
            results[m]['fixed'] = {'total_ret': total_ret, 'sharpe': sharpe, 'max_dd': max_dd, 'win_rate': tr_fix['ret'].gt(0).mean()}
    else:
        print(f"{m:<15} {'修正':<8} {'N/A':>5} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>6} {'N/A':>8}")

print("=" * 120)

# 绘图
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, m in enumerate(methods):
    ax = axes[i]

    # 原始
    orig_file = os.path.join(THIS_DIR, f'equity_{m}.csv')
    if os.path.exists(orig_file):
        eq = pd.read_csv(orig_file)['equity'].values
        eq_norm = eq / eq[0]
        ax.plot(eq_norm, label='Original', color='#e74c3c', linewidth=2, alpha=0.8)

    # 修正
    fixed_file = os.path.join(THIS_DIR, f'equity_{m}_Fixed.csv')
    if os.path.exists(fixed_file):
        eq = pd.read_csv(fixed_file)['equity'].values
        eq_norm = eq / eq[0]
        ax.plot(eq_norm, label='Fixed', color='#2ecc71', linewidth=2, alpha=0.8)

    ax.set_title(m, fontsize=14, fontweight='bold')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Normalized Equity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
out_path = os.path.join(THIS_DIR, 'all_methods_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n对比图已保存至: {out_path}")
plt.close()
