import pandas as pd
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

methods = ['Baseline_5feat', 'Chan_Only', 'Lynch_Only', 'Quant_Only', 'final_best']

print("=" * 120)
print("所有方法的原始回测问题检查 (快速版 - 不读取价格文件)")
print("=" * 120)
print(f"{'方法':<15} {'交易':>5} {'收益>20%':>8} {'创业板':>6} {'科创':>5} {'最高收益':>8} {'最低收益':>8} {'平均收益':>8}")
print("-" * 120)

for m in methods:
    tr_file = os.path.join(THIS_DIR, f'trades_{m}.csv')
    if not os.path.exists(tr_file):
        print(f"{m:<15} 文件不存在")
        continue

    trades = pd.read_csv(tr_file)
    if len(trades) == 0:
        print(f"{m:<15} 无交易")
        continue

    over_20 = len(trades[trades['ret'] > 0.20])
    cyb = len(trades[trades['ts_code'].str.startswith('300') | trades['ts_code'].str.startswith('301')])
    kcb = len(trades[trades['ts_code'].str.startswith('688') | trades['ts_code'].str.startswith('689')])

    print(f"{m:<15} {len(trades):>5} {over_20:>8} {cyb:>6} {kcb:>5} {trades['ret'].max()*100:>8.2f}% {trades['ret'].min()*100:>8.2f}% {trades['ret'].mean()*100:>8.2f}%")

print("=" * 120)
