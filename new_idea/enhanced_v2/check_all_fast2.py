import pandas as pd
import numpy as np
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'

methods = ['Baseline_5feat', 'Chan_Only', 'Lynch_Only', 'Quant_Only', 'final_best']

all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
date_idx = {d: i for i, d in enumerate(all_dates)}

print("=" * 120)
print("所有方法的原始回测问题检查")
print("=" * 120)
print(f"{'方法':<15} {'交易':>5} {'收益>20%':>8} {'创业板':>6} {'科创':>5} {'涨停买':>6} {'跌停卖':>6} {'最高收益':>8} {'最低收益':>8} {'平均收益':>8}")
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

    limit_up = 0
    limit_down = 0

    for idx, row in trades.iterrows():
        d_str = str(int(row['date']))
        ts_code = row['ts_code']

        if d_str not in date_idx:
            continue
        curr_idx = date_idx[d_str]
        if curr_idx + 2 >= len(all_dates):
            continue

        d_t1 = all_dates[curr_idx + 1]
        d_t2 = all_dates[curr_idx + 2]

        try:
            p_t0 = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_str}.parquet"), columns=['ts_code', 'close'])
            p_t1 = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_t1}.parquet"), columns=['ts_code', 'open', 'pre_close'])
            p_t2 = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_t2}.parquet"), columns=['ts_code', 'low', 'open'])
        except:
            continue

        t0_row = p_t0[p_t0['ts_code'] == ts_code]
        t1_row = p_t1[p_t1['ts_code'] == ts_code]
        t2_row = p_t2[p_t2['ts_code'] == ts_code]

        if t0_row.empty or t1_row.empty or t2_row.empty:
            continue

        t0_close = float(t0_row['close'].values[0])
        t1_open = float(t1_row['open'].values[0])
        t1_pre = float(t1_row['pre_close'].values[0]) if 'pre_close' in t1_row.columns else t0_close
        t2_low = float(t2_row['low'].values[0])

        t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
        if t1_open_chg >= 9.5:
            limit_up += 1

        t2_low_chg = (t2_low - t1_open) / t1_open * 100
        if t2_low_chg <= -9.5:
            limit_down += 1

    print(f"{m:<15} {len(trades):>5} {over_20:>8} {cyb:>6} {kcb:>5} {limit_up:>6} {limit_down:>6} {trades['ret'].max()*100:>8.2f}% {trades['ret'].min()*100:>8.2f}% {trades['ret'].mean()*100:>8.2f}%")

print("=" * 120)
