import pandas as pd
import numpy as np
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'

methods = ['Baseline_5feat', 'Chan_Only', 'Lynch_Only', 'Quant_Only', 'final_best']

print("=" * 100)
print("所有方法的原始回测问题检查")
print("=" * 100)

for m in methods:
    tr_file = os.path.join(THIS_DIR, f'trades_{m}.csv')
    if not os.path.exists(tr_file):
        print(f"\n{m}: 文件不存在")
        continue

    trades = pd.read_csv(tr_file)
    if len(trades) == 0:
        print(f"\n{m}: 无交易")
        continue

    print(f"\n{'='*60}")
    print(f"方法: {m} (交易数: {len(trades)})")
    print(f"{'='*60}")

    # 1. 单日收益超过20%
    over_20 = trades[trades['ret'] > 0.20]
    print(f"  收益>20%: {len(over_20)}笔")

    # 2. 创业板/科创板
    cyb = trades[trades['ts_code'].str.startswith('300') | trades['ts_code'].str.startswith('301')]
    kcb = trades[trades['ts_code'].str.startswith('688') | trades['ts_code'].str.startswith('689')]
    print(f"  创业板(300/301): {len(cyb)}笔")
    print(f"  科创板(688/689): {len(kcb)}笔")

    # 3. 检查涨跌停 (需要读取价格数据)
    limit_up = 0
    limit_down = 0

    for idx, row in trades.iterrows():
        d_str = str(int(row['date']))
        ts_code = row['ts_code']

        all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
        if d_str not in all_dates:
            continue
        curr_idx = all_dates.index(d_str)
        if curr_idx + 2 >= len(all_dates):
            continue

        d_t1 = all_dates[curr_idx + 1]

        try:
            p_t0 = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_str}.parquet"), columns=['ts_code', 'close'])
            p_t1 = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_t1}.parquet"), columns=['ts_code', 'open', 'pre_close'])
        except:
            continue

        t0_row = p_t0[p_t0['ts_code'] == ts_code]
        t1_row = p_t1[p_t1['ts_code'] == ts_code]

        if t0_row.empty or t1_row.empty:
            continue

        t0_close = float(t0_row['close'].values[0])
        t1_open = float(t1_row['open'].values[0])
        t1_pre = float(t1_row['pre_close'].values[0]) if 'pre_close' in t1_row.columns else t0_close

        t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
        if t1_open_chg >= 9.5:
            limit_up += 1

        # 检查T+2跌停
        d_t2 = all_dates[curr_idx + 2]
        try:
            p_t2 = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_t2}.parquet"), columns=['ts_code', 'low', 'open'])
            t2_row = p_t2[p_t2['ts_code'] == ts_code]
            if not t2_row.empty:
                t2_low = float(t2_row['low'].values[0])
                t2_low_chg = (t2_low - t1_open) / t1_open * 100
                if t2_low_chg <= -9.5:
                    limit_down += 1
        except:
            pass

    print(f"  涨停开盘买入: {limit_up}笔")
    print(f"  跌停日卖出: {limit_down}笔")

    # 4. 收益统计
    print(f"  最高单笔收益: {trades['ret'].max()*100:.2f}%")
    print(f"  最低单笔收益: {trades['ret'].min()*100:.2f}%")
    print(f"  平均收益: {trades['ret'].mean()*100:.2f}%")

print("\n" + "=" * 100)
