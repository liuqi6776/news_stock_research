import pandas as pd
import numpy as np
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'

trades = pd.read_csv(os.path.join(THIS_DIR, 'trades_Enhanced_All.csv'))

print("=" * 80)
print("涨跌停限制详细检查 (简化版)")
print("=" * 80)

# 获取所有日期
all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
date_idx = {d: i for i, d in enumerate(all_dates)}

limit_up_buys = []
limit_down_sells = []
all_records = []

# 批量处理，减少打印
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
        p_t1 = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_t1}.parquet"), columns=['ts_code', 'open', 'low', 'pre_close'])
        p_t2 = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_t2}.parquet"), columns=['ts_code', 'low', 'pre_close'])
    except Exception as e:
        continue

    t0_row = p_t0[p_t0['ts_code'] == ts_code]
    t1_row = p_t1[p_t1['ts_code'] == ts_code]
    t2_row = p_t2[p_t2['ts_code'] == ts_code]

    if t0_row.empty or t1_row.empty or t2_row.empty:
        continue

    t0_close = float(t0_row['close'].values[0])
    t1_open = float(t1_row['open'].values[0])
    t1_low = float(t1_row['low'].values[0])
    t2_low = float(t2_row['low'].values[0])

    # 使用pre_close计算涨跌幅更准确
    t1_pre = float(t1_row['pre_close'].values[0]) if 'pre_close' in t1_row.columns else t0_close
    t2_pre = float(t2_row['pre_close'].values[0]) if 'pre_close' in t2_row.columns else t0_close

    # T+1开盘相对前收盘涨幅
    t1_open_chg = (t1_open - t1_pre) / t1_pre * 100

    # T+2最低相对T+1开盘跌幅
    t2_low_chg = (t2_low - t1_open) / t1_open * 100

    # 判断板块
    is_cyb = ts_code.startswith('300') or ts_code.startswith('301')
    limit_pct = 20.0 if is_cyb else 10.0

    rec = {
        'date': d_str, 'ts_code': ts_code,
        't0_close': t0_close, 't1_open': t1_open, 't1_pre': t1_pre,
        't1_open_chg': t1_open_chg, 't2_low': t2_low, 't2_low_chg': t2_low_chg,
        'ret': row['ret'], 'is_cyb': is_cyb, 'limit_pct': limit_pct
    }
    all_records.append(rec)

    # 开盘涨幅 >= limit_pct - 0.5 视为接近涨停开盘
    if t1_open_chg >= (limit_pct - 0.5):
        limit_up_buys.append(rec)

    # T+2最低相对T+1开盘跌幅 <= -(limit_pct - 0.5)
    if t2_low_chg <= -(limit_pct - 0.5):
        limit_down_sells.append(rec)

    if (idx + 1) % 50 == 0:
        print(f"  已处理 {idx+1}/{len(trades)} 笔交易...")

print(f"\n总检查交易数: {len(all_records)}")

print(f"\n【1】开盘涨停买入检查 (T+1开盘相对前收盘涨幅 >= {10-0.5}%)")
print(f"  涉及涨停开盘买入的交易数: {len(limit_up_buys)}")
if len(limit_up_buys) > 0:
    print(f"  ⚠️ 发现 {len(limit_up_buys)} 笔交易在接近涨停价开盘时买入!")
    for r in limit_up_buys[:15]:
        print(f"    {r['date']} {r['ts_code']}: T+1开盘={r['t1_open']:.2f}, 前收={r['t1_pre']:.2f}, 涨幅={r['t1_open_chg']:.2f}%, limit={r['limit_pct']}%")
else:
    print(f"  ✅ 无开盘涨停买入情况")

print(f"\n【2】跌停日卖出检查")
print(f"  涉及跌停日卖出的交易数: {len(limit_down_sells)}")
if len(limit_down_sells) > 0:
    print(f"  ⚠️ 发现 {len(limit_down_sells)} 笔交易在接近跌停日!")
    for r in limit_down_sells[:15]:
        print(f"    {r['date']} {r['ts_code']}: T+1开盘={r['t1_open']:.2f}, T+2最低={r['t2_low']:.2f}, 跌幅={r['t2_low_chg']:.2f}%")
else:
    print(f"  ✅ 无跌停日情况")

# 统计分布
df = pd.DataFrame(all_records)
print(f"\n【3】T+1开盘涨幅分布")
print(f"  均值: {df['t1_open_chg'].mean():.2f}%")
print(f"  中位数: {df['t1_open_chg'].median():.2f}%")
print(f"  最大值: {df['t1_open_chg'].max():.2f}%")
print(f"  最小值: {df['t1_open_chg'].min():.2f}%")
print(f"  > 5%: {len(df[df['t1_open_chg'] > 5])} 笔")
print(f"  > 9%: {len(df[df['t1_open_chg'] > 9])} 笔")

print(f"\n【4】T+2最低相对T+1开盘跌幅分布")
print(f"  均值: {df['t2_low_chg'].mean():.2f}%")
print(f"  中位数: {df['t2_low_chg'].median():.2f}%")
print(f"  最大值: {df['t2_low_chg'].max():.2f}%")
print(f"  最小值: {df['t2_low_chg'].min():.2f}%")
print(f"  < -5%: {len(df[df['t2_low_chg'] < -5])} 笔")
print(f"  < -9%: {len(df[df['t2_low_chg'] < -9])} 笔")

# 保存结果
df.to_csv(os.path.join(THIS_DIR, 'limit_check_detail.csv'), index=False)
print(f"\n结果已保存至 limit_check_detail.csv")
