import pandas as pd
import numpy as np
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'

trades = pd.read_csv(os.path.join(THIS_DIR, 'trades_Enhanced_All.csv'))

print("=" * 80)
print("涨跌停限制详细检查")
print("=" * 80)

# 需要检查:
# 1. T+1日开盘价相对T日收盘价的涨幅 > 9.5% -> 不能买入
# 2. T+2日最低价相对T+1日开盘价的跌幅 < -9.5% -> 可能无法卖出（跌停）

limit_up_buys = []      # 开盘涨停买入（不应买入）
limit_down_sells = []   # 跌停日卖出（可能无法卖出）
all_limit_checks = []

for idx, row in trades.iterrows():
    d_str = str(int(row['date']))
    ts_code = row['ts_code']
    buy_open = row['buy_open']
    sell_close = row['sell_close']
    ret = row['ret']

    # 找到T日(预测日), T+1日(买入日), T+2日(卖出日)
    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    if d_str not in all_dates:
        continue
    curr_idx = all_dates.index(d_str)
    if curr_idx + 2 >= len(all_dates):
        continue

    d_t0 = all_dates[curr_idx]      # T日
    d_t1 = all_dates[curr_idx + 1]  # T+1日（买入日）
    d_t2 = all_dates[curr_idx + 2]  # T+2日（卖出日）

    try:
        p_t0 = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_t0}.parquet"), columns=['ts_code', 'close', 'open', 'high', 'low', 'pre_close'])
        p_t1 = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_t1}.parquet"), columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
        p_t2 = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_t2}.parquet"), columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
    except:
        continue

    t0_row = p_t0[p_t0['ts_code'] == ts_code]
    t1_row = p_t1[p_t1['ts_code'] == ts_code]
    t2_row = p_t2[p_t2['ts_code'] == ts_code]

    if t0_row.empty or t1_row.empty or t2_row.empty:
        continue

    t0_close = t0_row['close'].values[0]
    t1_open = t1_row['open'].values[0]
    t1_low = t1_row['low'].values[0]
    t1_high = t1_row['high'].values[0]
    t1_pre_close = t1_row.get('pre_close', t0_row['close']).values[0]
    t2_low = t2_row['low'].values[0]
    t2_open = t2_row['open'].values[0]
    t2_pre_close = t2_row.get('pre_close', t1_row['close']).values[0]

    # 检查T+1开盘相对T收盘的涨幅
    t1_open_chg = (t1_open - t0_close) / t0_close * 100

    # 检查T+1是否开盘涨停（主板10%，创业板20%）
    # 先判断是主板还是创业板
    is_cyb = ts_code.startswith('300') or ts_code.startswith('301')
    limit_pct = 20.0 if is_cyb else 10.0

    # T+1开盘涨幅
    t1_open_limit_up = t1_open_chg >= (limit_pct - 0.5)  # 接近涨停开盘

    # T+2最低价相对T+1开盘的跌幅
    t2_low_chg = (t2_low - t1_open) / t1_open * 100
    t2_limit_down = t2_low_chg <= -(limit_pct - 0.5)

    # T+2开盘相对T+1收盘的跌幅（影响能否在T+2开盘卖出）
    t2_open_chg_from_t1_close = (t2_open - t1_open) / t1_open * 100

    record = {
        'date': d_str,
        'ts_code': ts_code,
        't0_close': t0_close,
        't1_open': t1_open,
        't1_open_chg_pct': t1_open_chg,
        't1_low': t1_low,
        't2_low': t2_low,
        't2_low_chg_from_t1_open': t2_low_chg,
        't2_open': t2_open,
        't2_open_chg_from_t1_open': t2_open_chg_from_t1_close,
        'ret': ret,
        'is_limit_up_open': t1_open_limit_up,
        'is_limit_down_day': t2_limit_down,
        'limit_pct': limit_pct
    }
    all_limit_checks.append(record)

    if t1_open_limit_up:
        limit_up_buys.append(record)

    if t2_limit_down:
        limit_down_sells.append(record)

df_check = pd.DataFrame(all_limit_checks)

print(f"\n【1】开盘涨停买入检查 (T+1开盘相对T收盘涨幅 >= 9.5% / 19.5%)")
print(f"  涉及涨停开盘买入的交易数: {len(limit_up_buys)}")
if len(limit_up_buys) > 0:
    print(f"  ⚠️ 发现 {len(limit_up_buys)} 笔交易在接近涨停价开盘时买入!")
    for r in limit_up_buys[:10]:
        print(f"    {r['date']} {r['ts_code']}: T+1开盘={r['t1_open']:.2f}, 相对T收盘涨幅={r['t1_open_chg_pct']:.2f}%, limit={r['limit_pct']}%")
else:
    print(f"  ✅ 无开盘涨停买入情况")

print(f"\n【2】跌停日卖出检查 (T+2最低价相对T+1开盘跌幅 >= 9.5% / 19.5%)")
print(f"  涉及跌停日卖出的交易数: {len(limit_down_sells)}")
if len(limit_down_sells) > 0:
    print(f"  ⚠️ 发现 {len(limit_down_sells)} 笔交易在接近跌停日卖出!")
    for r in limit_down_sells[:10]:
        print(f"    {r['date']} {r['ts_code']}: T+1开盘={r['t1_open']:.2f}, T+2最低={r['t2_low']:.2f}, 跌幅={r['t2_low_chg_from_t1_open']:.2f}%")
else:
    print(f"  ✅ 无跌停日卖出情况")

# 统计T+1开盘涨幅分布
print(f"\n【3】T+1开盘涨幅分布统计")
print(f"  T+1开盘涨幅均值: {df_check['t1_open_chg_pct'].mean():.2f}%")
print(f"  T+1开盘涨幅中位数: {df_check['t1_open_chg_pct'].median():.2f}%")
print(f"  T+1开盘涨幅最大值: {df_check['t1_open_chg_pct'].max():.2f}%")
print(f"  T+1开盘涨幅最小值: {df_check['t1_open_chg_pct'].min():.2f}%")
print(f"  T+1开盘涨幅 > 5% 的交易数: {len(df_check[df_check['t1_open_chg_pct'] > 5])}")
print(f"  T+1开盘涨幅 > 9% 的交易数: {len(df_check[df_check['t1_open_chg_pct'] > 9])}")

# 统计T+2相对T+1的跌幅分布
print(f"\n【4】T+2相对T+1开盘跌幅分布统计")
print(f"  T+2最低相对T+1开盘跌幅均值: {df_check['t2_low_chg_from_t1_open'].mean():.2f}%")
print(f"  T+2最低相对T+1开盘跌幅中位数: {df_check['t2_low_chg_from_t1_open'].median():.2f}%")
print(f"  T+2最低相对T+1开盘跌幅最大值: {df_check['t2_low_chg_from_t1_open'].max():.2f}%")
print(f"  T+2最低相对T+1开盘跌幅最小值: {df_check['t2_low_chg_from_t1_open'].min():.2f}%")
print(f"  T+2最低相对T+1开盘跌幅 < -5% 的交易数: {len(df_check[df_check['t2_low_chg_from_t1_open'] < -5])}")
print(f"  T+2最低相对T+1开盘跌幅 < -9% 的交易数: {len(df_check[df_check['t2_low_chg_from_t1_open'] < -9])}")

# 保存详细结果
df_check.to_csv(os.path.join(THIS_DIR, 'limit_up_down_check.csv'), index=False)
print(f"\n详细检查结果已保存至: limit_up_down_check.csv")
