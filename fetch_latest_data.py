import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infra_data.fetcher import DataFetcher
from datetime import datetime

import sys
from datetime import datetime, timedelta

if len(sys.argv) > 1:
    target_date = sys.argv[1]
else:
    today = datetime.now()
    # If today is Saturday (5) or Sunday (6), default to the most recent Friday
    if today.weekday() == 5:
        target_date = (today - timedelta(days=1)).strftime('%Y%m%d')
    elif today.weekday() == 6:
        target_date = (today - timedelta(days=2)).strftime('%Y%m%d')
    else:
        target_date = today.strftime('%Y%m%d')

print(f"=== 开始获取 {target_date} 数据 ===")

fetcher = DataFetcher()

# 获取热度数据
print(f"\n1. 获取同花顺热度数据...")
save_path_rank = os.path.join(fetcher.data_path, 'ths_rank1', f"{target_date}.parquet")
if os.path.exists(save_path_rank):
    print(f"   [SKIP] 同花顺热度数据已存在，跳过下载: {save_path_rank}")
elif datetime.now().hour < 12:
    print(f"   [INFO] 当前时间为早晨，跳过同花顺热度榜下载以避免频次限制，由预测脚本自动执行降级。")
else:
    try:
        df = fetcher._api_call('ths_hot', trade_date=target_date, market='热股', fields='ts_code,ts_name,hot,concept')
        if df is not None and len(df) > 0:
            df.to_parquet(save_path_rank)
            print(f"   [SUCCESS] 成功: {len(df)} 条")
        else:
            print(f"   [EMPTY] 无数据")
    except Exception as e:
        print(f"   [ERROR] 错误: {e}")

# 获取日线数据
print(f"\n2. 获取日线行情数据...")
save_path_price = os.path.join(fetcher.data_path, 'data_day1', f"{target_date}.parquet")
if os.path.exists(save_path_price):
    print(f"   [SKIP] 日线行情数据已存在，跳过下载: {save_path_price}")
else:
    try:
        df = fetcher._api_call('daily', trade_date=target_date)
        if df is not None and len(df) > 0:
            df.to_parquet(save_path_price)
            print(f"   [SUCCESS] 成功: {len(df)} 条")
        else:
            print(f"   [EMPTY] 无数据")
    except Exception as e:
        print(f"   [ERROR] 错误: {e}")

# 获取每日指标
print(f"\n3. 获取每日指标数据...")
save_path_other = os.path.join(fetcher.data_path, 'other_day1', f"{target_date}.parquet")
if os.path.exists(save_path_other):
    print(f"   [SKIP] 每日指标数据已存在，跳过下载: {save_path_other}")
else:
    try:
        df = fetcher._api_call('daily_basic', trade_date=target_date)
        if df is not None and len(df) > 0:
            df.to_parquet(save_path_other)
            print(f"   [SUCCESS] 成功: {len(df)} 条")
        else:
            print(f"   [EMPTY] 无数据")
    except Exception as e:
        print(f"   [ERROR] 错误: {e}")

# 获取筹码数据
print(f"\n4. 获取筹码分布数据...")
save_path_chip = os.path.join(fetcher.data_path, 'cyq1', f"{target_date}.parquet")
if os.path.exists(save_path_chip):
    print(f"   [SKIP] 筹码分布数据已存在，跳过下载: {save_path_chip}")
else:
    try:
        df = fetcher._api_call('cyq_perf', trade_date=target_date)
        if df is not None and len(df) > 0:
            df.to_parquet(save_path_chip)
            print(f"   [SUCCESS] 成功: {len(df)} 条")
        else:
            print(f"   [EMPTY] 无数据")
    except Exception as e:
        print(f"   [ERROR] 错误: {e}")

print(f"\n=== 完成 ===")
