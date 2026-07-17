import os
import sys
import pandas as pd
import tushare as ts
import time
from datetime import datetime
from tqdm import tqdm

TUSHARE_TOKEN = "421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa"
pro = ts.pro_api(TUSHARE_TOKEN)
DATA_PATH = r"D:\iquant_data\data_v2"

def fetch_data_for_date(target_date):
    """
    下载指定日期的数据用于策略：
    - 价格数据 (data_day1)
    - 市值数据 (other_day1)
    - 同花顺热度 (ths_rank1)
    - 筹码分布 (cyq1)
    """
    print(f"--- [Step 2] 下载数据: {target_date} ---")
    
    os.makedirs(os.path.join(DATA_PATH, "data_day1"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "other_day1"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "ths_rank1"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "cyq1"), exist_ok=True)
    
    p_day1 = os.path.join(DATA_PATH, "data_day1", f"{target_date}.parquet")
    if not os.path.exists(p_day1):
        print(f"  下载每日行情数据...")
        try:
            df = pro.daily(trade_date=target_date)
            if not df.empty:
                df.to_parquet(p_day1)
                print(f"  ✓ 保存成功: {p_day1}")
        except Exception as e:
            print(f"  ✗ 下载行情失败: {e}")

    p_other = os.path.join(DATA_PATH, "other_day1", f"{target_date}.parquet")
    if not os.path.exists(p_other):
        print(f"  下载市值和基本面数据...")
        try:
            df = pro.daily_basic(trade_date=target_date, fields='ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pb,circ_mv')
            if not df.empty:
                df.to_parquet(p_other)
                print(f"  ✓ 保存成功: {p_other}")
        except Exception as e:
            print(f"  ✗ 下载基本面失败: {e}")

    p_rank = os.path.join(DATA_PATH, "ths_rank1", f"{target_date}.parquet")
    if not os.path.exists(p_rank):
        print(f"  下载同花顺热度排名...")
        try:
            df = pro.ths_hot(trade_date=target_date, market='热股', fields='ts_code,ts_name,hot,concept')
            if not df.empty:
                df.to_parquet(p_rank)
                print(f"  ✓ 保存成功: {p_rank}")
        except Exception as e:
            print(f"  ✗ 下载热度失败: {e}")

    p_chip = os.path.join(DATA_PATH, "cyq1", f"{target_date}.parquet")
    if not os.path.exists(p_chip):
        print(f"  下载筹码分布数据...")
        try:
            stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code')['ts_code'].tolist()
            dfs = []
            for i in range(0, len(stock_list), 100):
                batch = stock_list[i:i+100]
                try:
                    cdf = pro.cyq_perf(ts_code=','.join(batch), start_date=target_date, end_date=target_date)
                    if not cdf.empty:
                        dfs.append(cdf)
                except:
                    time.sleep(1)
            if dfs:
                final_chip = pd.concat(dfs)
                final_chip.to_parquet(p_chip)
                print(f"  ✓ 保存成功: {p_chip}")
        except Exception as e:
            print(f"  ✗ 下载筹码失败: {e}")

    print(f"--- [Step 2] 数据下载完成 ---")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fetch_data_for_date(sys.argv[1])
    else:
        today = datetime.now().strftime("%Y%m%d")
        fetch_data_for_date(today)
