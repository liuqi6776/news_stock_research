import os
import sys
import pandas as pd
import tushare as ts
import time
from datetime import datetime

TUSHARE_TOKEN = "421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa"
pro = ts.pro_api(TUSHARE_TOKEN)
DATA_PATH = r"D:\iquant_data\data_v2"
THS_DIR = os.path.join(DATA_PATH, "ths_rank1")

dates_to_fetch = ['20260415']

for target_date in dates_to_fetch:
    print(f"\n=== 下载 {target_date} 的同花顺热度数据 ===")
    
    p_rank = os.path.join(THS_DIR, f"{target_date}.parquet")
    
    try:
        print(f"  正在下载...")
        df = pro.ths_hot(trade_date=target_date, market='热股', fields='ts_code,ts_name,hot,concept')
        
        if df is not None and len(df) > 0:
            df.to_parquet(p_rank)
            print(f"  ✓ 成功: {len(df)} 条数据")
        else:
            print(f"  ✗ 无数据返回")
            
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    time.sleep(1)

print("\n=== 完成 ===")
