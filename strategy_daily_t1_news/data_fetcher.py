import os
import sys
import pandas as pd
import tushare as ts
import time
from datetime import datetime
from tqdm import tqdm

# Ensure absolute imports for tushare token
sys.path.append(r'C:\Users\liuqi\iquant\quant_trading_system')
try:
    from config import TUSHARE_TOKEN
except ImportError:
    TUSHARE_TOKEN = "421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa" # Fallback from extraction script

pro = ts.pro_api(TUSHARE_TOKEN)
DATA_PATH = r"D:\iquant_data\data_v2"

def fetch_data_for_date(target_date):
    """
    Fetches required features (Price, Basic, Rank, Chip) for a specific date
    to satisfy the News Strategy requirements.
    """
    print(f"--- Fetching Tushare Data for {target_date} ---")
    
    # 1. Daily Price (data_day1)
    p_day1 = os.path.join(DATA_PATH, "data_day1", f"{target_date}.parquet")
    if not os.path.exists(p_day1):
        print(f"Fetching Daily Price...")
        try:
            df = pro.daily(trade_date=target_date)
            if not df.empty:
                df.to_parquet(p_day1)
        except Exception as e:
            print(f"Error fetching price: {e}")

    # 2. Daily Basic (other_day1 for circ_mv)
    p_other = os.path.join(DATA_PATH, "other_day1", f"{target_date}.parquet")
    if not os.path.exists(p_other):
        print(f"Fetching Daily Basic (Market Cap)...")
        try:
            df = pro.daily_basic(trade_date=target_date, fields='ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pb,circ_mv')
            if not df.empty:
                df.to_parquet(p_other)
        except Exception as e:
            print(f"Error fetching basic: {e}")

    # 3. THS Hot Rank (ths_rank1)
    p_rank = os.path.join(DATA_PATH, "ths_rank1", f"{target_date}.parquet")
    if not os.path.exists(p_rank):
        print(f"Fetching THS Hot Rank...")
        try:
            df = pro.ths_hot(trade_date=target_date, market='热股', fields='ts_code,ts_name,hot,concept')
            if not df.empty:
                df.to_parquet(p_rank)
        except Exception as e:
            print(f"Error fetching rank: {e}")

    # 4. Chip Distribution (cyq1)
    p_chip = os.path.join(DATA_PATH, "cyq1", f"{target_date}.parquet")
    if not os.path.exists(p_chip):
        print(f"Fetching Chip Distribution (CYQ)...")
        try:
            # We fetch for all stocks in batches or one by one as per existing extraction logic
            # For a single day, we can try pulling it directly if the API allows or use a loop
            # Simplified for production script:
            stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code')['ts_code'].tolist()
            dfs = []
            for i in range(0, len(stock_list), 100): # Small batches to avoid timeout
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
        except Exception as e:
            print(f"Error fetching chip: {e}")

    print(f"--- Data Sync for {target_date} Complete ---")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fetch_data_for_date(sys.argv[1])
    else:
        today = datetime.now().strftime("%Y%m%d")
        fetch_data_for_date(today)
