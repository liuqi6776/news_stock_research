"""
step1_download_data.py
从 Tushare 增量下载 50ETF & 300ETF 期权基础合约列表、标的日线价格和每日所有期权合约的持仓/交易行情。
采用按月分批下载机制优化下载速度和频次。
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import tushare as ts

TOKEN = '421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa'
pro = ts.pro_api(TOKEN)

# 目录配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
DAILY_DIR = os.path.join(DATA_DIR, 'daily')
os.makedirs(DAILY_DIR, exist_ok=True)

BASIC_FILE = os.path.join(DATA_DIR, 'opt_basic.parquet')
UNDERLYING_FILE = os.path.join(DATA_DIR, 'underlying_daily.parquet')

def download_basic():
    print(">>> Downloading options basic contract info from Tushare...", flush=True)
    try:
        df = pro.opt_basic(exchange='SSE')
        print(f"Downloaded {len(df)} basic option contract records.", flush=True)
        # 仅保留50ETF和300ETF期权
        df_filtered = df[df['symbol'].str.startswith('510050') | df['symbol'].str.startswith('510300')].copy()
        print(f"Filtered to {len(df_filtered)} contracts for 50ETF and 300ETF options.", flush=True)
        df_filtered.to_parquet(BASIC_FILE, index=False)
        print(f"Saved basic contract info to {BASIC_FILE}", flush=True)
        return df_filtered
    except Exception as e:
        print(f"[ERROR] Failed to download basic contract info: {e}")
        sys.exit(1)

def download_underlying():
    print(">>> Downloading ETF daily close prices from Tushare...", flush=True)
    try:
        # 下载 50ETF (510050.SH) 和 300ETF (510300.SH) 的日线价格
        df_50 = pro.fund_daily(ts_code='510050.SH', start_date='20220101', end_date='20260311')
        df_300 = pro.fund_daily(ts_code='510300.SH', start_date='20220101', end_date='20260311')
        
        df = pd.concat([df_50, df_300], ignore_index=True)
        df.to_parquet(UNDERLYING_FILE, index=False)
        print(f"Saved underlying daily close prices to {UNDERLYING_FILE} (Total {len(df)} rows).", flush=True)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to download underlying daily prices: {e}")
        sys.exit(1)

def download_daily_options_by_month(trade_dates):
    print(f">>> Syncing daily options data for {len(trade_dates)} trading days (Optimized by month)...", flush=True)
    
    # 将交易日按月份分组
    trade_dates_df = pd.DataFrame({'date': sorted(trade_dates)})
    trade_dates_df['month'] = trade_dates_df['date'].str[:6]
    months_group = trade_dates_df.groupby('month')['date'].apply(list).to_dict()
    
    success_months = 0
    skip_months = 0
    
    for month, dates_in_month in sorted(months_group.items()):
        # 检查该月所有的交易日文件是否都已存在
        all_exist = True
        for d in dates_in_month:
            out_file = os.path.join(DAILY_DIR, f"opt_daily_{d}.parquet")
            if not os.path.exists(out_file):
                all_exist = False
                break
        
        if all_exist:
            skip_months += 1
            continue
            
        start_date = min(dates_in_month)
        end_date = max(dates_in_month)
        print(f"  Downloading options data for month {month} (Range: {start_date} to {end_date})...", flush=True)
        
        retries = 3
        df = None
        while retries > 0:
            try:
                df = pro.opt_daily(exchange='SSE', start_date=start_date, end_date=end_date)
                break
            except Exception as e:
                retries -= 1
                if "每分钟内请求次数" in str(e) or "接口限流" in str(e) or "limit" in str(e).lower():
                    print(f"    [LIMIT] Rate limited on month {month}, retrying in 5 seconds... (Retries left: {retries})", flush=True)
                    time.sleep(5)
                else:
                    print(f"    [ERROR] Error downloading month {month}: {e}, retrying in 2 seconds... (Retries left: {retries})", flush=True)
                    time.sleep(2)
        
        if df is not None and not df.empty:
            # 过滤只保留 50ETF 和 300ETF 相关的合约数据，防止保存不必要的数据
            # （注意：opt_daily 里只有 ts_code，我们要知道 ts_code 和 underlying_code 的对应关系）
            # 我们可以直接按 ts_code 过滤，但这需要在 main 里载入 basic_info。
            # 为了保持通用和稳妥，我们直接按 trade_date 拆分并存储，待后续 step2 做过滤。
            for date_val, group in df.groupby('trade_date'):
                out_file = os.path.join(DAILY_DIR, f"opt_daily_{date_val}.parquet")
                group.to_parquet(out_file, index=False)
            
            # 对于下载的数据里可能缺席的交易日，我们补一个空parquet以避免下次重复下载
            for d in dates_in_month:
                out_file = os.path.join(DAILY_DIR, f"opt_daily_{d}.parquet")
                if not os.path.exists(out_file):
                    pd.DataFrame().to_parquet(out_file)
            
            success_months += 1
            print(f"    [SUCCESS] Processed month {month}. Total rows: {len(df)}", flush=True)
        else:
            # 如果整月下载失败或为空，尝试退化为按天下载该月的日期
            print(f"    [WARNING] Month {month} download returned empty/failed. Falling back to daily download for this month...", flush=True)
            for d in dates_in_month:
                out_file = os.path.join(DAILY_DIR, f"opt_daily_{d}.parquet")
                if os.path.exists(out_file):
                    continue
                try:
                    df_day = pro.opt_daily(trade_date=d, exchange='SSE')
                    if df_day is not None and not df_day.empty:
                        df_day.to_parquet(out_file, index=False)
                    else:
                        pd.DataFrame().to_parquet(out_file)
                    time.sleep(0.12)
                except Exception as ex:
                    print(f"      [ERROR] Failed daily fallback for {d}: {ex}")
            success_months += 1
            
        # 适当休眠，防风控限流
        time.sleep(0.5)
        
    print(f"[FINISHED] Options Daily Sync Complete! Success Months: {success_months}, Skipped Months: {skip_months}.", flush=True)

if __name__ == '__main__':
    # 1. 基础合约信息
    if not os.path.exists(BASIC_FILE):
        df_basic = download_basic()
    else:
        df_basic = pd.read_parquet(BASIC_FILE)
        print(f"Loaded existing basic contracts info: {len(df_basic)} records.", flush=True)
        
    # 2. 标的日价格
    if not os.path.exists(UNDERLYING_FILE):
        df_und = download_underlying()
    else:
        df_und = pd.read_parquet(UNDERLYING_FILE)
        print(f"Loaded existing underlying daily close prices: {len(df_und)} rows.", flush=True)
        
    # 3. 提取所有的交易日期列表并排序
    trade_dates = sorted(df_und['trade_date'].unique())
    
    # 4. 下载每日的期权持仓/行情数据（按月合并下载）
    download_daily_options_by_month(trade_dates)
