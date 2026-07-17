import os
import sys
import tushare as ts
import pandas as pd
import time

# Add root folder to python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
os.makedirs(os.path.join(PROJECT_DIR, 'data'), exist_ok=True)

TOKEN = "421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa"
pro = ts.pro_api(TOKEN)

def fetch_valuation_chunked(ts_code):
    print(f"Downloading valuation for {ts_code} in chunks to bypass 3000-row limit...")
    df1 = pro.index_dailybasic(ts_code=ts_code, start_date="20100101", end_date="20171231")
    print(f"Chunk 1 (2010-2017): {df1.shape if df1 is not None else 'None'}")
    time.sleep(1)
    df2 = pro.index_dailybasic(ts_code=ts_code, start_date="20180101", end_date="20260315")
    print(f"Chunk 2 (2018-2026): {df2.shape if df2 is not None else 'None'}")
    
    if df1 is not None and df2 is not None:
        df = pd.concat([df1, df2], ignore_index=True)
        # Drop duplicates based on trade_date just in case
        df = df.drop_duplicates(subset=['trade_date'])
        # Sort by trade_date descending to match Tushare style, but we'll handle sorting in backtest
        df = df.sort_values(by='trade_date', ascending=False).reset_index(drop=True)
        return df
    elif df1 is not None:
        return df1
    elif df2 is not None:
        return df2
    else:
        raise ValueError(f"Failed to download valuation for {ts_code}")

def download_data():
    start_date = "20100101"
    end_date = "20260315"
    
    print(f"1. Downloading HS300 price (000300.SH) from {start_date} to {end_date}...")
    df_300 = pro.index_daily(ts_code="000300.SH", start_date=start_date, end_date=end_date)
    print(f"HS300 Price shape: {df_300.shape}")
    
    time.sleep(1)
    
    print(f"2. Downloading HS300 daily valuation (000300.SH)...")
    df_val_300 = fetch_valuation_chunked("000300.SH")
    print(f"HS300 Combined Valuation shape: {df_val_300.shape}")
    
    time.sleep(1)
    
    print(f"3. Downloading ZZ500 price (000905.SH) from {start_date} to {end_date}...")
    df_500 = pro.index_daily(ts_code="000905.SH", start_date=start_date, end_date=end_date)
    print(f"ZZ500 Price shape: {df_500.shape}")
    
    time.sleep(1)
    
    print(f"4. Downloading ZZ500 daily valuation (000905.SH)...")
    df_val_500 = fetch_valuation_chunked("000905.SH")
    print(f"ZZ500 Combined Valuation shape: {df_val_500.shape}")
    
    time.sleep(1)
    
    print(f"5. Downloading Treasury Bond ETF (511010.SH) from {start_date} to {end_date}...")
    df_bond = pro.fund_daily(ts_code="511010.SH", start_date=start_date, end_date=end_date)
    print(f"Bond ETF shape: {df_bond.shape}")
    
    # Save raw data to files
    out_dir = os.path.join(PROJECT_DIR, 'data')
    df_300.to_csv(os.path.join(out_dir, 'hs300_daily.csv'), index=False)
    df_val_300.to_csv(os.path.join(out_dir, 'hs300_valuation.csv'), index=False)
    df_500.to_csv(os.path.join(out_dir, 'zz500_daily.csv'), index=False)
    df_val_500.to_csv(os.path.join(out_dir, 'zz500_valuation.csv'), index=False)
    df_bond.to_csv(os.path.join(out_dir, 'bond_etf_daily.csv'), index=False)
    print("All raw data downloaded and saved successfully.")

if __name__ == "__main__":
    download_data()
