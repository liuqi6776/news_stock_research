import os
import tushare as ts
import pandas as pd
import time

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

TOKEN = "421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa"
pro = ts.pro_api(TOKEN)

def fetch_valuation_chunked(ts_code):
    print(f"Downloading valuation for {ts_code} in chunks...")
    df1 = pro.index_dailybasic(ts_code=ts_code, start_date="20100101", end_date="20171231")
    time.sleep(1)
    df2 = pro.index_dailybasic(ts_code=ts_code, start_date="20180101", end_date="20260315")
    
    if df1 is not None and df2 is not None:
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.drop_duplicates(subset=['trade_date'])
        df = df.sort_values(by='trade_date', ascending=False).reset_index(drop=True)
        return df
    elif df1 is not None:
        return df1
    elif df2 is not None:
        return df2
    else:
        raise ValueError(f"Failed to download valuation for {ts_code}")

def main():
    start_date = "20100101"
    end_date = "20260315"
    
    # 1. ChiNext Index (399006.SZ) Price & Valuation
    print("1. Downloading ChiNext (399006.SZ) price...")
    df_chinext_price = pro.index_daily(ts_code="399006.SZ", start_date=start_date, end_date=end_date)
    print(f"ChiNext Price shape: {df_chinext_price.shape}")
    df_chinext_price.to_csv(os.path.join(DATA_DIR, 'chinext_daily.csv'), index=False)
    time.sleep(1)
    
    print("2. Downloading ChiNext (399006.SZ) valuation...")
    df_chinext_val = fetch_valuation_chunked("399006.SZ")
    print(f"ChiNext Valuation shape: {df_chinext_val.shape}")
    df_chinext_val.to_csv(os.path.join(DATA_DIR, 'chinext_valuation.csv'), index=False)
    time.sleep(1)
    
    # 2. CSI Dividend Low Vol Index (H30269.CSI) Price
    print("3. Downloading Dividend Low Vol (H30269.CSI) price...")
    df_div_price = pro.index_daily(ts_code="H30269.CSI", start_date=start_date, end_date=end_date)
    print(f"Dividend Low Vol Price shape: {df_div_price.shape}")
    df_div_price.to_csv(os.path.join(DATA_DIR, 'div_low_vol_daily.csv'), index=False)
    time.sleep(1)
    
    # 3. SSE 50 Index (000016.SH) Valuation (Proxy for Dividend Low Vol)
    print("4. Downloading SSE 50 (000016.SH) valuation...")
    df_sse50_val = fetch_valuation_chunked("000016.SH")
    print(f"SSE 50 Valuation shape: {df_sse50_val.shape}")
    df_sse50_val.to_csv(os.path.join(DATA_DIR, 'sse50_valuation.csv'), index=False)
    time.sleep(1)
    
    # 4. Gold ETF (518880.SH) Price
    print("5. Downloading Gold ETF (518880.SH) price...")
    df_gold_price = pro.fund_daily(ts_code="518880.SH", start_date=start_date, end_date=end_date)
    print(f"Gold ETF Price shape: {df_gold_price.shape}")
    df_gold_price.to_csv(os.path.join(DATA_DIR, 'gold_etf_daily.csv'), index=False)
    time.sleep(1)
    
    # 5. Nasdaq ETF (513100.SH) Price
    print("6. Downloading Nasdaq ETF (513100.SH) price...")
    df_nasdaq_price = pro.fund_daily(ts_code="513100.SH", start_date=start_date, end_date=end_date)
    print(f"Nasdaq ETF Price shape: {df_nasdaq_price.shape}")
    df_nasdaq_price.to_csv(os.path.join(DATA_DIR, 'nasdaq_etf_daily.csv'), index=False)
    
    print("\nAll extended data downloaded and saved successfully.")

if __name__ == "__main__":
    main()
