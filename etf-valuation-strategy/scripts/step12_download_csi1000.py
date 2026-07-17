import os
import requests
import datetime
import pandas as pd
import akshare as ak
from akshare.stock_feature.stock_a_pe_and_pb import hash_code, get_cookie_csrf
import py_mini_racer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def download_csi1000_price():
    print("Downloading CSI 1000 daily price from Sina via Akshare...")
    df = ak.stock_zh_index_daily(symbol='sh000852')
    # df has columns: date, open, high, low, close, volume
    # We need to match Tushare format: ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount
    df['trade_date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    df['ts_code'] = '000852.SH'
    df['vol'] = df['volume']
    
    # Calculate pre_close, change, pct_chg
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['pre_close'] = df['close'].shift(1)
    df['change'] = df['close'] - df['pre_close']
    df['pct_chg'] = (df['change'] / df['pre_close']) * 100.0
    
    # Drop first row since it won't have pre_close
    df = df.dropna().reset_index(drop=True)
    
    # Reorder columns
    cols = ['ts_code', 'trade_date', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol']
    df_final = df[cols]
    
    price_path = os.path.join(DATA_DIR, 'zz1000_daily.csv')
    df_final.to_csv(price_path, index=False)
    print(f"Saved price data to {price_path}. Shape: {df_final.shape}")

def download_csi1000_valuation():
    print("Downloading CSI 1000 daily valuation from Lianghua Gu API...")
    js = py_mini_racer.MiniRacer()
    js.eval(hash_code)
    token = js.call('hex', datetime.datetime.now().date().isoformat()).lower()
    
    # Fetch PE
    print("Fetching PE...")
    r_pe = requests.get(
        'https://legulegu.com/api/stockdata/index-basic-pe',
        params={'token': token, 'indexCode': '000852.SH'},
        **get_cookie_csrf(url='https://legulegu.com/stockdata/sz50-ttm-lyr')
    )
    df_pe = pd.DataFrame(r_pe.json()['data'])
    df_pe['date'] = pd.to_datetime(df_pe['date'])
    
    # Fetch PB
    print("Fetching PB...")
    r_pb = requests.get(
        'https://legulegu.com/api/stockdata/index-basic-pb',
        params={'token': token, 'indexCode': '000852.SH'},
        **get_cookie_csrf(url='https://legulegu.com/stockdata/zz500-ttm-lyr')
    )
    df_pb = pd.DataFrame(r_pb.json()['data'])
    df_pb['date'] = pd.to_datetime(df_pb['date'])
    
    # Merge PE and PB
    print("Merging PE and PB...")
    df_merged = pd.merge(df_pe[['date', 'ttmPe']], df_pb[['date', 'pb']], on='date', how='inner')
    df_merged['trade_date'] = df_merged['date'].dt.strftime('%Y%m%d')
    df_merged['ts_code'] = '000852.SH'
    df_merged['pe_ttm'] = df_merged['ttmPe']
    
    # Sort and format columns
    df_merged = df_merged.sort_values('trade_date').reset_index(drop=True)
    df_final = df_merged[['ts_code', 'trade_date', 'pe_ttm', 'pb']]
    
    val_path = os.path.join(DATA_DIR, 'zz1000_valuation.csv')
    df_final.to_csv(val_path, index=False)
    print(f"Saved valuation data to {val_path}. Shape: {df_final.shape}")

def main():
    download_csi1000_price()
    download_csi1000_valuation()
    print("All CSI 1000 data downloaded successfully!")

if __name__ == '__main__':
    main()
