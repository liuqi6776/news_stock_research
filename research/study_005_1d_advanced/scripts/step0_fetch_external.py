import os
import pandas as pd
from dotenv import load_dotenv
import tushare as ts
import akshare as ak

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(STUDY_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_industry_map():
    print("Fetching Tushare stock_basic for industry mapping...")
    load_dotenv()
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        print("Warning: TUSHARE_TOKEN not found in .env. Industry fetching might fail.")
    else:
        ts.set_token(token)
    
    pro = ts.pro_api()
    try:
        # Fetch listed stocks
        df_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry')
        # Fetch delisted stocks to cover historical data
        df_delist = pro.stock_basic(exchange='', list_status='D', fields='ts_code,symbol,name,industry')
        
        df = pd.concat([df_list, df_delist], ignore_index=True)
        df = df.drop_duplicates(subset=['ts_code'])
        
        out_path = os.path.join(DATA_DIR, 'industry_map.csv')
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} industry records to {out_path}")
    except Exception as e:
        print(f"Error fetching industry map: {e}")

def fetch_index_data():
    print("Fetching CSI 1000 (sh000852) index data via akshare...")
    try:
        # CSI 1000
        df = ak.stock_zh_index_daily(symbol="sh000852")
        df['trade_date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # Calculate MA20
        df['close'] = df['close'].astype(float)
        df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
        
        out_path = os.path.join(DATA_DIR, 'index_regime.csv')
        df[['trade_date', 'close', 'ma20']].to_csv(out_path, index=False)
        print(f"Saved {len(df)} index records to {out_path}")
    except Exception as e:
        print(f"Error fetching index data: {e}")

if __name__ == "__main__":
    fetch_industry_map()
    fetch_index_data()
