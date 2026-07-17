
import tushare as ts
from datetime import datetime
import os
import sys

# Append project path to import settings
sys.path.append(r'C:\Users\liuqi\quant_system_v2')
from config.settings import settings

def check_status():
    print("Checking market status...")
    try:
        token = settings.TUSHARE_TOKEN
        if not token:
            print("No token found.")
            return

        pro = ts.pro_api(token)
        
        # Get index daily
        df = pro.index_daily(ts_code='000001.SH', end_date=datetime.now().strftime('%Y%m%d'), limit=100)
        
        if df.empty:
            print("DF empty")
            return

        df = df.sort_values('trade_date')
        closes = df['close'].values
        
        ma20 = closes[-20:].mean()
        ma60 = closes[-60:].mean()
        
        status = "BEAR" if ma20 < ma60 else "BULL"
        
        print(f"Status: {status}")
        print(f"MA20: {ma20:.2f}")
        print(f"MA60: {ma60:.2f}")
        print(f"CLOSE: {closes[-1]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_status()
