import tushare as ts
import pandas as pd

TOKEN = "421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa"
pro = ts.pro_api(TOKEN)

# Let's test 000832.SH or 000832.CSI
print("Testing 000832.SH/CSI index_daily...")
for code in ["000832.CSI", "000832.SH"]:
    try:
        df = pro.index_daily(ts_code=code, start_date="20150101", end_date="20260315")
        if df is not None and not df.empty:
            print(f"{code} downloaded successfully! Shape: {df.shape}")
            print(df.head())
            print(df.tail())
        else:
            print(f"{code} returned empty.")
    except Exception as e:
        print(f"Error downloading {code}: {e}")

# Let's test H30010.CSI or H30010.SH
print("\nTesting H30010 index_daily...")
for code in ["H30010.CSI", "H30010.SH"]:
    try:
        df = pro.index_daily(ts_code=code, start_date="20150101", end_date="20260315")
        if df is not None and not df.empty:
            print(f"{code} downloaded successfully! Shape: {df.shape}")
            print(df.head())
            print(df.tail())
        else:
            print(f"{code} returned empty.")
    except Exception as e:
        print(f"Error downloading {code}: {e}")
