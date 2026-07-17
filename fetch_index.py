import akshare as ak
import pandas as pd
import os

def fetch_index_data():
    print("Fetching SSE Index (000001) data for 2023...")
    try:
        df = ak.stock_zh_index_daily(symbol="sh000001")
        df['date'] = pd.to_datetime(df['date'])
        df_2023 = df[(df['date'] >= '2022-12-01') & (df['date'] <= '2024-01-31')]
        
        # Save to local project dir for easy access
        df_2023.to_csv('sse_index_2023.csv', index=False)
        print("Saved sse_index_2023.csv")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_index_data()
