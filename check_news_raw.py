import pandas as pd
import os

DATA_DIR = r'D:\iquant_data\data_v2'
RAW_NEWS_DIR = os.path.join(DATA_DIR, 'news_raw_data')

print("=== Checking news_raw_data format ===")

# Check latest file
files = [f for f in os.listdir(RAW_NEWS_DIR) if f.endswith('.parquet')]
files.sort(reverse=True)

if files:
    latest_file = files[0]
    print(f"\nLatest file: {latest_file}")
    
    df = pd.read_parquet(os.path.join(RAW_NEWS_DIR, latest_file))
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nShape: {df.shape}")
    print(f"\nSample rows (first 3):")
    print(df.head(3).to_string())
