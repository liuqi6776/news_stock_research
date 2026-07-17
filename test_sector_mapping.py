import pandas as pd
from processing.news_processor import load_and_process_news
import os

news_dir = r'D:\iquant_data\data_v2\news_major'
market_df, ss_df = load_and_process_news(news_dir, start_date='20220101', end_date='20261231')

print("Non-empty combinations fetched.")
print("Has market info:", not market_df.empty)
print("Has stock & sector info:", not ss_df.empty)

if not ss_df.empty:
    print("Columns:", ss_df.columns)
    if 'news_sector_impact' in ss_df.columns:
        valid_sectors = ss_df[ss_df['news_sector_impact'] > 0]
        print(f"Total rows with sector_impact > 0: {len(valid_sectors)}")
        if not valid_sectors.empty:
            print(valid_sectors.head())
    else:
        print("Column 'news_sector_impact' is missing!")
