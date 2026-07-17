import os
import shutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

parquet_path = r'D:\iquant_data\data_v2\tushare_concept_map_cached.parquet'
backup_path = r'D:\iquant_data\data_v2\tushare_concept_map_cached_BACKUP.parquet'

try:
    print("--- Step 1: Saving current 'With Sector' equity curve ---")
    shutil.copy('dragon_daily_news_equity.csv', 'dragon_daily_news_equity_with_sector.csv')

    print("--- Step 2: Hiding Tushare concept cache to simulate 'Stock Only' mode ---")
    if os.path.exists(parquet_path):
        shutil.move(parquet_path, backup_path)

    print("--- Step 3: Re-training model for 'Stock Only' mode ---")
    subprocess.run(['python', 'train_daily_dragon_with_news.py'], check=True)

    print("--- Step 4: Running backtest for 'Stock Only' mode ---")
    subprocess.run(['python', 'run_dragon_daily_strict_with_news.py'], check=True)
    shutil.copy('dragon_daily_news_equity.csv', 'dragon_daily_news_equity_stock_only.csv')

finally:
    print("--- Step 5: Restoring Tushare concept cache ---")
    if os.path.exists(backup_path):
        shutil.move(backup_path, parquet_path)

print("--- Step 6: Generating comparison plot ---")
df_with_sector = pd.read_csv('dragon_daily_news_equity_with_sector.csv')
df_stock_only = pd.read_csv('dragon_daily_news_equity_stock_only.csv')

df_with_sector['date'] = pd.to_datetime(df_with_sector['date'])
df_stock_only['date'] = pd.to_datetime(df_stock_only['date'])

plt.figure(figsize=(12, 7))
plt.plot(df_stock_only['date'], df_stock_only['nav'], label='Stock News Only (No Sector Logic) - ~4279%', color='red', linewidth=2)
plt.plot(df_with_sector['date'], df_with_sector['nav'], label='With Sector News (Follower Trap) - ~2424%', color='blue', linewidth=2)

plt.title('Daily T+1 Strict Strategy: Stock-Only vs Sector-Included')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (Base 100k)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('dragon_daily_sector_comparison.png')
print("Saved comparison plot to dragon_daily_sector_comparison.png")
