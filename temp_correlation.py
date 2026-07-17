import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infra_data.storage import DataStorage

PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'
storage = DataStorage()

train_start='20230101'
train_end='20231231'

dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
train_dates = [d for d in dates if train_start <= d <= train_end]

valid_dates_series = pd.Series([pd.to_datetime(d) for d in train_dates]).sort_values()
news_market_df, news_stock_sector_df = storage.load_news_data(train_start, train_end, valid_dates_series)

if not news_market_df.empty:
    news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
if not news_stock_sector_df.empty:
    news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')

all_data = []
# process full year of 2023 
for i in range(len(train_dates)-1):
    d_curr, d_next = train_dates[i], train_dates[i+1]
    
    p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
    if not os.path.exists(p_price): continue
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close'])
    
    p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
    if not os.path.exists(p_next): continue
    next_df = pd.read_parquet(p_next, columns=['ts_code', 'pct_chg', 'open', 'high'])
    next_df['T1_open_to_high_ret'] = (next_df['high'] / (next_df['open'] + 1e-8)) - 1
    next_df = next_df.rename(columns={'pct_chg': 'pct_chg_next'})
    
    m = pd.merge(price_df, next_df[['ts_code', 'pct_chg_next', 'T1_open_to_high_ret']], on='ts_code')
    m['trade_date'] = d_next
    all_data.append(m)

df = pd.concat(all_data, ignore_index=True)
if not news_market_df.empty:
    df = pd.merge(df, news_market_df, on='trade_date', how='left')
else:
    df['news_market_impact'] = 0.0

if not news_stock_sector_df.empty:
    df = pd.merge(df, news_stock_sector_df, on=['trade_date', 'ts_code'], how='left')
else:
    df['news_stock_impact'] = 0.0
    df['news_sector_impact'] = 0.0

df[['news_market_impact', 'news_stock_impact', 'news_sector_impact']] = df[['news_market_impact', 'news_stock_impact', 'news_sector_impact']].fillna(0.0)

print("=== Correlation across Full Universe (including non-news days) ===")
print(df[['news_market_impact', 'news_stock_impact', 'news_sector_impact', 'pct_chg_next', 'T1_open_to_high_ret']].corr().to_markdown())

has_news = df[(df['news_stock_impact'] != 0) | (df['news_sector_impact'] != 0)]
if not has_news.empty:
    print("\n=== Correlation restricted ONLY to stocks with active News hitting that day ===")
    print(has_news[['news_stock_impact', 'news_sector_impact', 'pct_chg_next', 'T1_open_to_high_ret']].corr().to_markdown())
