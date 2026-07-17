import pandas as pd
import os

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')

target_date = '20260414'

p_rank = os.path.join(RANK_DIR, f"{target_date}.parquet")
p_chip = os.path.join(CHIP_DIR, f"{target_date}.parquet")
p_price = os.path.join(PRICE_DIR, f"{target_date}.parquet")
p_other = os.path.join(OTHER_DIR, f"{target_date}.parquet")

print(f"检查数据文件 ({target_date})...")
print(f"  rank: {os.path.exists(p_rank)}, {p_rank}")
print(f"  chip: {os.path.exists(p_chip)}, {p_chip}")
print(f"  price: {os.path.exists(p_price)}, {p_price}")
print(f"  other: {os.path.exists(p_other)}, {p_other}")

if os.path.exists(p_rank):
    rank_df = pd.read_parquet(p_rank)
    print(f"\nrank_df shape: {rank_df.shape}")
    print(f"rank_df columns: {list(rank_df.columns)}")
    print(rank_df.head(10))

if os.path.exists(p_price):
    price_df = pd.read_parquet(p_price)
    print(f"\nprice_df shape: {price_df.shape}")
    print(f"price_df columns: {list(price_df.columns)}")
    print(price_df.head(10))

dates = ['20260413', '20260414']
for d in dates:
    p = os.path.join(RANK_DIR, f"{d}.parquet")
    if os.path.exists(p):
        df = pd.read_parquet(p)
        print(f"\n{d} - rank data: {len(df)} 只")
