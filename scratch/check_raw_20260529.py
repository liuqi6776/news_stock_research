import os

DATA_DIR = r'D:\iquant_data\data_v2'
target_date = "20260529"

paths = {
    "price": os.path.join(DATA_DIR, 'data_day1', f"{target_date}.parquet"),
    "rank": os.path.join(DATA_DIR, 'ths_rank1', f"{target_date}.parquet"),
    "chip": os.path.join(DATA_DIR, 'cyq1', f"{target_date}.parquet"),
    "other": os.path.join(DATA_DIR, 'other_day1', f"{target_date}.parquet"),
}

for name, path in paths.items():
    if os.path.exists(path):
        print(f"[SUCCESS] {name} raw data exists. Size: {os.path.getsize(path)} bytes")
    else:
        print(f"[ERROR] {name} raw data does NOT exist at {path}")
