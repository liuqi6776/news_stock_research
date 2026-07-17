import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
feature_cache_dir = os.path.join(THIS_DIR, 'feature_cache')

feat_files = sorted([f for f in os.listdir(feature_cache_dir) if f.endswith('.parquet')])
print(f"Total files: {len(feat_files)}")
print(f"First 5: {feat_files[:5]}")
print(f"Last 5: {feat_files[-5:]}")

# 提取日期
all_dates = [f.replace('.parquet', '').replace('feat_', '') for f in feat_files]
print(f"\nFirst 5 dates: {all_dates[:5]}")
print(f"Last 5 dates: {all_dates[-5:]}")

# 检查start_date
start_date = '20230801'
if start_date in all_dates:
    idx = all_dates.index(start_date)
    print(f"\nstart_date {start_date} found at index {idx}")
else:
    print(f"\nstart_date {start_date} NOT FOUND!")
    # 找最近的
    for d in all_dates:
        if d >= start_date:
            print(f"First date >= start_date: {d} at index {all_dates.index(d)}")
            break
