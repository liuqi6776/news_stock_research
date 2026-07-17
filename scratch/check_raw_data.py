import os

raw_dir = r"D:\iquant_data\data_v2\data_day1"
if os.path.exists(raw_dir):
    files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.parquet')])
    print(f"Found {len(files)} raw files in {raw_dir}")
    if files:
        print(f"Min file: {files[0]}")
        print(f"Max file: {files[-1]}")
        print("\nRecent 10 raw files:")
        for f in files[-10:]:
            print(f"  {f}")
else:
    print(f"Directory {raw_dir} does not exist!")
