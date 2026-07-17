import pandas as pd
import numpy as np
import os

print("Test started")

# Test data loading
data_dir = r'D:\iquant_data\data_v2\data_day1'
files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
print(f"Found {len(files)} data files")

# Load one file
if files:
    test_file = os.path.join(data_dir, files[0])
    df = pd.read_parquet(test_file)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    print(f"Sample ts_code: {df['ts_code'].iloc[0]}")

print("Test completed")
