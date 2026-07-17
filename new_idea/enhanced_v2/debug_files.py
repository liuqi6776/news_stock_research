import os
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
feature_cache_dir = os.path.join(THIS_DIR, 'feature_cache')

feat_files = sorted([f for f in os.listdir(feature_cache_dir) if f.endswith('.parquet')])
print(f"Total feature cache files: {len(feat_files)}")

for f in feat_files[:5]:
    df = pd.read_parquet(os.path.join(feature_cache_dir, f))
    print(f"{f}: shape={df.shape}, cols={len(df.columns)}")

# 检查靠后的文件
for f in feat_files[-5:]:
    df = pd.read_parquet(os.path.join(feature_cache_dir, f))
    print(f"{f}: shape={df.shape}, cols={len(df.columns)}")
