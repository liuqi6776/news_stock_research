# -*- coding: utf-8 -*-
"""检查 income1 利润表 schema 与覆盖范围"""
import os, glob
import pandas as pd

files = sorted(glob.glob(r"D:/iquant_data/data_v2/income1/*.parquet"))
print(f"income1 文件数: {len(files)}")
first = files[0]
df = pd.read_parquet(first)
print(f"\n{os.path.basename(first)}: shape={df.shape}")
print("columns:", list(df.columns))
print(df.head(2).to_string())

# 按文件名确认时间范围
dates = sorted(int(os.path.basename(f)[:8]) for f in files)
print(f"\n日期范围: {dates[0]} ~ {dates[-1]}, 共 {len(dates)} 天")
