# -*- coding: utf-8 -*-
"""确认 index_weight 中证1000成分 + industry_map 覆盖"""
import os, glob
import pandas as pd

# 1. index_weight: 有哪些指数?
files = sorted(glob.glob(r"D:/iquant_data/data_v2/index_weight/*.parquet"))
print(f"index_weight 文件数: {len(files)}")
df = pd.read_parquet(files[0])
print(f"\n{os.path.basename(files[0])}: columns={list(df.columns)}")
if "index_code" in df.columns:
    print("指数:", df["index_code"].unique()[:20])
elif "ts_code" in df.columns:
    print("ts_code 样例:", df["ts_code"].head(5).tolist())
    print("n:", len(df))
dates = sorted(int(os.path.basename(f).split("_")[-1][:8]) for f in files)
print(f"日期范围: {dates[0]} ~ {dates[-1]}, {len(dates)} 个快照")

# 2. industry_map 覆盖
im = pd.read_parquet(r"c:\Users\liuqi\quant_system_v2\research\studies\study_008_enhancements\data\industry_map.parquet")
print(f"\nindustry_map: shape={im.shape}, columns={list(im.columns)}")
print("n unique ts_code:", im["ts_code"].nunique())
print("行业数:", im["industry"].nunique())
print("行业列表:", sorted(im["industry"].unique())[:30])
