# -*- coding: utf-8 -*-
"""检查 fundamental1 目录 + 正确解析 ann_date (B0 补充)"""
import os, glob
import pandas as pd

d = r"D:/iquant_data/data_v2/fundamental1"
for f in sorted(glob.glob(os.path.join(d, "*"))):
    sz = os.path.getsize(f)
    print(f"{os.path.basename(f):<45} {sz/1e6:8.1f} MB")

# 正确解析 ann_date
fin = pd.read_parquet(os.path.join(d, "fina_indicator_cache.parquet"))
print(f"\nfina_indicator_cache: shape={fin.shape}")
print(f"columns: {list(fin.columns)}")
print(f"ts_code nunique: {fin['ts_code'].nunique()}")
print(f"ann_date 样例: {fin['ann_date'].head(3).tolist()}, dtype={fin['ann_date'].dtype}")

ann = fin["ann_date"].astype(str).str.replace("-", "").str[:8]
print(f"ann_date 范围: {ann.min()} ~ {ann.max()}")
y = ann.str[:4]
print("\n公告按年:")
for yy in range(2018, 2027):
    n = (y == str(yy)).sum()
    if n:
        print(f"  {yy}: {n:,}")

# 每股多少条
print(f"\n每股公告数分布: min={fin.groupby('ts_code')['ann_date'].count().min()}, "
      f"median={fin.groupby('ts_code')['ann_date'].count().median():.0f}, "
      f"max={fin.groupby('ts_code')['ann_date'].count().max()}")

# roe 覆盖
for col in ["roe", "or_yoy", "netprofit_yoy"]:
    if col in fin.columns:
        print(f"{col}: 非空 {fin[col].notna().sum():,}/{len(fin)}")
