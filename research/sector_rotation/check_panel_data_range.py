# -*- coding: utf-8 -*-
"""快速确认 data_day1 时间范围 (按文件名) + 财务数据覆盖 (B0, pandas版)"""
import os, glob
import pandas as pd

# 1. 行情文件名 = 日期
all_files = sorted(glob.glob(r"D:/iquant_data/data_v2/data_day1/*.parquet"))
dates = sorted(int(os.path.basename(f)[:8]) for f in all_files if os.path.getsize(f) > 1024)
print(f"有效文件: {len(dates)} 个, 日期范围: {dates[0]} ~ {dates[-1]}")

# 按月统计覆盖
import collections
ym_counts = collections.Counter(d // 100 for d in dates)
print("\n按年交易日数:")
for y in range(2015, 2027):
    n = sum(v for k, v in ym_counts.items() if k // 100 == y)
    if n:
        print(f"  {y}: {n} 个交易日")

# 2019-10 之后月份覆盖
ym_2019 = sorted(k for k in ym_counts if k >= 201910)
print(f"\n2019-10 起月数: {len(ym_2019)} ({ym_2019[0]} ~ {ym_2019[-1]})")

# 2. 财务 PIT
fin = pd.read_parquet(r"D:/iquant_data/data_v2/fundamental1/fina_indicator_cache.parquet")
print(f"\n财务: shape={fin.shape}, columns={list(fin.columns)[:12]}")
if "ann_date" in fin.columns:
    ann = pd.to_datetime(fin["ann_date"], errors="coerce").dropna()
    ann_int = ann.astype("int64").astype(str).str[:8]
    print(f"ann_date 范围: {ann_int.min()} ~ {ann_int.max()}, 非空 {len(ann)}/{len(fin)}")
    y = ann_int.str[:4]
    print("\n财务公告按年:")
    for yy in range(2015, 2027):
        n = (y == str(yy)).sum()
        if n:
            print(f"  {yy}: {n:,}")
