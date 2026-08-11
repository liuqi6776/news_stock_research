# -*- coding: utf-8 -*-
"""查看 data_day1 行情 parquet schema 与文件规模 (快速)"""
import os, glob
import pandas as pd

all_files = sorted(glob.glob(r"D:/iquant_data/data_v2/data_day1/*.parquet"))
print(f"文件数: {len(all_files)}")
for f in all_files[:2]:
    df = pd.read_parquet(f, columns=None)
    print(f"\n--- {os.path.basename(f)} ---")
    print("columns:", list(df.columns))
    print("shape:", df.shape)
    if "trade_date" in df.columns:
        print("trade_date dtype:", df["trade_date"].dtype, "range:", df["trade_date"].min(), "~", df["trade_date"].max())
    print("n unique ts_code:", df["ts_code"].nunique() if "ts_code" in df.columns else "N/A")

# 文件大小统计
sizes = [os.path.getsize(f) for f in all_files]
print(f"\n文件大小: min={min(sizes)/1e6:.1f}MB max={max(sizes)/1e6:.1f}MB total={sum(sizes)/1e9:.1f}GB")
