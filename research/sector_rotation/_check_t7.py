# -*- coding: utf-8 -*-
"""检查T7 index格式"""
import pickle, pandas as pd, os
ROOT = r"c:\Users\liuqi\quant_system_v2"
with open(os.path.join(ROOT, "research/sector_rotation/results/stock_gbdt_s123_results.pkl"), "rb") as f:
    d = pickle.load(f)
t7_nav = d["t7"]["nav"]
print(f"type: {type(t7_nav)}")
print(f"len: {len(t7_nav)}")
print(f"index first 10: {list(t7_nav.index)[:10]}")
print(f"index dtype: {t7_nav.index.dtype}")
print(f"values first 5: {list(t7_nav.values[:5])}")
