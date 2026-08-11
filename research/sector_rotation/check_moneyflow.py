# -*- coding: utf-8 -*-
"""检查 moneyflow1 原始字段, 确认大单定义和时段可用性"""
import glob, os, pandas as pd, numpy as np
DATA = r'D:/iquant_data/data_v2'

fs = sorted(glob.glob(os.path.join(DATA, 'moneyflow1', '*.parquet')))
fs = [f for f in fs if os.path.getsize(f) > 1024]
print(f'moneyflow1: {len(fs)} files, {os.path.basename(fs[0])} ~ {os.path.basename(fs[-1])}')

df = pd.read_parquet(fs[-1])
print(f'cols: {list(df.columns)}')
print(f'rows: {len(df)}')
print(df.head(3).to_string()[:1200])
print(f'\n各列非空率:')
print(df.notna().mean().round(3).to_string())

# 检查是否有时段数据 (早盘/尾盘)
tc_col = 'ts_code' if 'ts_code' in df.columns else df.columns[0]
print(f'\n唯一ts_code数: {df[tc_col].nunique()}')
