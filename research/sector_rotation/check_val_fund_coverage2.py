# -*- coding: utf-8 -*-
"""检查 income1/fundamental1 覆盖 (跳过损坏文件), 现有面板是否含 PE/PB"""
import glob, pandas as pd, os, numpy as np
DATA = r'D:/iquant_data/data_v2'
ROOT = r'c:\Users\liuqi\quant_system_v2'

# income1 全量 (跳过损坏)
fs2 = sorted(glob.glob(os.path.join(DATA, 'income1', '*.parquet')))
parts2 = []
for f in fs2:
    if os.path.getsize(f) <= 1024: continue
    try:
        parts2.append(pd.read_parquet(f))
    except Exception as e:
        print('skip', os.path.basename(f), type(e).__name__)
inc = pd.concat(parts2, ignore_index=True)
print('== income1:', len(inc), '行, trade_date', inc.trade_date.min(), '~', inc.trade_date.max())
print('   ann_date', inc.ann_date.min(), '~', inc.ann_date.max())
print('   股票数:', inc.ts_code.nunique(), 'cols:', list(inc.columns))
print(inc.head(3).to_string()[:800])

# fundamental1 覆盖
fd = pd.read_parquet(os.path.join(DATA, 'fundamental1', 'fina_indicator_cache.parquet'))
print('\n== fundamental1:', len(fd), '行, ann_date', fd.ann_date.min(), '~', fd.ann_date.max())
print('   按 ann_date 年份分布:')
print(fd.groupby(fd.ann_date.astype(str).str[:4]).size().to_string())

# 现有 ortho 面板列
panel = pd.read_parquet(os.path.join(ROOT, 'research/sector_rotation/stock_ml_panel_ortho_72m.parquet'))
print('\n== ortho 面板:', len(panel), '行, cols:', list(panel.columns))
