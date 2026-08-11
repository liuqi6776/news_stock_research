# -*- coding: utf-8 -*-
"""检查 other_day1(估值/流动性) 与 income1(基本面) 的完整覆盖, 决定正交因子可行性"""
import glob, pandas as pd, os, numpy as np
DATA = r'D:/iquant_data/data_v2'

# other_day1: 估值因子 PE/PB/circ_mv/volume_ratio/turnover_rate
fs = sorted(glob.glob(os.path.join(DATA, 'other_day1', '*.parquet')))
parts = []
for f in fs:
    if os.path.getsize(f) <= 1024: continue
    parts.append(pd.read_parquet(f))
od = pd.concat(parts, ignore_index=True)
od['trade_date'] = od['trade_date'].astype(int)
print('== other_day1:', len(od), '行, 时间', od.trade_date.min(), '~', od.trade_date.max())
print('   股票数:', od.ts_code.nunique())
for c in ['pe','pb','circ_mv','turnover_rate','volume_ratio']:
    m = od[c].replace([np.inf,-np.inf], np.nan)
    print(f'   {c}: 非空率 {m.notna().mean():.1%}, 中位 {m.median():.3g}')
# 月末快照覆盖
od_m = od[od.groupby(od.trade_date//100)['trade_date'].transform('max')==od.trade_date]
print('   月末快照:', len(od_m), '行')

# income1: 财务
fs2 = sorted(glob.glob(os.path.join(DATA, 'income1', '*.parquet')))
parts2 = []
for f in fs2:
    if os.path.getsize(f) <= 1024: continue
    parts2.append(pd.read_parquet(f))
inc = pd.concat(parts2, ignore_index=True)
print('\n== income1:', len(inc), '行, trade_date', inc.trade_date.min(), '~', inc.trade_date.max())
print('   ann_date', inc.ann_date.min(), '~', inc.ann_date.max())
print('   股票数:', inc.ts_code.nunique(), 'cols:', list(inc.columns))
