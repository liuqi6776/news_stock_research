"""
fix_1d_target.py — 修正 target 为 T+1 开盘买入基准

输入:  data/all_features_v2.parquet  (由 step1_build_features.py 生成)
输出:  data/all_features_v2.parquet  (原地追加/更新 return_1d_open 列)

target 定义:
    return_1d_open = (d+2 收盘价 - d+1 开盘价) / d+1 开盘价
    即: T 日信号 → T+1 日开盘买入 → T+2 日收盘卖出 的实际收益率

运行方式:
    cd scripts/
    python fix_1d_target.py
"""
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_FILE = os.path.join(SCRIPT_DIR, 'data', 'all_features_v2.parquet')

df = pd.read_parquet(FEATURES_FILE)
df = df.sort_values(['ts_code', 'trade_date'])

# T+1 开盘价（即买入价，entry_price）
df['next_open'] = df.groupby('ts_code')['open'].shift(-1)
# T+2 收盘价（即卖出价）
df['d2_close'] = df.groupby('ts_code')['close'].shift(-2)

# target: T+1 开盘买入 → T+2 收盘卖出
df['return_1d_open'] = (df['d2_close'] - df['next_open']) / df['next_open']

# 清理临时列
df = df.drop(columns=['d2_close'], errors='ignore')

# 打印统计
valid = df['return_1d_open'].dropna()
print(f'return_1d_open (T+1_open -> T+2_close): count={len(valid)}, mean={valid.mean():.4f}, std={valid.std():.4f}')
print(f'  >0: {(valid>0).mean():.1%}, >1%: {(valid>0.01).mean():.1%}, >2%: {(valid>0.02).mean():.1%}')

if 'return_1d' in df.columns:
    old_1d = df['return_1d'].dropna()
    print(f'return_1d (old T_close -> T+1_close): count={len(old_1d)}, mean={old_1d.mean():.4f}')

df.to_parquet(FEATURES_FILE)
print(f'Saved: {FEATURES_FILE}')
