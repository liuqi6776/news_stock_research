import pandas as pd
import numpy as np

df = pd.read_parquet('data/all_features_v2.parquet')
df = df.sort_values(['ts_code', 'trade_date'])

df['next_open'] = df.groupby('ts_code')['open'].shift(-1)
df['return_1d_open'] = (df['exit_price_1d'] - df['next_open']) / df['next_open']

valid = df['return_1d_open'].dropna()
print(f'return_1d_open: count={len(valid)}, mean={valid.mean():.4f}, std={valid.std():.4f}')
print(f'  >0: {(valid>0).mean():.1%}, >0.01: {(valid>0.01).mean():.1%}')

old = df['return_1d'].dropna()
print(f'return_1d (old): count={len(old)}, mean={old.mean():.4f}, std={old.std():.4f}')
print(f'  Gap effect: {valid.mean()-old.mean():.4f}')

df.to_parquet('data/all_features_v2.parquet')
print('Saved!')
