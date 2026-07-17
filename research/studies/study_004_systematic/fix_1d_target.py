import pandas as pd
import numpy as np

df = pd.read_parquet('data/all_features_v2.parquet')
df = df.sort_values(['ts_code', 'trade_date'])

df['next_open'] = df.groupby('ts_code')['open'].shift(-1)
df['d2_close'] = df.groupby('ts_code')['close'].shift(-2)

df['return_1d_open'] = (df['d2_close'] - df['next_open']) / df['next_open']

if 'return_1d_open_old' not in df.columns:
    df['return_1d_open_old'] = df.get('return_1d_open_old', np.nan)

df = df.drop(columns=['d2_close'], errors='ignore')

valid = df['return_1d_open'].dropna()
print(f'return_1d_open (T+1_open -> T+2_close): count={len(valid)}, mean={valid.mean():.4f}, std={valid.std():.4f}')
print(f'  >0: {(valid>0).mean():.1%}, >0.01: {(valid>0.01).mean():.1%}')

old_1d = df['return_1d'].dropna()
print(f'return_1d (old, T_close -> T+1_close): count={len(old_1d)}, mean={old_1d.mean():.4f}, std={old_1d.std():.4f}')

df.to_parquet('data/all_features_v2.parquet')
print('Saved!')
