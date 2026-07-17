import os
import pandas as pd
import numpy as np

# Define paths
DATA_DIR = r"c:\Users\liuqi\quant_system_v2\etf-valuation-strategy\data"

# Test Nasdaq adjustment
df_nasdaq = pd.read_csv(os.path.join(DATA_DIR, 'nasdaq_etf_daily.csv'))
df_nasdaq['trade_date'] = pd.to_datetime(df_nasdaq['trade_date'].astype(str))
df_nasdaq.sort_values('trade_date', inplace=True)
df_nasdaq.reset_index(drop=True, inplace=True)

split_date = pd.to_datetime('2022-01-14')
adj_factor = 1.038 / 5.192
mask = df_nasdaq['trade_date'] < split_date

print("Before adjustment:")
print(df_nasdaq[df_nasdaq['trade_date'] == '2022-01-13'][['trade_date', 'close']])
print(df_nasdaq[df_nasdaq['trade_date'] == '2022-01-14'][['trade_date', 'close']])

for col in ['close', 'open', 'high', 'low', 'pre_close']:
    df_nasdaq.loc[mask, col] *= adj_factor

print("\nAfter adjustment:")
print(df_nasdaq[df_nasdaq['trade_date'] == '2022-01-13'][['trade_date', 'close']])
print(df_nasdaq[df_nasdaq['trade_date'] == '2022-01-14'][['trade_date', 'close']])

# Recalculate CAGR and MDD of Nasdaq alone
is_start, is_end = '2015-01-01', '2024-02-05'
df_sub = df_nasdaq[(df_nasdaq['trade_date'] >= is_start) & (df_nasdaq['trade_date'] <= is_end)].reset_index(drop=True)
years = (df_sub['trade_date'].iloc[-1] - df_sub['trade_date'].iloc[0]).days / 365.25
cagr = (df_sub['close'].iloc[-1] / df_sub['close'].iloc[0])**(1.0/years) - 1
cum_max = df_sub['close'].cummax()
dd = (df_sub['close'] - cum_max) / cum_max
mdd = dd.min()
print(f"\nAdjusted Nasdaq In-Sample CAGR: {cagr:.2%}  MDD: {mdd:.2%}")
