import pandas as pd
import numpy as np

df = pd.read_parquet('data/all_features_v2.parquet')
df['ds'] = df['trade_date'].astype(str)

sample = df.dropna(subset=['return_1d_open', 'next_open', 'exit_price_1d']).head(10)
print('Sample data:')
print(sample[['ts_code', 'trade_date', 'close', 'next_open', 'exit_price_1d', 'return_1d_open']].to_string())

sample = sample.copy()
sample['calc'] = (sample['exit_price_1d'] - sample['next_open']) / sample['next_open']
sample['diff'] = sample['calc'] - sample['return_1d_open']
print()
print('Verification (calc - actual):')
print(sample[['ts_code', 'trade_date', 'calc', 'return_1d_open', 'diff']].to_string())
max_diff = sample['diff'].abs().max()
print(f'Max diff: {max_diff:.8f}')

print()
print('=== Key question: is return_1d_open really T+1_open to T+2_close? ===')
print('Or is it T close to T+1 close (which would be data leakage)?')

# Check: what is exit_price_1d?
# If return_1d_open = (exit_price_1d - next_open) / next_open
# And next_open = T+1 open
# Then exit_price_1d should be T+2 close

# Let me verify with actual OHLC
ohlc = pd.read_parquet('data/ashare_ohlc.parquet')
ohlc['ds'] = ohlc['trade_date'].astype(str)

# Pick a specific stock and date
row = df.dropna(subset=['return_1d_open', 'next_open', 'exit_price_1d']).iloc[0]
ts_code = row['ts_code']
trade_date = row['trade_date']
close_t = row['close']
next_open = row['next_open']
exit_price = row['exit_price_1d']
ret = row['return_1d_open']

print(f'\nStock: {ts_code}, Date: {trade_date}')
print(f'  T close: {close_t}')
print(f'  T+1 open (next_open): {next_open}')
print(f'  exit_price_1d: {exit_price}')
print(f'  return_1d_open: {ret:.6f}')
print(f'  (exit_price - next_open) / next_open = {(exit_price - next_open) / next_open:.6f}')

# Find this stock in OHLC data
stock_ohlc = ohlc[ohlc['ts_code'] == ts_code].sort_values('ds')
dates = stock_ohlc['ds'].tolist()
if str(trade_date) in dates:
    idx = dates.index(str(trade_date))
    if idx + 2 < len(dates):
        t_close = stock_ohlc.iloc[idx]['close']
        t1_open = stock_ohlc.iloc[idx+1]['open']
        t2_close = stock_ohlc.iloc[idx+2]['close']
        print(f'\n  OHLC verification:')
        print(f'  T close (from OHLC): {t_close}')
        print(f'  T+1 open (from OHLC): {t1_open}')
        print(f'  T+2 close (from OHLC): {t2_close}')
        print(f'  (T+2 close - T+1 open) / T+1 open = {(t2_close - t1_open) / t1_open:.6f}')
        print(f'  return_1d_open from features: {ret:.6f}')
        if abs((t2_close - t1_open) / t1_open - ret) < 0.001:
            print('  MATCH: return_1d_open = (T+2 close - T+1 open) / T+1 open')
        else:
            print('  MISMATCH! Checking alternatives...')
            print(f'  (T+1 close - T+1 open) / T+1 open = ?')
            t1_close = stock_ohlc.iloc[idx+1]['close']
            print(f'  T+1 close (from OHLC): {t1_close}')
            print(f'  (T+1 close - T+1 open) / T+1 open = {(t1_close - t1_open) / t1_open:.6f}')
