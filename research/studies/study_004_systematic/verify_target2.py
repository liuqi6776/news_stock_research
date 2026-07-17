import pandas as pd
import numpy as np

df = pd.read_parquet('data/all_features_v2.parquet')
df['ds'] = df['trade_date'].astype(str)

# The verification shows that (exit_price_1d - next_open) / next_open != return_1d_open
# This means return_1d_open is NOT calculated from next_open and exit_price_1d
# Let me figure out what return_1d_open actually is

# Check: is return_1d_open = (exit_price_1d - close) / close ?
sample = df.dropna(subset=['return_1d_open', 'next_open', 'exit_price_1d', 'close']).head(20).copy()
sample['calc_from_close'] = (sample['exit_price_1d'] - sample['close']) / sample['close']
sample['calc_from_next_open'] = (sample['exit_price_1d'] - sample['next_open']) / sample['next_open']
sample['diff_close'] = (sample['calc_from_close'] - sample['return_1d_open']).abs()
sample['diff_next_open'] = (sample['calc_from_next_open'] - sample['return_1d_open']).abs()

print('Which formula matches return_1d_open?')
print(f'  Avg diff from (exit_price_1d - close)/close: {sample["diff_close"].mean():.6f}')
print(f'  Avg diff from (exit_price_1d - next_open)/next_open: {sample["diff_next_open"].mean():.6f}')

# Maybe return_1d_open is calculated differently
# Let me check: what if exit_price_1d is T+1 close (not T+2 close)?
# Then return_1d_open = (T+1 close - T+1 open) / T+1 open
# But that violates T+1 rules...

# Or maybe return_1d_open was built with a different formula
# Let me check the build_new_targets.py script
print()
print('Checking build_new_targets.py for return_1d_open definition...')

# Let me also check: is exit_price_1d = T+1 close or T+2 close?
# If exit_price_1d is T+1 close, then:
# return_1d_open = (T+1 close - T+1 open) / T+1 open
# This is a 1-day intraday return, which violates T+1 rules

# If exit_price_1d is T+2 close, then:
# return_1d_open = (T+2 close - T+1 open) / T+1 open
# This is a 2-day return, which is correct for T+1 trading

# Let me check by looking at a specific stock
stock = df[df['ts_code'] == '000001.SZ'].sort_values('ds').head(30)
print()
print('000001.SZ first 30 days:')
for _, row in stock.iterrows():
    td = row['trade_date']
    c = row['close']
    no = row['next_open']
    ep = row['exit_price_1d']
    r = row['return_1d_open']
    print(f'  {td}: close={c:.2f}, next_open={no:.2f}, exit_price_1d={ep:.2f}, return_1d_open={r:.4f}')
