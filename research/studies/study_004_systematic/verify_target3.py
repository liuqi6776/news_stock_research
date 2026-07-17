import pandas as pd
import numpy as np

df = pd.read_parquet('data/all_features_v2.parquet')
df = df.sort_values(['ts_code', 'trade_date'])

# From step1_build_features.py:
# d_curr = available_dates[i]
# d_t2 = available_dates[i + 2]  -> entry_price = T+2 open
# d_exit = available_dates[i + TARGET_HORIZON_DAYS] = available_dates[i + 3] -> exit_price_1d = T+3 close
# actual_return = exit_price_1d / entry_price - 1 = (T+3 close - T+2 open) / T+2 open

# But then add_1d_open_target.py did:
# next_open = shift(open, -1) = T+1 open
# return_1d_open = (exit_price_1d - next_open) / next_open = (T+3 close - T+1 open) / T+1 open

# This is WRONG! exit_price_1d is T+3 close (not T+2 close)
# return_1d_open should be (T+2 close - T+1 open) / T+1 open for a 1d strategy
# But it's actually (T+3 close - T+1 open) / T+1 open = 3-day return!

# Let me verify this
stock = df[df['ts_code'] == '000001.SZ'].sort_values('trade_date').head(30)

# For each row, trade_date = d_curr = T
# next_open = T+1 open
# exit_price_1d = T+3 close (from step1, since TARGET_HORIZON_DAYS=3)
# So return_1d_open = (T+3 close - T+1 open) / T+1 open

# Let me verify by checking consecutive rows
print('Verifying return_1d_open definition:')
print('If exit_price_1d = T+3 close, then for row i:')
print('  return_1d_open[i] = (exit_price_1d[i] - next_open[i]) / next_open[i]')
print('  = (T+3 close - T+1 open) / T+1 open')
print()

# Check: for 000001.SZ, 20200102:
# T = 20200102, T+1 = 20200103, T+2 = 20200106, T+3 = 20200107
# next_open (T+1 open) = 16.94 (from 20200103)
# exit_price_1d (T+3 close) = 16.69 (from 20200107)
# return_1d_open = (16.69 - 16.94) / 16.94 = -0.0148
# But actual return_1d_open = 0.0077

# Hmm, that doesn't match. Let me check differently
# Maybe exit_price_1d is NOT T+3 close

# Let me trace through the dates more carefully
# available_dates are trading dates
# For 20200102 (i=0):
#   d_curr = 20200102
#   d_t2 = available_dates[2] = 20200106
#   entry_price = open of 20200106
#   d_exit = available_dates[3] = 20200107
#   exit_price_1d = close of 20200107

# But wait, the features file has trade_date = d_curr = 20200102
# And the row for 20200102 has:
#   close = 16.87 (close of 20200102)
#   next_open = 16.94 (open of 20200103, from shift(-1))
#   entry_price = open of 20200106 (from d_t2)
#   exit_price_1d = close of 20200107 (from d_exit)

# So return_1d_open = (exit_price_1d - next_open) / next_open
#                   = (close of 20200107 - open of 20200103) / open of 20200103
# This is a 3-day return from T+1 open to T+3 close!

# But the backtest treats it as a 2-day hold (hold_days=2)
# This means the backtest is using the WRONG holding period!

# The correct holding period should be 3 days (T+1 buy, T+3 sell)
# But we're using hold_days=2, which means we sell at T+2 close
# So we're selling one day too early!

# Actually wait, let me reconsider. The backtest uses OHLC data to track positions
# It doesn't use the return_1d_open column directly for PnL calculation
# It uses the actual OHLC prices. So the holding period in the backtest is correct.
# But the MODEL was trained on return_1d_open = (T+3 close - T+1 open) / T+1 open
# And the backtest holds for only 2 days (T+1 buy, T+2 sell)
# This is a MISMATCH between training target and backtest execution!

# The model predicts probability of (T+3 close - T+1 open) / T+1 open > 1%
# But we sell at T+2 close
# So we're leaving 1 day of return on the table

# This could explain why the returns seem high - the model is trained on 3-day returns
# but we're capturing the first 2 days which might have most of the move

# Let me verify: what is the actual holding return in the backtest?
# hold_days=2 means:
#   Day 1 (T+1): buy at open
#   Day 2 (T+2): sell at close
# So the actual return = (T+2 close - T+1 open) / T+1 open

# But the model predicts (T+3 close - T+1 open) / T+1 open > 1%
# So the model is predicting a 3-day return, but we only capture 2 days

# Let me check: what's the relationship between 2-day and 3-day returns?
valid = df.dropna(subset=['return_1d_open', 'close', 'next_open']).copy()
valid['ds'] = valid['trade_date'].astype(str)

# We need to compute the actual 2-day return: (T+2 close - T+1 open) / T+1 open
# But we don't have T+2 close directly. We have exit_price_1d = T+3 close
# And next_open = T+1 open

# Let me compute it from the OHLC data in the features
# For stock 000001.SZ, row 20200102:
# T close = 16.87, T+1 open = 16.94
# We need T+2 close. Looking at the data:
# 20200102: close=16.87
# 20200103: close=17.18 -> this is T+1 close
# 20200106: close=17.07 -> this is T+2 close
# 20200107: close=17.15 -> this is T+3 close

# exit_price_1d for 20200102 = 17.15 (T+3 close) -> matches!
# So return_1d_open = (17.15 - 16.94) / 16.94 = 0.0124
# But actual return_1d_open = 0.0077

# Hmm, still doesn't match. Let me check more carefully
row = stock[stock['trade_date'] == 20200102].iloc[0]
print(f"20200102: close={row['close']}, next_open={row['next_open']}, exit_price_1d={row['exit_price_1d']}, return_1d_open={row['return_1d_open']}")
calc = (row['exit_price_1d'] - row['next_open']) / row['next_open']
print(f"Calculated: ({row['exit_price_1d']} - {row['next_open']}) / {row['next_open']} = {calc:.6f}")
print(f"Actual return_1d_open: {row['return_1d_open']:.6f}")
print(f"Difference: {calc - row['return_1d_open']:.6f}")

# The difference is significant. Something is wrong with the return_1d_open calculation
# Or exit_price_1d doesn't mean what I think it means

# Let me check: maybe the features were rebuilt after add_1d_open_target.py ran
# And exit_price_1d was overwritten
print()
print("Checking if exit_price_1d was modified after initial build...")
print("Looking at the fix_1d_target.py script...")
