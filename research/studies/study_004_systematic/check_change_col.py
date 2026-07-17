import pandas as pd
import numpy as np

df = pd.read_parquet('data/all_features_v2.parquet')

print("Column 'change' sample:")
sample = df[df['ts_code'] == '000001.SZ'].sort_values('trade_date').tail(10)
for _, row in sample.iterrows():
    print(f"  {row['trade_date']}: close={row['close']:.2f}, pre_close={row.get('pre_close', np.nan):.2f}, "
          f"change={row.get('change', np.nan)}, pct_chg={row.get('pct_chg', np.nan):.4f}")

print("\nColumn 'change' stats:")
print(f"  dtype: {df['change'].dtype}")
print(f"  non-null: {df['change'].notna().sum()}")
print(f"  mean: {df['change'].mean():.4f}")
print(f"  Is change == close - pre_close?")
check = df.dropna(subset=['change', 'close', 'pre_close']).head(1000)
calc = check['close'] - check['pre_close']
diff = (check['change'] - calc).abs()
print(f"  Max diff between 'change' and (close - pre_close): {diff.max():.4f}")
if diff.max() < 0.01:
    print("  ✅ change = close - pre_close (T-day known, no future function)")
else:
    print("  ⚠️ change != close - pre_close, need further investigation")

print("\nAll feature columns and their descriptions:")
exclude_cols = {'ts_code', 'trade_date', 'ds',
                'open', 'high', 'low', 'close', 'pre_close',
                'entry_price', 'next_open',
                'exit_price_1d', 'return_1d', 'return_1d_open',
                'exit_price_5d', 'return_5d', 'return_5d_open',
                'exit_price_28d', 'return_28d', 'return_28d_open',
                'exit_28d_close',
                'calc_ret5d', 'return_5d_from_open', 'return_28d_from_open',
                'entry_vs_close',
                'return_1d_open_old', 'actual_return'}

feature_cols = [c for c in df.columns
                if c not in exclude_cols
                and not c.startswith('hist_')
                and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

print(f"\n{len(feature_cols)} features used in training:")
for col in feature_cols:
    n = df[col].notna().sum()
    print(f"  {col}: non-null={n}, dtype={df[col].dtype}")
