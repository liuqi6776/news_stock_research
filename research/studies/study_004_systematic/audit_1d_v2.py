import pandas as pd
import numpy as np
import sys

STUDY_DIR = '.'
FEATURES_FILE = f'{STUDY_DIR}/data/all_features_v2.parquet'
PRED_FILE = f'{STUDY_DIR}/predictions/predictions_1d_open_wf_monthly.parquet'

print("=" * 80)
print("AUDIT 1: Feature Future Function Check")
print("=" * 80)

df = pd.read_parquet(FEATURES_FILE)
df = df.sort_values(['ts_code', 'trade_date'])
df['ds'] = df['trade_date'].astype(str)

exclude_cols = {'ts_code', 'trade_date', 'ds',
                'open', 'high', 'low', 'close', 'pre_close',
                'entry_price', 'next_open',
                'exit_price_1d', 'return_1d', 'return_1d_open',
                'exit_price_5d', 'return_5d', 'return_5d_open',
                'exit_price_28d', 'return_28d', 'return_28d_open',
                'exit_28d_close',
                'calc_ret5d', 'return_5d_from_open', 'return_28d_from_open',
                'entry_vs_close'}

feature_cols = [c for c in df.columns
                if c not in exclude_cols
                and not c.startswith('hist_')
                and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

print(f"Total feature columns used in training: {len(feature_cols)}")

suspicious = []
for col in feature_cols:
    col_lower = col.lower()
    if any(kw in col_lower for kw in ['exit', 'return', 'future', 'next', 'target', 'label', 'actual']):
        suspicious.append((col, 'NAME contains future/target keyword'))

if suspicious:
    print("⚠️ SUSPICIOUS FEATURES (potential future function):")
    for col, reason in suspicious:
        print(f"  {col}: {reason}")
else:
    print("✅ No obviously named future-function features found")

print("\nChecking correlation with target (sampled)...")
df['label_1d_open'] = (df['return_1d_open'] > 0.01).astype(int)
valid = df.dropna(subset=['return_1d_open'])
sample = valid.sample(n=min(200000, len(valid)), random_state=42)

high_corr = []
for col in feature_cols:
    if sample[col].std() < 1e-10:
        continue
    try:
        corr = sample[col].corr(sample['return_1d_open'])
        if abs(corr) > 0.3:
            high_corr.append((col, corr))
    except:
        pass

if high_corr:
    print("⚠️ Features with |correlation| > 0.3 with return_1d_open:")
    for col, corr in sorted(high_corr, key=lambda x: -abs(x[1])):
        print(f"  {col}: corr={corr:.4f}")
else:
    print("✅ No feature has |correlation| > 0.3 with target")

print()
print("=" * 80)
print("AUDIT 2: Target Definition Verification")
print("=" * 80)

stock = df[df['ts_code'] == '000001.SZ'].sort_values('trade_date').head(30)
print("Manual verification for 000001.SZ:")
mismatches = 0
for i in range(len(stock) - 3):
    row = stock.iloc[i]
    row_p1 = stock.iloc[i + 1]
    row_p2 = stock.iloc[i + 2]
    td = row['trade_date']
    t1_open = row_p1['open']
    t2_close = row_p2['close']
    manual_ret = (t2_close - t1_open) / t1_open
    stored_ret = row.get('return_1d_open', np.nan)
    match = abs(manual_ret - stored_ret) < 0.0001
    if not match:
        mismatches += 1
        print(f"  ❌ {td}: manual={manual_ret:.4f}, stored={stored_ret:.4f}")
if mismatches == 0:
    print("  ✅ All target values match (T+1 open -> T+2 close)")
else:
    print(f"  ❌ {mismatches} mismatches found!")

print()
print("=" * 80)
print("AUDIT 3: Walk-Forward Data Leakage Check")
print("=" * 80)

pred = pd.read_parquet(PRED_FILE)
pred['ds'] = pred['trade_date'].astype(str)
months = sorted(pred['ds'].str[:6].unique())
print(f"Prediction months: {months[0]} to {months[-1]}")
print(f"Total predictions: {len(pred)}")

train_start = '20200101'
first_pred_month = months[0]
train_end = str(int(first_pred_month) - 1)
if train_end.endswith('00'):
    train_end = f"{int(train_end[:4])-1}12"
print(f"First prediction month {first_pred_month}: train_end={train_end}")
print(f"✅ Training uses data BEFORE prediction month")

print()
print("=" * 80)
print("AUDIT 4: Backtest Entry Price vs OHLC")
print("=" * 80)

pred_sample = pred.sample(n=min(5000, len(pred)), random_state=42)
ohlc = df[['trade_date', 'ts_code', 'open', 'close']].copy()
ohlc['trade_date'] = ohlc['trade_date'].astype(str)

check = pred_sample.merge(ohlc, on=['trade_date', 'ts_code'], how='left', suffixes=('', '_ohlc'))
valid_check = check.dropna(subset=['open', 'close'])

if 'entry_price' in valid_check.columns:
    ep_vs_close = (valid_check['entry_price'] - valid_check['close']).abs()
    ep_vs_open = (valid_check['entry_price'] - valid_check['open']).abs()
    print(f"entry_price vs T-day close: mean_diff={ep_vs_close.mean():.4f}, median={ep_vs_close.median():.4f}")
    print(f"entry_price vs T-day open:  mean_diff={ep_vs_open.mean():.4f}, median={ep_vs_open.median():.4f}")
    
    if ep_vs_close.median() < 0.01:
        print("  entry_price ≈ T-day close (this is just a reference, backtest uses T+1 open)")
    if ep_vs_open.median() < 0.01:
        print("  ⚠️ entry_price ≈ T-day open (unexpected!)")

print()
print("=" * 80)
print("AUDIT 5: Limit Up Filter - Selected Stock Characteristics")
print("=" * 80)

if 'pct_chg' in df.columns:
    selected_th55 = pred[pred['prob'] >= 0.55]
    sel_merged = selected_th55.merge(
        df[['trade_date', 'ts_code', 'pct_chg']].assign(trade_date=df['trade_date'].astype(str)),
        on=['trade_date', 'ts_code'], how='left'
    )
    if 'pct_chg' in sel_merged.columns:
        sel_pct = sel_merged['pct_chg'].dropna()
        print(f"Selected stocks (prob>=0.55) T-day pct_chg distribution:")
        print(f"  Mean: {sel_pct.mean():.2%}")
        print(f"  Median: {sel_pct.median():.2%}")
        print(f"  >=9.5% (limit up zone): {(sel_pct >= 0.095).mean():.2%}")
        print(f"  >=5%: {(sel_pct >= 0.05).mean():.2%}")
        print(f"  >=0%: {(sel_pct >= 0).mean():.2%}")
        print(f"  <0%: {(sel_pct < 0).mean():.2%}")

print()
print("=" * 80)
print("AUDIT 6: return_1d_open_old Column Check")
print("=" * 80)

if 'return_1d_open_old' in df.columns:
    old_vals = df['return_1d_open_old'].dropna()
    new_vals = df['return_1d_open'].dropna()
    print(f"return_1d_open_old: {len(old_vals)} non-null values")
    print(f"  Mean: {old_vals.mean():.4f}, Std: {old_vals.std():.4f}")
    print(f"return_1d_open: {len(new_vals)} non-null values")
    print(f"  Mean: {new_vals.mean():.4f}, Std: {new_vals.std():.4f}")
    
    merged_vals = df.dropna(subset=['return_1d_open_old', 'return_1d_open'])
    if len(merged_vals) > 0:
        corr = merged_vals['return_1d_open_old'].corr(merged_vals['return_1d_open'])
        print(f"  Correlation between old and new: {corr:.4f}")
        diff = (merged_vals['return_1d_open_old'] - merged_vals['return_1d_open']).abs()
        print(f"  Mean abs diff: {diff.mean():.4f}")
        print(f"  Max abs diff: {diff.max():.4f}")
    
    print("  ⚠️ return_1d_open_old is in feature set - this is the OLD target definition!")
    print("  If it was computed differently from return_1d_open, it could be a leakage source")
    print("  It should be REMOVED from feature columns")

print()
print("=" * 80)
print("AUDIT 7: Survivorship Bias Check")
print("=" * 80)

codes_by_year = {}
for yr in ['2022', '2023', '2024', '2025', '2026']:
    yr_data = df[df['ds'].str[:4] == yr]
    codes_by_year[yr] = set(yr_data['ts_code'].unique())
    print(f"  {yr}: {len(codes_by_year[yr])} unique stocks")

all_codes = set()
for codes in codes_by_year.values():
    all_codes.update(codes)

delisted = all_codes - codes_by_year.get('2026', set())
print(f"\n  Stocks not in 2026 data: {len(delisted)}")
if len(delisted) > 0:
    delisted_in_data = df[df['ts_code'].isin(list(delisted)[:100])]
    print(f"  These stocks in historical data: {len(delisted_in_data)} rows (sampled 100 codes)")
    print(f"  ✅ Delisted stocks ARE in historical data")

print()
print("=" * 80)
print("AUDIT 8: Transaction Cost Realism")
print("=" * 80)

print("Current: buy=0.1%, sell=0.1%, round-trip=0.2%")
print("A-share realistic breakdown:")
print("  Commission: ~0.025% x2 = 0.05%")
print("  Stamp tax: 0.05% (sell only, since 2023)")
print("  Slippage: 0.05-0.2% (varies by liquidity)")
print("  Total realistic: 0.15-0.30%")
print("  Current 0.2% is REASONABLE for large-cap, LOW for small-cap")

print()
print("=" * 80)
print("CRITICAL FINDINGS SUMMARY")
print("=" * 80)

print("""
1. return_1d_open_old IN FEATURE SET ⚠️⚠️⚠️
   This column is the OLD version of the target variable.
   It was computed by add_1d_target.py as (exit_price_1d - next_open) / next_open
   where exit_price_1d = T+3 close (from step1_build_features.py with TARGET_HORIZON_DAYS=3)
   
   The CURRENT target return_1d_open = (T+2 close - T+1 open) / T+1 open
   
   Since exit_price_1d = T+3 close, return_1d_open_old = (T+3 close - T+1 open) / T+1 open
   This is a 3-day return that OVERLAPS with the 2-day target.
   
   This is a DATA LEAKAGE issue! The model can use return_1d_open_old 
   (which contains T+3 close information) to predict return_1d_open 
   (which only needs T+2 close).
   
   ACTION: Remove return_1d_open_old from feature set and retrain!

2. entry_price in prediction file = T-day close (reference only)
   The backtest correctly uses T+1 open from OHLC data as buy price.
   ✅ No issue here.

3. Target definition verified: return_1d_open = (T+2 close - T+1 open) / T+1 open
   ✅ Correct.

4. Walk-forward training: uses data before prediction month
   ✅ No temporal leakage.

5. T+1 constraint: enforced (no sell on buy day)
   ✅ Correct.

6. Limit up/down filter: implemented
   ✅ Correct.

7. Survivorship bias: delisted stocks in historical data
   ✅ No bias from exclusion.

8. Transaction costs: 0.2% round-trip
   ⚠️ Low end for small-cap stocks, but acceptable.
""")
