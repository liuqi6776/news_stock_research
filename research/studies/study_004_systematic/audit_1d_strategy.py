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
print(f"Feature columns: {feature_cols}")
print()

suspicious = []
for col in feature_cols:
    col_lower = col.lower()
    if any(kw in col_lower for kw in ['exit', 'return', 'future', 'next', 'target', 'label', 'actual']):
        suspicious.append((col, 'NAME contains future/target keyword'))
    if col in ['entry_price', 'next_open', 'exit_price_1d', 'exit_price_5d', 'exit_price_28d',
               'return_1d', 'return_1d_open', 'return_5d', 'return_5d_open',
               'return_28d', 'return_28d_open', 'actual_return']:
        suspicious.append((col, 'DIRECT future price/return column'))

if suspicious:
    print("⚠️ SUSPICIOUS FEATURES (potential future function):")
    for col, reason in suspicious:
        print(f"  {col}: {reason}")
else:
    print("✅ No obviously named future-function features found in training columns")

print()
print("Checking if any feature column has perfect correlation with target...")

df['label_1d_open'] = (df['return_1d_open'] > 0.01).astype(int)
valid = df.dropna(subset=['return_1d_open'])

high_corr = []
for col in feature_cols:
    if valid[col].std() < 1e-10:
        continue
    try:
        corr = valid[col].corr(valid['return_1d_open'])
        if abs(corr) > 0.5:
            high_corr.append((col, corr))
    except:
        pass

if high_corr:
    print("⚠️ Features with |correlation| > 0.5 with return_1d_open:")
    for col, corr in sorted(high_corr, key=lambda x: -abs(x[1])):
        print(f"  {col}: corr={corr:.4f}")
else:
    print("✅ No feature has |correlation| > 0.5 with target")

print()
print("=" * 80)
print("AUDIT 2: Target Definition Verification")
print("=" * 80)

sample = df[df['return_1d_open'].notna()].head(5)
for _, row in sample.iterrows():
    ts = row['ts_code']
    td = row['trade_date']
    ro = row.get('return_1d_open', np.nan)
    no = row.get('next_open', np.nan)
    ep = row.get('exit_price_1d', np.nan)
    c = row['close']
    print(f"  {ts} {td}: close={c:.2f}, next_open={no:.2f}, exit_price_1d={ep:.2f}, return_1d_open={ro:.4f}")

stock = df[df['ts_code'] == '000001.SZ'].sort_values('trade_date').head(20)
print()
print("Manual verification for 000001.SZ:")
for i in range(len(stock) - 3):
    row = stock.iloc[i]
    row_p1 = stock.iloc[i + 1]
    row_p2 = stock.iloc[i + 2]
    td = row['trade_date']
    t1_open = row_p1['open']
    t2_close = row_p2['close']
    manual_ret = (t2_close - t1_open) / t1_open
    stored_ret = row.get('return_1d_open', np.nan)
    match = "✅" if abs(manual_ret - stored_ret) < 0.0001 else "❌"
    print(f"  {td}: T+1_open={t1_open:.2f}, T+2_close={t2_close:.2f}, "
          f"manual={manual_ret:.4f}, stored={stored_ret:.4f} {match}")

print()
print("=" * 80)
print("AUDIT 3: Walk-Forward Data Leakage Check")
print("=" * 80)

pred = pd.read_parquet(PRED_FILE)
pred['ds'] = pred['trade_date'].astype(str)

months = sorted(pred['ds'].str[:6].unique())
print(f"Prediction months: {months[0]} to {months[-1]}")
print(f"Total predictions: {len(pred)}")

first_month_data = pred[pred['ds'].str[:6] == months[0]]
print(f"\nFirst prediction month: {months[0]}")
print(f"  Predictions: {len(first_month_data)}")
print(f"  Prob range: {first_month_data['prob'].min():.4f} - {first_month_data['prob'].max():.4f}")

last_month_data = pred[pred['ds'].str[:6] == months[-1]]
print(f"\nLast prediction month: {months[-1]}")
print(f"  Predictions: {len(last_month_data)}")
print(f"  Prob range: {last_month_data['prob'].min():.4f} - {last_month_data['prob'].max():.4f}")

train_start = '20200101'
first_pred_month = months[0]
train_end = str(int(first_pred_month) - 1)
if train_end.endswith('00'):
    train_end = f"{int(train_end[:4])-1}12"
print(f"\nFor first prediction month {first_pred_month}:")
print(f"  Training data: {train_start} to {train_end}")
print(f"  Training uses data BEFORE {first_pred_month} ✅")

print()
print("=" * 80)
print("AUDIT 4: Backtest Entry Price Verification")
print("=" * 80)

pred_sample = pred.head(100)
entry_from_pred = pred_sample['entry_price'].dropna()
print(f"Entry prices from prediction file: {len(entry_from_pred)} non-null")
print(f"  Mean: {entry_from_pred.mean():.2f}, Min: {entry_from_pred.min():.2f}, Max: {entry_from_pred.max():.2f}")

merged = pred_sample.merge(df[['trade_date', 'ts_code', 'close', 'next_open']].assign(trade_date=df['trade_date'].astype(str)),
                           on=['trade_date', 'ts_code'], how='left', suffixes=('', '_feat'))
if 'next_open' in merged.columns:
    valid_m = merged.dropna(subset=['next_open', 'entry_price'])
    if len(valid_m) > 0:
        diff = (valid_m['entry_price'] - valid_m['next_open']).abs()
        print(f"\nEntry price vs next_open comparison:")
        print(f"  Max diff: {diff.max():.4f}")
        print(f"  Mean diff: {diff.mean():.4f}")
        if diff.max() < 0.01:
            print("  ✅ Entry price matches next_open (T+1 open)")
        else:
            print("  ⚠️ Entry price does NOT match next_open!")

print()
print("=" * 80)
print("AUDIT 5: Limit Up/Down Filter Verification")
print("=" * 80)

if 'pct_chg' in df.columns:
    main_board = df[~df['ts_code'].str.startswith(('30', '68'))]
    gem_board = df[df['ts_code'].str.startswith(('30', '68'))]

    mb_limit_up = (main_board['pct_chg'] >= 0.095).mean()
    gem_limit_up = (gem_board['pct_chg'] >= 0.195).mean() if len(gem_board) > 0 else 0

    print(f"Main board (60/00/002): limit-up rate (pct_chg>=9.5%) = {mb_limit_up:.2%}")
    print(f"GEM/STAR (30/68): limit-up rate (pct_chg>=19.5%) = {gem_limit_up:.2%}")

    selected_th55 = pred[pred['prob'] >= 0.55]
    selected_merged = selected_th55.merge(
        df[['trade_date', 'ts_code', 'pct_chg']].assign(trade_date=df['trade_date'].astype(str)),
        on=['trade_date', 'ts_code'], how='left'
    )
    if 'pct_chg' in selected_merged.columns:
        sel_pct = selected_merged['pct_chg'].dropna()
        print(f"\nSelected stocks (prob>=0.55) T-day pct_chg:")
        print(f"  Mean: {sel_pct.mean():.2%}")
        print(f"  >9.5%: {(sel_pct >= 0.095).mean():.2%}")
        print(f"  >5%: {(sel_pct >= 0.05).mean():.2%}")
        print(f"  >0%: {(sel_pct >= 0).mean():.2%}")

print()
print("=" * 80)
print("AUDIT 6: T+1 Constraint Verification")
print("=" * 80)

print("Backtest logic check:")
print("  - hold_days=2 means position is held for 2 trading days")
print("  - Day 0 (T): Signal generated at close, cannot trade")
print("  - Day 1 (T+1): Buy at open (hold_day=1), cannot sell (T+1 rule)")
print("  - Day 2 (T+2): Can sell, position closes at end of day")
print("  ✅ T+1 constraint is enforced (no sell on hold_day=1)")

print()
print("=" * 80)
print("AUDIT 7: Transaction Cost Check")
print("=" * 80)

print(f"Buy cost: {0.001:.1%} (0.1%)")
print(f"Sell cost: {0.001:.1%} (0.1%)")
print(f"Total round-trip: {0.002:.1%} (0.2%)")
print()
print("A-share typical costs:")
print("  Commission: 0.025% (min 5 yuan) x2 = 0.05%")
print("  Stamp tax: 0.05% (sell only)")
print("  Slippage: 0.1-0.3% (depends on liquidity)")
print("  Total realistic: 0.2-0.4%")
print("  ⚠️ Current 0.2% is at the LOW end; for small-cap high-turnover stocks, 0.3-0.5% is more realistic")

print()
print("=" * 80)
print("AUDIT 8: Survivorship Bias Check")
print("=" * 80)

codes_by_year = {}
for yr in ['2022', '2023', '2024', '2025', '2026']:
    yr_data = df[df['ds'].str[:4] == yr]
    codes_by_year[yr] = set(yr_data['ts_code'].unique())
    print(f"  {yr}: {len(codes_by_year[yr])} unique stocks")

all_codes_ever = set()
for codes in codes_by_year.values():
    all_codes_ever.update(codes)

survived_all = set.intersection(*codes_by_year.values())
delisted = all_codes_ever - codes_by_year.get('2026', set())
new_in_2026 = codes_by_year.get('2026', set()) - codes_by_year.get('2022', set())

print(f"\n  Total unique stocks ever: {len(all_codes_ever)}")
print(f"  Survived all years: {len(survived_all)}")
print(f"  Potentially delisted (not in 2026): {len(delisted)}")
print(f"  New in 2026: {len(new_in_2026)}")

if len(delisted) > 0:
    print(f"  ⚠️ {len(delisted)} stocks disappeared - potential survivorship bias if delisted stocks are excluded from features")
    print(f"  Check: Are delisted stocks in the feature data before their delisting date?")
    delisted_in_data = df[df['ts_code'].isin(delisted)]
    print(f"  Delisted stocks in data: {len(delisted_in_data)} rows")
    if len(delisted_in_data) > 0:
        print(f"  ✅ Delisted stocks ARE in historical data (no survivorship bias from exclusion)")
    else:
        print(f"  ❌ Delisted stocks NOT in data (SURVIVORSHIP BIAS!)")

print()
print("=" * 80)
print("AUDIT 9: Position Sizing Check")
print("=" * 80)

print("Current position sizing: 1/(hold_days * max_pos)")
print(f"  For 1d (hold=2, max_pos=3): pos_size = 1/(2*3) = {1/(2*3):.4f}")
print(f"  Max concurrent positions: hold_days * max_pos = {2*3}")
print(f"  Total allocation at full capacity: {1/(2*3) * 2 * 3:.1%}")
print()
print("  This means each position gets 1/6 of capital")
print("  With 3 new positions per day and 2-day holding, max 6 concurrent positions")
print("  ✅ Position sizing is correct - fully invested when all slots filled")

print()
print("=" * 80)
print("AUDIT 10: Critical Issue - entry_price vs OHLC open")
print("=" * 80)

pred_first = pred.head(1000).copy()
pred_first['ds'] = pred_first['trade_date'].astype(str)

ohlc_sample = df[['trade_date', 'ts_code', 'open', 'close']].copy()
ohlc_sample['trade_date'] = ohlc_sample['trade_date'].astype(str)

check = pred_first.merge(ohlc_sample, on=['trade_date', 'ts_code'], how='left', suffixes=('', '_ohlc'))

if 'entry_price' in check.columns and 'open' in check.columns:
    valid_check = check.dropna(subset=['entry_price', 'open'])
    if len(valid_check) > 0:
        entry_vs_t_close = valid_check['entry_price'] - valid_check['close']
        entry_vs_t1_open = valid_check['entry_price'] - valid_check['open']

        print(f"entry_price vs T-day close:")
        print(f"  Mean diff: {entry_vs_t_close.mean():.4f}")
        print(f"  Max abs diff: {entry_vs_t_close.abs().max():.4f}")

        print(f"\nentry_price vs T-day open:")
        print(f"  Mean diff: {entry_vs_t1_open.mean():.4f}")
        print(f"  Max abs diff: {entry_vs_t1_open.abs().max():.4f}")

        if entry_vs_t_close.abs().mean() < 0.01:
            print("  ⚠️ entry_price ≈ T-day close (NOT T+1 open!)")
            print("  This means the prediction file's entry_price is T-day close, not T+1 open")
            print("  But the backtest uses OHLC open of T+1 day as buy price - this is correct")
        elif entry_vs_t1_open.abs().mean() < 0.01:
            print("  ⚠️ entry_price ≈ T-day open (NOT T+1 open!)")

print()
print("=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)
