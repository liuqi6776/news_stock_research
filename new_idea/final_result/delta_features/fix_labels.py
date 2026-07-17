"""
Fix: Re-generate labeled panel with correct label assignment.
The previous version only labeled 13 dates due to merge issues.
"""
import os, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def int_to_date(date_int):
    s = str(int(date_int))
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))

def main():
    print("Re-generating labeled panel...", flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)

    # Load features panel
    feat_panel = pd.read_parquet(os.path.join(THIS_DIR, 'ts_panel_features.parquet'))
    print(f"Features: {len(feat_panel)} rows, {feat_panel['date'].nunique()} dates", flush=True)

    # Pre-load all T+1 open and T+2 close prices
    print("Pre-loading T+1/T+2 prices...", flush=True)

    # Build a mapping: for each date, find T+1 and T+2
    date_to_t1t2 = {}
    for d in feat_panel['date'].unique():
        d_int = int(d)
        dt = int_to_date(d_int)
        t1, t2 = None, None
        for i in range(1, 10):
            nd = int((dt + timedelta(days=i)).strftime('%Y%m%d'))
            if nd in all_dates_set:
                if t1 is None:
                    t1 = nd
                elif t2 is None:
                    t2 = nd
                    break
        if t1 is not None and t2 is not None:
            date_to_t1t2[d_int] = (t1, t2)

    print(f"Dates with T+1/T+2 mapping: {len(date_to_t1t2)}", flush=True)

    # Load all needed price files
    t1_dates = set(v[0] for v in date_to_t1t2.values())
    t2_dates = set(v[1] for v in date_to_t1t2.values())
    all_needed = t1_dates | t2_dates

    price_open_cache = {}
    price_close_cache = {}

    for i, d in enumerate(all_needed):
        p = os.path.join(PRICE_DIR, f"{d}.parquet")
        if not os.path.exists(p):
            continue
        pdf = pd.read_parquet(p, columns=['ts_code', 'open', 'close'])
        price_open_cache[d] = dict(zip(pdf['ts_code'], pdf['open']))
        price_close_cache[d] = dict(zip(pdf['ts_code'], pdf['close']))
        if (i + 1) % 200 == 0:
            print(f"  Loaded {i+1}/{len(all_needed)} price files", flush=True)

    print(f"Price cache: {len(price_open_cache)} open, {len(price_close_cache)} close", flush=True)

    # Now assign labels
    print("Assigning labels...", flush=True)
    feat_panel['label'] = np.float32(np.nan)
    feat_panel['label_ret'] = np.float32(np.nan)

    labeled_count = 0
    for d_int, (t1, t2) in date_to_t1t2.items():
        if t1 not in price_open_cache or t2 not in price_close_cache:
            continue

        mask = feat_panel['date'] == d_int
        day_indices = feat_panel.index[mask]

        open_map = price_open_cache[t1]
        close_map = price_close_cache[t2]

        for idx in day_indices:
            ts_code = feat_panel.at[idx, 'ts_code']
            open_t1 = open_map.get(ts_code, np.nan)
            close_t2 = close_map.get(ts_code, np.nan)

            if pd.notna(open_t1) and pd.notna(close_t2) and open_t1 > 0:
                ret = close_t2 / open_t1 - 1
                feat_panel.at[idx, 'label_ret'] = np.float32(ret)
                feat_panel.at[idx, 'label'] = np.float32(1 if ret > 0.04 else 0)
                labeled_count += 1

    print(f"Labeled: {labeled_count} / {len(feat_panel)} rows", flush=True)
    print(f"Label non-null: {feat_panel['label'].notna().sum()}", flush=True)
    print(f"Pos rate: {feat_panel[feat_panel['label'].notna()]['label'].mean():.3f}", flush=True)

    # Save
    labeled_df = feat_panel[feat_panel['label'].notna()].copy()
    labeled_df.to_parquet(os.path.join(THIS_DIR, 'ts_panel_labeled.parquet'), index=False)
    print(f"Saved: {len(labeled_df)} rows", flush=True)

if __name__ == "__main__":
    main()
