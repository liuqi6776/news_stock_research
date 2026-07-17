"""
Fix v2: Re-generate labeled panel using vectorized operations.
Avoids row-by-row assignment on the 2.9M row DataFrame.
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
    print("Re-generating labeled panel (vectorized)...", flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)

    # Load features panel - only need ts_code and date for labeling
    print("Loading features panel (key columns only)...", flush=True)
    keys = pd.read_parquet(os.path.join(THIS_DIR, 'ts_panel_features.parquet'),
                            columns=['ts_code', 'date'])
    print(f"Keys: {len(keys)} rows, {keys['date'].nunique()} dates", flush=True)

    # Build T+1/T+2 mapping
    print("Building T+1/T+2 mapping...", flush=True)
    unique_dates = sorted(keys['date'].unique())
    date_to_t1t2 = {}
    for d in unique_dates:
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

    print(f"Dates with T+1/T+2: {len(date_to_t1t2)}", flush=True)

    # Process in chunks by date
    print("Processing dates...", flush=True)
    label_records = []

    for i, (d_int, (t1, t2)) in enumerate(date_to_t1t2.items()):
        p1 = os.path.join(PRICE_DIR, f"{t1}.parquet")
        p2 = os.path.join(PRICE_DIR, f"{t2}.parquet")
        if not os.path.exists(p1) or not os.path.exists(p2):
            continue

        day_stocks = keys[keys['date'] == d_int]['ts_code'].values
        if len(day_stocks) == 0:
            continue

        price_t1 = pd.read_parquet(p1, columns=['ts_code', 'open'])
        price_t2 = pd.read_parquet(p2, columns=['ts_code', 'close'])

        m = pd.DataFrame({'ts_code': day_stocks})
        m = pd.merge(m, price_t1, on='ts_code', how='left')
        m = pd.merge(m, price_t2, on='ts_code', how='left')

        m['label_ret'] = m['close'] / m['open'] - 1
        m['label'] = (m['label_ret'] > 0.04).astype(np.float32)
        m['date'] = d_int

        label_records.append(m[['ts_code', 'date', 'label', 'label_ret']])

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(date_to_t1t2)} dates", flush=True)

    labels_df = pd.concat(label_records, ignore_index=True)
    labels_df = labels_df.dropna(subset=['label'])
    print(f"Labels: {len(labels_df)} rows, pos_rate={labels_df['label'].mean():.3f}", flush=True)

    # Now merge labels back into features panel
    print("Merging labels into features panel...", flush=True)

    # Load features in chunks and merge
    feat_panel = pd.read_parquet(os.path.join(THIS_DIR, 'ts_panel_features.parquet'))
    print(f"Features: {len(feat_panel)} rows", flush=True)

    # Merge
    feat_panel = pd.merge(feat_panel, labels_df, on=['ts_code', 'date'], how='left')
    labeled_df = feat_panel[feat_panel['label'].notna()].copy()
    print(f"Labeled: {len(labeled_df)} rows, pos_rate={labeled_df['label'].mean():.3f}", flush=True)

    labeled_df.to_parquet(os.path.join(THIS_DIR, 'ts_panel_labeled.parquet'), index=False)
    print(f"Saved!", flush=True)

if __name__ == "__main__":
    main()
