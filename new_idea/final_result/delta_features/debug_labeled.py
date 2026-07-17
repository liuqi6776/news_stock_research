import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')

feat = pd.read_parquet(r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\delta_features\ts_panel_features.parquet',
                        columns=['ts_code', 'date'])
print(f'Features panel: {len(feat)} rows')

labeled = pd.read_parquet(r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\delta_features\ts_panel_labeled.parquet',
                           columns=['ts_code', 'date', 'label'])
print(f'Labeled panel: {len(labeled)} rows')
print(f'Label non-null: {labeled["label"].notna().sum()}')

# Check: how many dates in features vs labeled
feat_dates = feat['date'].nunique()
labeled_dates = labeled['date'].nunique()
print(f'Feature dates: {feat_dates}, Labeled dates: {labeled_dates}')

# Check a specific date
d = 20230103
feat_day = feat[feat['date'] == d]
labeled_day = labeled[labeled['date'] == d]
print(f'\nDate {d}: features={len(feat_day)}, labeled={len(labeled_day)}, labeled_nonnull={labeled_day["label"].notna().sum()}')

# Check if T+1/T+2 price files exist
dt = datetime(2023, 1, 3)
all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
all_dates_set = set(int(d) for d in all_dates)

t1, t2 = None, None
for i in range(1, 10):
    nd = int((dt + timedelta(days=i)).strftime('%Y%m%d'))
    if nd in all_dates_set:
        if t1 is None:
            t1 = nd
        elif t2 is None:
            t2 = nd
            break
print(f'T+1={t1}, T+2={t2}')

# Check if these price files have the stocks
if t1:
    p1 = os.path.join(PRICE_DIR, f"{t1}.parquet")
    if os.path.exists(p1):
        price1 = pd.read_parquet(p1, columns=['ts_code', 'open'])
        print(f'T+1 price: {len(price1)} stocks')
        feat_stocks = set(feat_day['ts_code'])
        price_stocks = set(price1['ts_code'])
        overlap = feat_stocks & price_stocks
        print(f'Overlap: {len(overlap)} / {len(feat_stocks)}')
