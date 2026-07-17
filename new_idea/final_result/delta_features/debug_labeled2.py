import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')

labeled = pd.read_parquet(r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\delta_features\ts_panel_labeled.parquet',
                           columns=['date', 'label'])
print(f'Labeled rows: {len(labeled)}')
print(f'Unique dates: {labeled["date"].nunique()}')
print(f'Date samples: {sorted(labeled["date"].unique())[:20]}')
print(f'Label NaN count: {labeled["label"].isna().sum()}')

# The issue: only 13 dates have labels. Let's check what dates those are
labeled_dates = labeled[labeled['label'].notna()]['date'].unique()
print(f'\nDates with labels: {sorted(labeled_dates)}')

# Check: features panel dates
feat = pd.read_parquet(r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\delta_features\ts_panel_features.parquet',
                        columns=['date'])
feat_dates = sorted(feat['date'].unique())
print(f'\nFeature dates: {len(feat_dates)}')
print(f'First 10: {feat_dates[:10]}')
print(f'Last 10: {feat_dates[-10:]}')

# Check date types
print(f'\nFeature date dtype: {feat["date"].dtype}')
print(f'Feature date sample values: {feat["date"].values[:5]}')
