import os, gc, time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
CHIP_DIR = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
RANK_DIR = os.path.join(DATA_DIR, 'ths_rank1')
FINAL_DIR = r'c:\Users\liuqi\quant_system_v2\new_idea\final_result'

def is_main_board(ts_code):
    code = ts_code[:6]
    if code.startswith('60') and ts_code.endswith('.SH'): return True
    if code.startswith('00') and ts_code.endswith('.SZ'): return True
    return False

def int_to_date(date_int):
    s = str(int(date_int))
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))

all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])

doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
loaded = joblib.load(doubao_model_path)
base_model = loaded[0] if isinstance(loaded, tuple) else loaded
print('Model loaded')

ts_model_path = os.path.join(FINAL_DIR, 'delta_features', 'models', 'ts_model_v3.joblib')
ts_feat_path = os.path.join(FINAL_DIR, 'delta_features', 'models', 'ts_feat_cols_v3.joblib')
ts_model = joblib.load(ts_model_path)
ts_feat_cols = joblib.load(ts_feat_path)
print(f'TS model: {len(ts_feat_cols)} features')
print(f'TS feat cols: {ts_feat_cols}')

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate', 'news_market_impact', 'news_stock_impact']

for idx in range(1947, 1950):
    d_t = all_dates[idx]
    d = str(int(d_t))
    t0 = time.time()

    price = pd.read_parquet(os.path.join(PRICE_DIR, f'{d}.parquet'))
    price = price[price['ts_code'].apply(is_main_board)]
    chip = pd.read_parquet(os.path.join(CHIP_DIR, f'{d}.parquet'))
    chip = chip[chip['ts_code'].apply(is_main_board)]
    chip['chip_concentration'] = (chip['cost_85pct'] - chip['cost_15pct']) / (chip['cost_50pct'] + 1e-8)
    other = pd.read_parquet(os.path.join(OTHER_DIR, f'{d}.parquet'), columns=['ts_code', 'turnover_rate', 'volume_ratio', 'circ_mv'])
    other = other[other['ts_code'].apply(is_main_board)]

    current = pd.merge(price, chip[['ts_code', 'chip_concentration', 'winner_rate', 'cost_50pct', 'weight_avg']], on='ts_code', how='left')
    current = pd.merge(current, other, on='ts_code', how='left')
    current = current[current['circ_mv'] <= 1000000]
    current['date'] = int(d_t)

    rank_path = os.path.join(RANK_DIR, f'{d}.parquet')
    if os.path.exists(rank_path):
        rank = pd.read_parquet(rank_path)
        if len(rank) > 0:
            rank = rank[rank['ts_code'].apply(is_main_board)]
            if len(rank) > 0:
                rank['hot_rank_pct'] = rank['hot'].rank(pct=True)
                current = pd.merge(current, rank[['ts_code', 'hot_rank_pct']], on='ts_code', how='left')
                current['hot_rank_pct'] = current['hot_rank_pct'].fillna(0.5)
            else:
                current['hot_rank_pct'] = 0.5
        else:
            current['hot_rank_pct'] = 0.5
    else:
        current['hot_rank_pct'] = 0.5

    current['news_market_impact'] = 0
    current['news_stock_impact'] = 0

    for feat in BASE_FEATS:
        if feat not in current.columns:
            current[feat] = 0
    current['base_prob'] = base_model.predict_proba(current[BASE_FEATS].fillna(0))[:, 1]

    for feat in ts_feat_cols:
        if feat not in current.columns:
            current[feat] = 0
    current['ts_prob'] = ts_model.predict_proba(current[ts_feat_cols].fillna(0))[:, 1]
    current['comb_prob'] = 0.5 * current['base_prob'] + 0.5 * current['ts_prob']

    t1 = time.time()
    print(f'{d_t}: {len(current)} rows, base_prob mean={current["base_prob"].mean():.4f}, ts_prob mean={current["ts_prob"].mean():.4f}, time={t1-t0:.1f}s')

print('SUCCESS')
