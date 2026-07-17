import os, gc, time, traceback
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
FINAL_DIR = r'c:\Users\liuqi\quant_system_v2\new_idea\final_result'

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']

doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
loaded = joblib.load(doubao_model_path)
base_model = loaded[0] if isinstance(loaded, tuple) else loaded
print('Model loaded', flush=True)

all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])

# Process first 5 test dates
for idx in range(1947, 1952):
    d_t = all_dates[idx]
    d_t1 = all_dates[idx + 1]
    d_t2 = all_dates[idx + 2]
    t0 = time.time()

    # Load features like doubao
    p_rank = os.path.join(RANK_DIR, f"{d_t}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d_t}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{d_t}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d_t}.parquet")

    rank_df = pd.read_parquet(p_rank, columns=['ts_code', 'hot'])
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    chip_df = pd.read_parquet(p_chip, columns=['ts_code', 'winner_rate', 'cost_85pct', 'cost_15pct', 'cost_50pct'])
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close', 'open', 'high', 'low'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])

    if len(rank_df) > 0:
        df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    else:
        df = price_df.copy()
        df['hot_rank_pct'] = 0.5
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['circ_mv'] <= 1000000]
    df['news_market_impact'] = 0
    df['news_stock_impact'] = 0

    # Base model prediction
    X = df[BASE_FEATS].fillna(0)
    df['prob'] = base_model.predict_proba(X)[:, 1]

    # Select like doubao
    picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
    if picks.empty:
        picks = df.sort_values('prob', ascending=False).head(1)

    # Load prev 1 day for TS features (only for picked stocks)
    ts_codes_set = set(picks['ts_code'].values)
    prev_idx = idx - 1
    prev_d = all_dates[prev_idx]

    prev_price = pd.read_parquet(os.path.join(PRICE_DIR, f"{prev_d}.parquet"), columns=['ts_code', 'close', 'vol'])
    prev_price = prev_price[prev_price['ts_code'].isin(ts_codes_set)]
    prev_chip = pd.read_parquet(os.path.join(CHIP_DIR, f"{prev_d}.parquet"), columns=['ts_code', 'winner_rate', 'cost_85pct', 'cost_15pct', 'cost_50pct'])
    prev_chip = prev_chip[prev_chip['ts_code'].isin(ts_codes_set)]
    prev_chip['chip_concentration'] = (prev_chip['cost_85pct'] - prev_chip['cost_15pct']) / (prev_chip['cost_50pct'] + 1e-8)

    prev_data = pd.merge(prev_price, prev_chip[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code', how='left')
    prev_dict = {rec['ts_code']: rec for rec in prev_data.to_dict('records')}

    # Compute TS features for picks only
    for _, pick in picks.iterrows():
        tc = pick['ts_code']
        if tc in prev_dict:
            p = prev_dict[tc]
            ret_1d = pick['close'] / (p.get('close', 0) + 1e-8) - 1
            delta_winner_rate = pick['winner_rate'] - p.get('winner_rate', 0)
            delta_chip_conc = pick['chip_concentration'] - p.get('chip_concentration', 0)
            print(f"  {tc}: prob={pick['prob']:.4f}, ret_1d={ret_1d:.4f}, dWR={delta_winner_rate:.4f}, dCC={delta_chip_conc:.4f}", flush=True)
        else:
            print(f"  {tc}: prob={pick['prob']:.4f}, NO PREV DATA", flush=True)

    t1 = time.time()
    n_picks = len(picks)
    n_above_08 = len(df[df['prob'] > 0.8])
    print(f"{d_t}: {len(df)} stocks, {n_above_08} prob>0.8, {n_picks} picks, time={t1-t0:.1f}s", flush=True)

    del df, picks, prev_data, prev_dict
    gc.collect()

print('DONE', flush=True)
