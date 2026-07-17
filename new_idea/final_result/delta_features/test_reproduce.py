import os, gc, time
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
NEWS_MAJOR_DIR = r"C:\Users\liuqi\clowspace\toolstock\news_major1"
FINAL_DIR = r'c:\Users\liuqi\quant_system_v2\new_idea\final_result'

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']
CIRC_MV_LIMIT = 1000000

def is_gem_or_star(ts_code):
    return ('300' in ts_code) or ('301' in ts_code) or ('688' in ts_code) or ('689' in ts_code)

model, feats = joblib.load(os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib'))
print('Model loaded', flush=True)

all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])

trades = []
t_start = time.time()

for idx in range(1, len(all_dates) - 2):
    d_t = all_dates[idx]
    d_t1 = all_dates[idx + 1]
    d_t2 = all_dates[idx + 2]
    if d_t < '20230801' or d_t > '20231001':
        continue

    p_rank = os.path.join(RANK_DIR, f"{d_t}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d_t}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{d_t}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d_t}.parquet")
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other]):
        continue

    rank_df = pd.read_parquet(p_rank)
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close', 'open', 'high', 'low'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])

    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    if len(df) == 0:
        print(f"  {d_t}: EMPTY after rank merge (rank={len(rank_df)}, price={len(price_df)})", flush=True)
        del df, rank_df, chip_df, price_df, other_df
        gc.collect()
        continue
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['circ_mv'] <= CIRC_MV_LIMIT]
    df['news_market_impact'] = 0
    df['news_stock_impact'] = 0

    if len(df) == 0:
        print(f"  {d_t}: EMPTY after filters", flush=True)
        del df, rank_df, chip_df, price_df, other_df
        gc.collect()
        continue

    X = df[BASE_FEATS].fillna(0)
    proba = model.predict_proba(X)
    if proba.shape[1] < 2:
        print(f"  {d_t}: predict_proba shape={proba.shape}, X shape={X.shape}", flush=True)
        del df, rank_df, chip_df, price_df, other_df
        gc.collect()
        continue
    df['prob'] = proba[:, 1]

    picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
    if picks.empty:
        picks = df.sort_values('prob', ascending=False).head(1)

    n_above = len(df[df['prob'] > 0.8])
    print(f"  {d_t}: {len(df)} stocks, {n_above} prob>0.8, {len(picks)} picks", flush=True)

    for _, pick in picks.iterrows():
        trades.append({'ts_code': pick['ts_code'], 'prob': pick['prob'], 'date_t': d_t})

    del df, rank_df, chip_df, price_df, other_df, picks
    gc.collect()

t_end = time.time()
print(f"\nTotal trades: {len(trades)}, Time: {t_end-t_start:.1f}s", flush=True)

if trades:
    tdf = pd.DataFrame(trades)
    print(f"Prob stats: mean={tdf['prob'].mean():.4f}, min={tdf['prob'].min():.4f}, max={tdf['prob'].max():.4f}", flush=True)
