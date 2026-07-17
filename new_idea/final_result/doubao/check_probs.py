import os, pandas as pd, numpy as np, joblib
DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate', 'news_market_impact', 'news_stock_impact']
model, feats = joblib.load('models/doubao_t1t2_model.joblib')
all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])

count_above_08 = 0
total_days = 0

for idx in range(1, len(all_dates) - 2):
    d_t = all_dates[idx]
    if d_t < '20230101':
        continue
    if d_t > '20260324':
        break

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

    X = df[FEATS].fillna(0)
    df['prob'] = model.predict_proba(X)[:, 1]

    n_above_08 = len(df[df['prob'] > 0.8])
    total_days += 1
    if n_above_08 > 0:
        count_above_08 += 1
        top = df[df['prob'] > 0.8].nlargest(1, 'prob').iloc[0]
        print(f"  {d_t}: {n_above_08} prob>0.8, top={top['ts_code']} prob={top['prob']:.4f}", flush=True)

    if total_days <= 10 or total_days % 100 == 0:
        top1 = df.nlargest(1, 'prob').iloc[0]
        print(f"  Day {total_days}: {d_t} stocks={len(df)} above08={n_above_08} top1={top1['ts_code']} prob={top1['prob']:.4f}", flush=True)

print(f"\nTotal: {total_days} days, {count_above_08} days with prob>0.8 ({count_above_08/total_days*100:.1f}%)", flush=True)
