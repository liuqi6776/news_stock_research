import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_backtest import *

all_dates = sorted([f.replace('.parquet','') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
all_dates_set = set(int(d) for d in all_dates)
news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)
delta_model = joblib.load(os.path.join(MODEL_DIR, 'delta_model.joblib'))

test_dates = [(idx, all_dates[idx]) for idx in range(5, len(all_dates) - 2)
               if all_dates[idx] >= TEST_START and all_dates[idx] <= TEST_END][:3]

for i, (idx, d_t) in enumerate(test_dates):
    prev_dates = [all_dates[idx - j] for j in range(1, min(6, idx))]
    d_t1 = all_dates[idx + 1]
    d_t2 = all_dates[idx + 2]
    try:
        df = load_delta_features(d_t, prev_dates, news_mkt, news_stk)
        if df is None:
            print(f'{d_t}: no data')
            continue
        X = df[DELTA_FEATS].fillna(0)
        df['prob'] = delta_model.predict_proba(X)[:, 1]

        df_t1 = pd.read_parquet(os.path.join(PRICE_DIR, f'{d_t1}.parquet'), columns=['ts_code', 'open', 'pre_close'])
        df_t2 = pd.read_parquet(os.path.join(PRICE_DIR, f'{d_t2}.parquet'), columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
        df_t1 = df_t1.rename(columns={'open': 'open_t1', 'pre_close': 'pre_close_t1'})
        df_t2 = df_t2.rename(columns={'open': 'open_t2', 'high': 'high_t2', 'low': 'low_t2',
                                       'close': 'close_t2', 'pre_close': 'pre_close_t2'})

        merged = pd.merge(df[['ts_code', 'prob', 'delta_winner_rate_1d', 'delta_chip_concentration_1d',
                               'ret_1d', 'ret_3d', 'delta_turnover_rate_1d',
                               'chip_price_diverge', 'vol_price_diverge']],
                          df_t1, on='ts_code', how='inner')
        merged = pd.merge(merged, df_t2, on='ts_code', how='inner')

        merged['is_gem'] = merged['ts_code'].str.contains('300|301|688|689', regex=True)
        merged['up_limit'] = np.where(merged['is_gem'],
                                       (merged['pre_close_t1'] * 1.2).round(2),
                                       (merged['pre_close_t1'] * 1.1).round(2))
        valid = merged[~merged['open_t1'].isna() & (merged['open_t1'] < merged['up_limit'])].copy()
        prob_mean = valid['prob'].mean()
        print(f'{d_t}: {len(df)} stocks -> {len(valid)} valid trades, prob mean={prob_mean:.4f}')
    except Exception as e:
        print(f'{d_t}: ERROR - {e}')
        traceback.print_exc()
