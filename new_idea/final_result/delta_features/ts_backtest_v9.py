"""
TS Enhanced Backtest v9 - Faithful reproduction of doubao strategy + TS re-ranking.
Key: Use same data loading and selection logic as doubao's strategy_code.py
"""
import os, sys, json, gc, traceback, time
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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.dirname(THIS_DIR)

CIRC_MV_LIMIT = 1000000
TEST_START = '20230101'
TEST_END = '20260324'

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']

def is_gem_or_star(ts_code):
    return ('300' in ts_code) or ('301' in ts_code) or ('688' in ts_code) or ('689' in ts_code)

def int_to_date(date_int):
    s = str(int(date_int))
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))

def get_next_trading_day(date_int, all_dates_set):
    current_dt = int_to_date(date_int)
    for i in range(1, 10):
        next_dt = current_dt + timedelta(days=i)
        next_int = int(next_dt.strftime('%Y%m%d'))
        if next_int in all_dates_set:
            return next_int
    return None

def process_news(news_dir):
    market_records, stock_records = [], []
    if not os.path.exists(news_dir):
        return pd.DataFrame(market_records), pd.DataFrame(stock_records)
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'):
            continue
        try:
            with open(os.path.join(news_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        date_str = data.get("article_date", "")
        if not date_str:
            continue
        trade_date = pd.to_datetime(date_str)
        market_records.append({'trade_date': trade_date, 'news_market_impact': float(data.get("market_impact", 0))})
        for s in data.get("stocks", []):
            code = s.get("stock_code")
            if not code:
                continue
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ" if (code.startswith('0') or code.startswith('3')) else code
            stock_records.append({'trade_date': trade_date, 'ts_code': ts_code, 'news_stock_impact': float(s.get("impact", 0))})
    return pd.DataFrame(market_records), pd.DataFrame(stock_records)

def add_news_features(df, d_curr, news_mkt, news_stk):
    if news_mkt is not None and not news_mkt.empty:
        nm = news_mkt.copy()
        if pd.api.types.is_datetime64_any_dtype(nm['trade_date']):
            nm['trade_date'] = nm['trade_date'].dt.strftime('%Y%m%d')
        same_date = nm[nm['trade_date'] == d_curr]
        if not same_date.empty:
            df['news_market_impact'] = same_date['news_market_impact'].mean()
        else:
            df['news_market_impact'] = 0.0
    else:
        df['news_market_impact'] = 0.0
    if news_stk is not None and not news_stk.empty:
        ns = news_stk.copy()
        if pd.api.types.is_datetime64_any_dtype(ns['trade_date']):
            ns['trade_date'] = ns['trade_date'].dt.strftime('%Y%m%d')
        same_date = ns[ns['trade_date'] == d_curr]
        if not same_date.empty:
            df = pd.merge(df, same_date[['ts_code', 'news_stock_impact']], on='ts_code', how='left')
            df['news_stock_impact'] = df['news_stock_impact'].fillna(0.0)
        else:
            df['news_stock_impact'] = 0.0
    else:
        df['news_stock_impact'] = 0.0
    return df

def load_features_like_doubao(d_curr, news_mkt, news_stk):
    """Exact same logic as doubao's load_features"""
    p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other]):
        return None

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
    df = df[df['circ_mv'] <= CIRC_MV_LIMIT]
    df = add_news_features(df, d_curr, news_mkt, news_stk)
    for f in BASE_FEATS:
        if f not in df.columns:
            df[f] = 0.0
    return df

def load_prev_data(d_prev, ts_codes_set):
    """Load previous day data for TS features."""
    p_chip = os.path.join(CHIP_DIR, f"{d_prev}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{d_prev}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d_prev}.parquet")
    p_rank = os.path.join(RANK_DIR, f"{d_prev}.parquet")

    if not os.path.exists(p_chip) or not os.path.exists(p_price) or not os.path.exists(p_other):
        return None

    price = pd.read_parquet(p_price, columns=['ts_code', 'close', 'vol', 'pct_chg'])
    price = price[price['ts_code'].isin(ts_codes_set)]
    chip = pd.read_parquet(p_chip, columns=['ts_code', 'winner_rate', 'cost_85pct', 'cost_15pct', 'cost_50pct'])
    chip = chip[chip['ts_code'].isin(ts_codes_set)]
    chip['chip_concentration'] = (chip['cost_85pct'] - chip['cost_15pct']) / (chip['cost_50pct'] + 1e-8)
    other = pd.read_parquet(p_other, columns=['ts_code', 'turnover_rate', 'volume_ratio'])
    other = other[other['ts_code'].isin(ts_codes_set)]

    df = pd.merge(price, chip[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code', how='left')
    df = pd.merge(df, other, on='ts_code', how='left')

    result = {}
    for rec in df.to_dict('records'):
        result[rec['ts_code']] = rec

    if os.path.exists(p_rank):
        rank = pd.read_parquet(p_rank, columns=['ts_code', 'hot'])
        rank = rank[rank['ts_code'].isin(ts_codes_set)]
        if len(rank) > 0:
            rank['hot_rank_pct'] = rank['hot'].rank(pct=True)
            for rec in rank.to_dict('records'):
                if rec['ts_code'] in result:
                    result[rec['ts_code']]['hot_rank_pct'] = rec['hot_rank_pct']

    return result

def compute_ts_features_for_picks(picks_df, current_df, prev_data_list):
    """Compute TS features only for picked stocks."""
    n = len(picks_df)
    if n == 0:
        return picks_df

    ts_codes = picks_df['ts_code'].values

    ret_1d = np.zeros(n)
    delta_chip_conc_1d = np.zeros(n)
    delta_winner_rate_1d = np.zeros(n)
    delta_vol_1d = np.zeros(n)
    delta_turnover_1d = np.zeros(n)
    delta_hot_rank_1d = np.zeros(n)
    ret_5d = np.zeros(n)
    vol_mean_5 = np.zeros(n)
    ma5 = np.zeros(n)

    current_dict = {}
    for _, row in current_df.iterrows():
        current_dict[row['ts_code']] = row

    for i in range(n):
        tc = ts_codes[i]
        curr = current_dict.get(tc, {})
        curr_close = curr.get('close', 0)
        curr_vol = curr.get('vol', 0)
        curr_chip_conc = curr.get('chip_concentration', 0)
        curr_winner_rate = curr.get('winner_rate', 0)
        curr_turnover = curr.get('turnover_rate', 0)
        curr_hot_rank = curr.get('hot_rank_pct', 0.5)

        if len(prev_data_list) >= 1 and tc in prev_data_list[0]:
            p = prev_data_list[0][tc]
            c1 = p.get('close', 0)
            if c1 > 0:
                ret_1d[i] = curr_close / c1 - 1
            v1 = p.get('vol', 0)
            if v1 > 0:
                delta_vol_1d[i] = curr_vol / v1 - 1
            delta_chip_conc_1d[i] = curr_chip_conc - p.get('chip_concentration', 0)
            delta_winner_rate_1d[i] = curr_winner_rate - p.get('winner_rate', 0)
            delta_turnover_1d[i] = curr_turnover - p.get('turnover_rate', 0)
            delta_hot_rank_1d[i] = curr_hot_rank - p.get('hot_rank_pct', 0.5)

        if len(prev_data_list) >= 5:
            vs, cs, cnt = 0.0, 0.0, 0
            for j in range(5):
                if tc in prev_data_list[j]:
                    vs += prev_data_list[j][tc].get('vol', 0)
                    cs += prev_data_list[j][tc].get('close', 0)
                    cnt += 1
            if cnt >= 3:
                vol_mean_5[i] = vs / cnt
                ma5[i] = cs / cnt
            if tc in prev_data_list[4]:
                c5 = prev_data_list[4][tc].get('close', 0)
                if c5 > 0:
                    ret_5d[i] = curr_close / c5 - 1

    picks_df = picks_df.copy()
    picks_df['ret_1d'] = ret_1d
    picks_df['delta_chip_conc_1d'] = delta_chip_conc_1d
    picks_df['delta_winner_rate_1d'] = delta_winner_rate_1d
    picks_df['delta_vol_1d'] = delta_vol_1d
    picks_df['delta_turnover_1d'] = delta_turnover_1d
    picks_df['delta_hot_rank_1d'] = delta_hot_rank_1d
    picks_df['ret_5d'] = ret_5d
    picks_df['vol_mean_5'] = vol_mean_5
    picks_df['vol_ratio_5d'] = curr_vol / (vol_mean_5 + 1e-8) if 'vol' in picks_df.columns else 0
    picks_df['ma5'] = ma5
    picks_df['ma5_dist'] = (curr_close / (ma5 + 1e-8) - 1) if 'close' in picks_df.columns else 0

    # TS composite score
    picks_df['ts_score'] = (
        -picks_df['ret_1d'].abs() * 0.3 +
        picks_df['delta_winner_rate_1d'] * 2.0 +
        -picks_df['delta_chip_conc_1d'].abs() * 1.0 +
        picks_df['delta_turnover_1d'] * 0.5 +
        -picks_df['ret_5d'].abs() * 0.1
    )

    return picks_df

def select_like_doubao(df, model, feats):
    """Exact same selection as doubao: prob > 0.8 Top3, fallback Top1"""
    X = df[feats].fillna(0)
    if len(X) == 0:
        return pd.DataFrame()
    df = df.copy()
    df['prob'] = model.predict_proba(X)[:, 1]
    picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
    if picks.empty:
        picks = df.sort_values('prob', ascending=False).head(1)
    return picks

def select_ts_rerank(df, model, feats, top_n=1):
    """Select using base model prob > 0.8, then re-rank by TS score"""
    X = df[feats].fillna(0)
    if len(X) == 0:
        return pd.DataFrame()
    df = df.copy()
    df['prob'] = model.predict_proba(X)[:, 1]
    candidates = df[df['prob'] > 0.8]
    if len(candidates) == 0:
        candidates = df.sort_values('prob', ascending=False).head(max(5, top_n))
    if 'ts_score' in candidates.columns:
        picks = candidates.nlargest(top_n, 'ts_score')
    else:
        picks = candidates.nlargest(top_n, 'prob')
    return picks

def select_combined(df, model, feats, top_n=1, ts_weight=0.3):
    """Select using combined base_prob + ts_score"""
    X = df[feats].fillna(0)
    if len(X) == 0:
        return pd.DataFrame()
    df = df.copy()
    df['prob'] = model.predict_proba(X)[:, 1]
    candidates = df[df['prob'] > 0.8]
    if len(candidates) == 0:
        candidates = df.sort_values('prob', ascending=False).head(max(5, top_n))
    if 'ts_score' in candidates.columns:
        ts_min = candidates['ts_score'].min()
        ts_max = candidates['ts_score'].max()
        if ts_max > ts_min:
            candidates = candidates.copy()
            candidates['ts_score_norm'] = (candidates['ts_score'] - ts_min) / (ts_max - ts_min)
        else:
            candidates['ts_score_norm'] = 0.5
        candidates['combined_score'] = (1 - ts_weight) * candidates['prob'] + ts_weight * candidates['ts_score_norm']
        picks = candidates.nlargest(top_n, 'combined_score')
    else:
        picks = candidates.nlargest(top_n, 'prob')
    return picks

def backtest(trades_df, all_dates_set, take_profit=None):
    if trades_df.empty:
        return pd.DataFrame(), {}
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    total_trades = 0
    cannot_sell_trades = 0

    for date_t2, group in trades_df.groupby('date_t2', sort=True):
        alloc = capital / len(group)
        day_pnl = 0.0
        for _, trade in group.iterrows():
            total_trades += 1
            ts_code = trade['ts_code']
            buy_price = trade['buy_price']
            sell_close = trade['sell_close']
            sell_high = trade['sell_high']
            sell_pre_close = trade['sell_pre_close']

            limit_down_pct = 0.8 if is_gem_or_star(ts_code) else 0.9
            limit_down_price = round(sell_pre_close * limit_down_pct, 2)
            is_cannot_sell = (sell_high == limit_down_price)

            if is_cannot_sell:
                cannot_sell_trades += 1
                date_t3 = get_next_trading_day(date_t2, all_dates_set)
                if date_t3:
                    p_t3 = os.path.join(PRICE_DIR, f"{date_t3}.parquet")
                    if os.path.exists(p_t3):
                        df_t3 = pd.read_parquet(p_t3, columns=['ts_code', 'open'])
                        t3_row = df_t3[df_t3['ts_code'] == ts_code]
                        sell_price = t3_row.iloc[0]['open'] if not t3_row.empty else sell_close
                    else:
                        sell_price = sell_close
                else:
                    sell_price = sell_close
            elif take_profit and sell_high >= buy_price * (1 + take_profit):
                sell_price = buy_price * (1 + take_profit)
            else:
                sell_price = sell_close

            ret = (sell_price / buy_price) - 1 - 0.0015
            day_pnl += alloc * ret

        capital += day_pnl
        equity.append({'date': int_to_date(date_t2), 'nav': capital})

    eq_df = pd.DataFrame(equity)
    if len(eq_df) == 0:
        return eq_df, {}
    total_ret = capital / initial_cap - 1
    years = len(eq_df) / 252.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    df_ret = eq_df['nav'].pct_change()
    mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    win_rate = (df_ret > 0).mean()
    return eq_df, {'total': total_ret, 'ann': ann_ret, 'sharpe': sharpe, 'mdd': mdd,
                   'calmar': calmar, 'win_rate': win_rate, 'trades': total_trades,
                   'cannot_sell': cannot_sell_trades, 'final_nav': capital}

def main():
    print("=" * 90, flush=True)
    print("  TS Enhanced Backtest v9 - Faithful doubao + TS re-ranking", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    print("\n[Step 1] Loading base model...", flush=True)
    doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
    loaded = joblib.load(doubao_model_path)
    base_model = loaded[0] if isinstance(loaded, tuple) else loaded
    print("  Base model loaded", flush=True)

    # Define schemes
    schemes = {
        'Doubao_Original': {'method': 'doubao', 'top_n': 3, 'tp': None},
        'Doubao_TP18': {'method': 'doubao', 'top_n': 3, 'tp': 0.18},
        'Doubao_TP20': {'method': 'doubao', 'top_n': 3, 'tp': 0.20},
        'TS_Rerank_Top1': {'method': 'ts_rerank', 'top_n': 1, 'tp': None},
        'TS_Rerank_Top2': {'method': 'ts_rerank', 'top_n': 2, 'tp': None},
        'TS_Rerank_Top3': {'method': 'ts_rerank', 'top_n': 3, 'tp': None},
        'TS_Rerank_Top1_TP18': {'method': 'ts_rerank', 'top_n': 1, 'tp': 0.18},
        'TS_Rerank_Top1_TP20': {'method': 'ts_rerank', 'top_n': 1, 'tp': 0.20},
        'Comb_w30_Top1': {'method': 'combined', 'top_n': 1, 'ts_weight': 0.3, 'tp': None},
        'Comb_w30_Top2': {'method': 'combined', 'top_n': 2, 'ts_weight': 0.3, 'tp': None},
        'Comb_w30_Top3': {'method': 'combined', 'top_n': 3, 'ts_weight': 0.3, 'tp': None},
        'Comb_w50_Top1': {'method': 'combined', 'top_n': 1, 'ts_weight': 0.5, 'tp': None},
        'Comb_w30_Top1_TP18': {'method': 'combined', 'top_n': 1, 'ts_weight': 0.3, 'tp': 0.18},
        'Comb_w30_Top1_TP20': {'method': 'combined', 'top_n': 1, 'ts_weight': 0.3, 'tp': 0.20},
    }

    scheme_trades = {sname: [] for sname in schemes}

    print("\n[Step 2] Generating trades for all schemes...", flush=True)
    test_dates = []
    for idx in range(5, len(all_dates) - 2):
        d = all_dates[idx]
        if d >= TEST_START and d <= TEST_END:
            test_dates.append((idx, d))

    total = len(test_dates)
    print(f"  Test dates: {total}", flush=True)

    for i, (idx, d_t) in enumerate(test_dates):
        try:
            d_t1 = all_dates[idx + 1]
            d_t2 = all_dates[idx + 2]

            # Load features like doubao
            df = load_features_like_doubao(d_t, news_mkt, news_stk)
            if df is None:
                continue

            # Load previous 5 days for TS features
            ts_codes_set = set(df['ts_code'].values)
            prev_data_list = []
            for j in range(1, 6):
                prev_idx = idx - j
                if prev_idx < 0:
                    break
                prev_d = all_dates[prev_idx]
                prev_dict = load_prev_data(prev_d, ts_codes_set)
                if prev_dict is not None:
                    prev_data_list.append(prev_dict)

            # Compute TS features for all stocks
            if prev_data_list:
                n = len(df)
                ts_codes = df['ts_code'].values
                close_vals = df['close'].values
                vol_vals = df['vol'].values
                chip_conc_vals = df['chip_concentration'].values
                winner_rate_vals = df['winner_rate'].values
                hot_rank_vals = df['hot_rank_pct'].values

                ret_1d = np.zeros(n)
                delta_chip_conc_1d = np.zeros(n)
                delta_winner_rate_1d = np.zeros(n)
                delta_vol_1d = np.zeros(n)
                delta_hot_rank_1d = np.zeros(n)
                ret_5d = np.zeros(n)
                ma5 = np.zeros(n)
                vol_mean_5 = np.zeros(n)

                for k in range(n):
                    tc = ts_codes[k]
                    if len(prev_data_list) >= 1 and tc in prev_data_list[0]:
                        p = prev_data_list[0][tc]
                        c1 = p.get('close', 0)
                        if c1 > 0:
                            ret_1d[k] = close_vals[k] / c1 - 1
                        v1 = p.get('vol', 0)
                        if v1 > 0:
                            delta_vol_1d[k] = vol_vals[k] / v1 - 1
                        delta_chip_conc_1d[k] = chip_conc_vals[k] - p.get('chip_concentration', 0)
                        delta_winner_rate_1d[k] = winner_rate_vals[k] - p.get('winner_rate', 0)
                        delta_hot_rank_1d[k] = hot_rank_vals[k] - p.get('hot_rank_pct', 0.5)

                    if len(prev_data_list) >= 5:
                        vs, cs, cnt = 0.0, 0.0, 0
                        for j in range(5):
                            if tc in prev_data_list[j]:
                                vs += prev_data_list[j][tc].get('vol', 0)
                                cs += prev_data_list[j][tc].get('close', 0)
                                cnt += 1
                        if cnt >= 3:
                            vol_mean_5[k] = vs / cnt
                            ma5[k] = cs / cnt
                        if tc in prev_data_list[4]:
                            c5 = prev_data_list[4][tc].get('close', 0)
                            if c5 > 0:
                                ret_5d[k] = close_vals[k] / c5 - 1

                df['ret_1d'] = ret_1d
                df['delta_chip_conc_1d'] = delta_chip_conc_1d
                df['delta_winner_rate_1d'] = delta_winner_rate_1d
                df['delta_vol_1d'] = delta_vol_1d
                df['delta_hot_rank_1d'] = delta_hot_rank_1d
                df['ret_5d'] = ret_5d
                df['vol_mean_5'] = vol_mean_5
                df['ma5'] = ma5
                df['ma5_dist'] = (df['close'] / (ma5 + 1e-8) - 1).fillna(0)

                df['ts_score'] = (
                    -df['ret_1d'].abs() * 0.3 +
                    df['delta_winner_rate_1d'] * 2.0 +
                    -df['delta_chip_conc_1d'].abs() * 1.0 +
                    -df['ret_5d'].abs() * 0.1
                )
            else:
                df['ts_score'] = 0.0
                df['ret_1d'] = 0.0
                df['delta_chip_conc_1d'] = 0.0
                df['delta_winner_rate_1d'] = 0.0
                df['delta_vol_1d'] = 0.0
                df['delta_hot_rank_1d'] = 0.0
                df['ret_5d'] = 0.0
                df['ma5_dist'] = 0.0

            del prev_data_list
            gc.collect()

            # Get T+1/T+2 prices
            p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
            p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                continue

            df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'pre_close'])
            df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])

            # Apply each scheme's selection
            for sname, scheme in schemes.items():
                method = scheme['method']

                if method == 'doubao':
                    picks = select_like_doubao(df, base_model, BASE_FEATS)
                elif method == 'ts_rerank':
                    picks = select_ts_rerank(df, base_model, BASE_FEATS, top_n=scheme['top_n'])
                elif method == 'combined':
                    picks = select_combined(df, base_model, BASE_FEATS, top_n=scheme['top_n'],
                                           ts_weight=scheme.get('ts_weight', 0.3))
                else:
                    continue

                if picks.empty:
                    continue

                for _, pick in picks.iterrows():
                    ts_code = pick['ts_code']
                    t1r = df_t1[df_t1['ts_code'] == ts_code]
                    if t1r.empty:
                        continue
                    t1 = t1r.iloc[0]
                    pre_close_t1 = t1['pre_close']
                    up_limit = round(pre_close_t1 * 1.2, 2) if is_gem_or_star(ts_code) else round(pre_close_t1 * 1.1, 2)
                    if pd.isna(t1['open']) or t1['open'] >= up_limit:
                        continue
                    t2r = df_t2[df_t2['ts_code'] == ts_code]
                    if t2r.empty:
                        continue
                    t2 = t2r.iloc[0]

                    scheme_trades[sname].append({
                        'date_t': d_t, 'date_t1': d_t1, 'date_t2': int(d_t2),
                        'ts_code': ts_code, 'buy_price': t1['open'],
                        'sell_open': t2['open'], 'sell_high': t2['high'],
                        'sell_close': t2['close'], 'sell_pre_close': t2['pre_close'],
                        'prob': pick['prob'],
                        'ts_score': pick.get('ts_score', 0),
                        'ret_1d': pick.get('ret_1d', 0),
                        'delta_winner_rate_1d': pick.get('delta_winner_rate_1d', 0),
                        'delta_chip_conc_1d': pick.get('delta_chip_conc_1d', 0),
                    })

            del df
            gc.collect()

            if (i + 1) % 50 == 0:
                counts = {s: len(t) for s, t in scheme_trades.items()}
                print(f"  {i+1}/{total} days, trades: {counts}", flush=True)

        except Exception as e:
            print(f"  ERROR on {d_t}: {e}", flush=True)
            traceback.print_exc()
            gc.collect()
            continue

    print("\n[Step 3] Backtesting...", flush=True)
    results = {}

    for sname, scheme in schemes.items():
        trades_list = scheme_trades[sname]
        if not trades_list:
            continue
        trades_df = pd.DataFrame(trades_list)
        tp = scheme.get('tp', None)
        eq, stats = backtest(trades_df, all_dates_set, take_profit=tp)
        if stats:
            results[sname] = (eq, stats, trades_df)
            print(f"  {sname:<30} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  "
                  f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}  Trades={stats['trades']:>5d}", flush=True)

    sorted_r = sorted(results.items(), key=lambda x: x[1][1]['sharpe'], reverse=True)

    print("\n[Step 4] Plotting...", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(18, 14))

    ax = axes[0]
    for sname, (eq, stats, _) in sorted_r[:8]:
        if not eq.empty:
            ax.plot(eq['date'], eq['nav'], label=f"{sname} (S={stats['sharpe']:.2f})")
    ax.set_title('Equity Curves - Top 8 Schemes')
    ax.set_xlabel('Date')
    ax.set_ylabel('NAV')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    names = [s[0] for s in sorted_r]
    sharpes = [s[1][1]['sharpe'] for s in sorted_r]
    totals = [s[1][1]['total'] for s in sorted_r]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, sharpes, width, label='Sharpe', color='steelblue')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, totals, width, label='Total Return', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Sharpe Ratio')
    ax2.set_ylabel('Total Return')
    ax.set_title('Scheme Comparison')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(THIS_DIR, 'ts_backtest_v9_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Chart saved", flush=True)

    print(f"\n{'Rank':>4} {'Scheme':<30} {'Total':>10} {'Ann':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'WinRate':>8} {'Trades':>7}")
    print('-' * 105)
    for rank, (sname, (eq, stats, tdf)) in enumerate(sorted_r, 1):
        print(f"{rank:>4} {sname:<30} {stats['total']:>9.2%} {stats['ann']:>9.2%} {stats['sharpe']:>7.2f} "
              f"{stats['mdd']:>9.2%} {stats['calmar']:>7.2f} {stats['win_rate']:>7.2%} {stats['trades']:>7}")

    for sname, (eq, stats, tdf) in sorted_r[:5]:
        if not eq.empty:
            eq.to_csv(os.path.join(THIS_DIR, f'equity_ts_v9_{sname}.csv'), index=False)
            tdf.to_csv(os.path.join(THIS_DIR, f'trades_ts_v9_{sname}.csv'), index=False)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
