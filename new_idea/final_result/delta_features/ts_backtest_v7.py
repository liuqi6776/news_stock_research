"""
TS Enhanced Backtest v7 - Minimal memory, no caching.
Load prev data on-the-fly, compute features, discard immediately.
"""
import os, sys, json, gc, traceback
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

def is_main_board(ts_code):
    code = ts_code[:6]
    if code.startswith(('60',)) and ts_code.endswith('.SH'):
        return True
    if code.startswith(('00',)) and ts_code.endswith('.SZ'):
        return True
    return False

def is_gem_or_star(ts_code):
    return any(x in ts_code for x in ['300', '301', '688', '689'])

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

def load_day_data(d_int):
    d = str(d_int)
    p_price = os.path.join(PRICE_DIR, f"{d}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d}.parquet")
    p_rank = os.path.join(RANK_DIR, f"{d}.parquet")

    if not os.path.exists(p_price) or not os.path.exists(p_chip) or not os.path.exists(p_other):
        return None

    price = pd.read_parquet(p_price, columns=['ts_code', 'open', 'close', 'high', 'low',
                                               'pct_chg', 'vol', 'amount', 'pre_close'])
    price = price[price['ts_code'].apply(is_main_board)]

    chip = pd.read_parquet(p_chip)
    chip = chip[chip['ts_code'].apply(is_main_board)]
    chip['chip_concentration'] = (chip['cost_85pct'] - chip['cost_15pct']) / (chip['cost_50pct'] + 1e-8)

    other = pd.read_parquet(p_other, columns=['ts_code', 'turnover_rate', 'volume_ratio', 'circ_mv'])
    other = other[other['ts_code'].apply(is_main_board)]

    df = pd.merge(price, chip[['ts_code', 'chip_concentration', 'winner_rate', 'cost_50pct', 'weight_avg']], on='ts_code', how='left')
    df = pd.merge(df, other, on='ts_code', how='left')
    df = df[df['circ_mv'] <= CIRC_MV_LIMIT]

    if os.path.exists(p_rank):
        rank = pd.read_parquet(p_rank)
        if len(rank) > 0:
            rank = rank[rank['ts_code'].apply(is_main_board)]
            if len(rank) > 0:
                rank['hot_rank_pct'] = rank['hot'].rank(pct=True)
                df = pd.merge(df, rank[['ts_code', 'hot_rank_pct']], on='ts_code', how='left')
                df['hot_rank_pct'] = df['hot_rank_pct'].fillna(0.5)
            else:
                df['hot_rank_pct'] = 0.5
        else:
            df['hot_rank_pct'] = 0.5
    else:
        df['hot_rank_pct'] = 0.5

    return df

def load_prev_series(d_int, ts_codes_set):
    """Load only close/vol/chip_conc/winner_rate/hot_rank_pct/turnover_rate for given ts_codes.
    Returns dict: {ts_code: {col: val}} or None.
    """
    d = str(d_int)
    p_price = os.path.join(PRICE_DIR, f"{d}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d}.parquet")

    if not os.path.exists(p_price) or not os.path.exists(p_chip) or not os.path.exists(p_other):
        return None

    price = pd.read_parquet(p_price, columns=['ts_code', 'close', 'vol'])
    price = price[price['ts_code'].isin(ts_codes_set)]

    chip = pd.read_parquet(p_chip, columns=['ts_code', 'winner_rate', 'cost_85pct', 'cost_15pct', 'cost_50pct'])
    chip = chip[chip['ts_code'].isin(ts_codes_set)]
    chip['chip_concentration'] = (chip['cost_85pct'] - chip['cost_15pct']) / (chip['cost_50pct'] + 1e-8)

    other = pd.read_parquet(p_other, columns=['ts_code', 'turnover_rate'])
    other = other[other['ts_code'].isin(ts_codes_set)]

    df = pd.merge(price, chip[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code', how='left')
    df = pd.merge(df, other, on='ts_code', how='left')

    # Convert to dict using to_dict('records') - much faster than iterrows
    result = {}
    for rec in df.to_dict('records'):
        result[rec['ts_code']] = rec
    return result

def compute_ts_features(current_df, prev_data_list):
    """Compute TS features from prev_data_list (list of dicts).
    prev_data_list[0] = yesterday, prev_data_list[1] = 2 days ago, etc.
    """
    n = len(current_df)
    ts_codes = current_df['ts_code'].values
    close_vals = current_df['close'].values
    vol_vals = current_df['vol'].values
    chip_conc_vals = current_df['chip_concentration'].values
    winner_rate_vals = current_df['winner_rate'].values
    hot_rank_vals = current_df['hot_rank_pct'].values
    turnover_vals = current_df['turnover_rate'].values

    ret_1d = np.zeros(n)
    ret_2d = np.zeros(n)
    ret_5d = np.zeros(n)
    ret_10d = np.zeros(n)
    log_ret_1d = np.zeros(n)
    log_ret_2d = np.zeros(n)
    delta_vol_1d = np.zeros(n)
    delta_chip_conc_1d = np.zeros(n)
    delta_winner_rate_1d = np.zeros(n)
    delta_hot_rank_1d = np.zeros(n)
    delta_turnover_1d = np.zeros(n)
    vol_mean_5 = np.zeros(n)
    ma5 = np.zeros(n)
    ma10 = np.zeros(n)
    rsi_14 = np.full(n, 50.0)

    for i in range(n):
        tc = ts_codes[i]
        c = close_vals[i]
        v = vol_vals[i]

        # 1-day
        if len(prev_data_list) >= 1 and tc in prev_data_list[0]:
            p = prev_data_list[0][tc]
            c1 = p['close']
            if c1 > 0:
                ret_1d[i] = c / c1 - 1
                log_ret_1d[i] = np.log(c / c1)
            v1 = p['vol']
            if v1 > 0:
                delta_vol_1d[i] = v / v1 - 1
            delta_chip_conc_1d[i] = chip_conc_vals[i] - p.get('chip_concentration', 0)
            delta_winner_rate_1d[i] = winner_rate_vals[i] - p.get('winner_rate', 0)
            delta_hot_rank_1d[i] = hot_rank_vals[i] - 0.5  # prev day default
            delta_turnover_1d[i] = turnover_vals[i] - p.get('turnover_rate', 0)

        # 2-day
        if len(prev_data_list) >= 2 and tc in prev_data_list[1]:
            c2 = prev_data_list[1][tc]['close']
            if c2 > 0:
                ret_2d[i] = c / c2 - 1
                log_ret_2d[i] = np.log(c / c2)

        # 5-day
        if len(prev_data_list) >= 5:
            vs, cs, cnt = 0, 0, 0
            for j in range(5):
                if tc in prev_data_list[j]:
                    vs += prev_data_list[j][tc]['vol']
                    cs += prev_data_list[j][tc]['close']
                    cnt += 1
            if cnt >= 3:
                vol_mean_5[i] = vs / cnt
                ma5[i] = cs / cnt
            if tc in prev_data_list[4] and prev_data_list[4][tc]['close'] > 0:
                ret_5d[i] = c / prev_data_list[4][tc]['close'] - 1

        # 10-day
        if len(prev_data_list) >= 10:
            cs10, cnt10 = 0, 0
            for j in range(10):
                if tc in prev_data_list[j]:
                    cs10 += prev_data_list[j][tc]['close']
                    cnt10 += 1
            if cnt10 >= 5:
                ma10[i] = cs10 / cnt10
            if tc in prev_data_list[9] and prev_data_list[9][tc]['close'] > 0:
                ret_10d[i] = c / prev_data_list[9][tc]['close'] - 1

        # RSI
        if len(prev_data_list) >= 14:
            gains, losses = 0, 0
            for j in range(13):
                if tc in prev_data_list[j] and tc in prev_data_list[j + 1]:
                    ch = prev_data_list[j][tc]['close'] - prev_data_list[j + 1][tc]['close']
                    if ch > 0:
                        gains += ch
                    else:
                        losses += abs(ch)
            if gains + losses > 0:
                rs = gains / (losses + 1e-8)
                rsi_14[i] = 100 - 100 / (1 + rs)

    df = current_df.copy()
    df['ret_1d'] = ret_1d
    df['ret_2d'] = ret_2d
    df['ret_5d'] = ret_5d
    df['ret_10d'] = ret_10d
    df['log_ret_1d'] = log_ret_1d
    df['log_ret_2d'] = log_ret_2d
    df['delta_vol_1d'] = delta_vol_1d
    df['delta_chip_conc_1d'] = delta_chip_conc_1d
    df['delta_winner_rate_1d'] = delta_winner_rate_1d
    df['delta_hot_rank_1d'] = delta_hot_rank_1d
    df['delta_turnover_1d'] = delta_turnover_1d
    df['vol_mean_5'] = vol_mean_5
    df['vol_ratio_5d'] = df['vol'] / (vol_mean_5 + 1e-8)
    df['ma5'] = ma5
    df['ma5_dist'] = (df['close'] / (ma5 + 1e-8) - 1).fillna(0)
    df['ma10'] = ma10
    df['ma10_dist'] = (df['close'] / (ma10 + 1e-8) - 1).fillna(0)
    df['rsi_14'] = rsi_14
    df['vol_price_diverge'] = delta_vol_1d - np.abs(ret_1d)
    df['chip_price_diverge'] = delta_chip_conc_1d - ret_1d
    df['ret_accel'] = df['ret_1d'] - df.get('ret_5d', pd.Series(0, index=df.index)) / 5.0

    df['intraday_range'] = (df['high'] - df['low']) / (df['pre_close'] + 1e-8)
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['pre_close'] + 1e-8)
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['pre_close'] + 1e-8)

    dt = int_to_date(int(current_df['date'].iloc[0]))
    df['day_of_week'] = dt.weekday()
    df['month'] = dt.month
    df['is_month_start'] = 1 if dt.day <= 5 else 0
    df['is_month_end'] = 1 if dt.day >= 25 else 0

    return df

def add_news_features(df, d_curr, news_mkt, news_stk):
    if news_mkt is not None and not news_mkt.empty:
        nm = news_mkt.copy()
        if pd.api.types.is_datetime64_any_dtype(nm['trade_date']):
            nm['trade_date'] = nm['trade_date'].dt.strftime('%Y%m%d')
        same_date = nm[nm['trade_date'] == str(d_curr)]
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
        same_date = ns[ns['trade_date'] == str(d_curr)]
        if not same_date.empty:
            df = pd.merge(df, same_date[['ts_code', 'news_stock_impact']], on='ts_code', how='left')
            df['news_stock_impact'] = df['news_stock_impact'].fillna(0.0)
        else:
            df['news_stock_impact'] = 0.0
    else:
        df['news_stock_impact'] = 0.0
    return df

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

def apply_selection(all_trades, prob_col='prob', prob_thresh=0.0, top_n=1):
    if prob_thresh > 0:
        filtered = all_trades[all_trades[prob_col] >= prob_thresh].copy()
    else:
        filtered = all_trades.copy()
    daily_groups = filtered.groupby('date_t', sort=True)
    selected = []
    for date_t, group in daily_groups:
        top = group.nlargest(top_n, prob_col)
        selected.append(top)
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected)

def main():
    print("=" * 90, flush=True)
    print("  TS Enhanced Backtest v7 - Minimal Memory", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    print("\n[Step 1] Loading base model...", flush=True)
    doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
    loaded = joblib.load(doubao_model_path)
    base_model = loaded[0] if isinstance(loaded, tuple) else loaded
    print("  Base model loaded", flush=True)

    print("\n[Step 2] Loading TS model...", flush=True)
    ts_model_path = os.path.join(THIS_DIR, 'models', 'ts_model_v3.joblib')
    ts_feat_path = os.path.join(THIS_DIR, 'models', 'ts_feat_cols_v3.joblib')

    if os.path.exists(ts_model_path) and os.path.exists(ts_feat_path):
        ts_model = joblib.load(ts_model_path)
        ts_feat_cols = joblib.load(ts_feat_path)
        print(f"  Loaded TS model ({len(ts_feat_cols)} features)", flush=True)
    else:
        ts_model = None
        ts_feat_cols = []
        print("  No TS model found.", flush=True)

    print("\n[Step 3] Generating trade pool...", flush=True)
    trades_path = os.path.join(THIS_DIR, 'pool_ts_v7.csv')

    if os.path.exists(trades_path):
        print("  Loading existing trade pool...", flush=True)
        trades_df = pd.read_csv(trades_path)
    else:
        test_dates = []
        for idx in range(len(all_dates) - 2):
            d = all_dates[idx]
            if d >= TEST_START and d <= TEST_END:
                test_dates.append((idx, d))

        total = len(test_dates)
        print(f"  Test dates: {total}", flush=True)

        all_picks = []

        for i, (idx, d_t) in enumerate(test_dates):
            try:
                d_t_int = int(d_t)
                d_t1 = all_dates[idx + 1]
                d_t2 = all_dates[idx + 2]

                # Load current day
                current = load_day_data(d_t_int)
                if current is None:
                    continue
                current['date'] = d_t_int

                # Get set of ts_codes for efficient prev loading
                ts_codes_set = set(current['ts_code'].values)

                # Load previous 5 days as dicts (only needed columns)
                prev_data_list = []
                for j in range(1, 6):
                    prev_idx = idx - j
                    if prev_idx < 0:
                        break
                    prev_d = all_dates[prev_idx]
                    prev_dict = load_prev_series(int(prev_d), ts_codes_set)
                    if prev_dict is not None:
                        prev_data_list.append(prev_dict)

                if not prev_data_list:
                    continue

                # Compute TS features
                current = compute_ts_features(current, prev_data_list)
                del prev_data_list
                gc.collect()

                # Add news
                current = add_news_features(current, d_t, news_mkt, news_stk)

                # Base model predictions
                for feat in BASE_FEATS:
                    if feat not in current.columns:
                        current[feat] = 0
                current['base_prob'] = base_model.predict_proba(current[BASE_FEATS].fillna(0))[:, 1]

                # TS model predictions
                if ts_model is not None:
                    for feat in ts_feat_cols:
                        if feat not in current.columns:
                            current[feat] = 0
                    current['ts_prob'] = ts_model.predict_proba(current[ts_feat_cols].fillna(0))[:, 1]
                    current['comb_prob'] = 0.5 * current['base_prob'] + 0.5 * current['ts_prob']
                else:
                    current['ts_prob'] = 0.5
                    current['comb_prob'] = current['base_prob']

                # Get T+1/T+2 prices
                p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
                p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
                if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                    continue

                df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'pre_close'])
                df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
                df_t1 = df_t1.rename(columns={'open': 'open_t1', 'pre_close': 'pre_close_t1'})
                df_t2 = df_t2.rename(columns={'open': 'open_t2', 'high': 'high_t2', 'low': 'low_t2',
                                               'close': 'close_t2', 'pre_close': 'pre_close_t2'})

                merged = pd.merge(current[['ts_code', 'base_prob', 'ts_prob', 'comb_prob']],
                                  df_t1, on='ts_code', how='inner')
                merged = pd.merge(merged, df_t2, on='ts_code', how='inner')

                merged['is_gem'] = merged['ts_code'].apply(is_gem_or_star)
                merged['up_limit'] = np.where(merged['is_gem'],
                                               (merged['pre_close_t1'] * 1.2).round(2),
                                               (merged['pre_close_t1'] * 1.1).round(2))
                valid = merged[~merged['open_t1'].isna() & (merged['open_t1'] < merged['up_limit'])].copy()

                if valid.empty:
                    continue

                batch = valid[['ts_code', 'base_prob', 'ts_prob', 'comb_prob',
                               'open_t1', 'open_t2', 'high_t2', 'close_t2', 'pre_close_t2']].copy()
                batch['date_t'] = d_t
                batch['date_t1'] = d_t1
                batch['date_t2'] = d_t2
                batch = batch.rename(columns={'open_t1': 'buy_price', 'open_t2': 'sell_open',
                                               'high_t2': 'sell_high', 'close_t2': 'sell_close',
                                               'pre_close_t2': 'sell_pre_close'})
                all_picks.append(batch)

                del current, merged, valid, batch
                gc.collect()

                if (i + 1) % 20 == 0:
                    n_trades = sum(len(b) for b in all_picks)
                    print(f"  {i+1}/{total} days, {n_trades} trades", flush=True)

            except Exception as e:
                print(f"  ERROR on {d_t}: {e}", flush=True)
                traceback.print_exc()
                gc.collect()
                continue

        trades_df = pd.concat(all_picks, ignore_index=True)
        trades_df.to_csv(trades_path, index=False)
        print(f"  Trade pool saved: {len(trades_df)} trades", flush=True)

    print(f"  Trade pool: {len(trades_df)} trades", flush=True)
    print(f"  base_prob: mean={trades_df['base_prob'].mean():.4f}, std={trades_df['base_prob'].std():.4f}", flush=True)
    print(f"  ts_prob:   mean={trades_df['ts_prob'].mean():.4f}, std={trades_df['ts_prob'].std():.4f}", flush=True)
    print(f"  comb_prob: mean={trades_df['comb_prob'].mean():.4f}, std={trades_df['comb_prob'].std():.4f}", flush=True)

    print("\n[Step 4] Backtesting...", flush=True)

    schemes = [
        ('Base_Top1_P04',        'base_prob', 0.4, 1, None),
        ('Base_Top1_P05',        'base_prob', 0.5, 1, None),
        ('Base_Top2_P04',        'base_prob', 0.4, 2, None),
        ('Base_Top3_P04',        'base_prob', 0.4, 3, None),
        ('TS_Top1_P04',          'ts_prob',   0.4, 1, None),
        ('TS_Top2_P04',          'ts_prob',   0.4, 2, None),
        ('Comb_Top1_P04',        'comb_prob', 0.4, 1, None),
        ('Comb_Top1_P05',        'comb_prob', 0.5, 1, None),
        ('Comb_Top2_P04',        'comb_prob', 0.4, 2, None),
        ('Comb_Top3_P04',        'comb_prob', 0.4, 3, None),
        ('Base_Top1_P04_TP18',   'base_prob', 0.4, 1, 0.18),
        ('Comb_Top1_P04_TP18',   'comb_prob', 0.4, 1, 0.18),
        ('Base_Top1_P04_TP20',   'base_prob', 0.4, 1, 0.20),
        ('Comb_Top1_P04_TP20',   'comb_prob', 0.4, 1, 0.20),
    ]

    results = {}
    for sname, prob_col, p_thresh, top_n, tp in schemes:
        selected = apply_selection(trades_df, prob_col=prob_col, prob_thresh=p_thresh, top_n=top_n)
        eq, stats = backtest(selected, all_dates_set, take_profit=tp)
        if stats:
            results[sname] = (eq, stats, selected)
            print(f"  {sname:<30} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  "
                  f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}  Trades={stats['trades']:>5d}", flush=True)

    sorted_r = sorted(results.items(), key=lambda x: x[1][1]['sharpe'], reverse=True)

    print("\n[Step 5] Plotting...", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    ax = axes[0]
    for sname, (eq, stats, _) in sorted_r[:6]:
        if not eq.empty:
            ax.plot(eq['date'], eq['nav'], label=f"{sname} (Sharpe={stats['sharpe']:.2f})")
    ax.set_title('Equity Curves - Top 6 Schemes')
    ax.set_xlabel('Date')
    ax.set_ylabel('NAV')
    ax.legend(fontsize=8)
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
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Sharpe Ratio')
    ax2.set_ylabel('Total Return')
    ax.set_title('Scheme Comparison')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(THIS_DIR, 'ts_backtest_v7_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Chart saved", flush=True)

    print(f"\n{'Rank':>4} {'Scheme':<35} {'Total':>10} {'Ann':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'WinRate':>8} {'Trades':>7}")
    print('-' * 110)
    for rank, (sname, (eq, stats, tdf)) in enumerate(sorted_r, 1):
        print(f"{rank:>4} {sname:<35} {stats['total']:>9.2%} {stats['ann']:>9.2%} {stats['sharpe']:>7.2f} "
              f"{stats['mdd']:>9.2%} {stats['calmar']:>7.2f} {stats['win_rate']:>7.2%} {stats['trades']:>7}")

    for sname, (eq, stats, tdf) in sorted_r[:5]:
        if not eq.empty:
            eq.to_csv(os.path.join(THIS_DIR, f'equity_ts_v7_{sname}.csv'), index=False)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
