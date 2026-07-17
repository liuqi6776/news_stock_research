"""
TS Enhanced Backtest v4 - Day-by-day loading (same strategy as run_backtest.py).
Memory efficient: only loads 30-day lookback window per test date.

Strategy:
1. Train TS model using sampled training dates (already done in v3)
2. For each test date, load 30-day lookback, compute vectorized TS features
3. Use base model + TS model for predictions
4. Backtest multiple schemes and compare
"""
import os, sys, json, gc
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
LOOKBACK = 30

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

def compute_ts_features_vectorized(panel):
    """Compute TS features on a small panel (30 days x ~2000 stocks = ~60k rows)."""
    df = panel.copy()
    df = df.sort_values(['ts_code', 'date']).reset_index(drop=True)

    df['ret_1d'] = df.groupby('ts_code')['close'].pct_change(1)
    df['ret_2d'] = df.groupby('ts_code')['close'].pct_change(2)
    df['ret_5d'] = df.groupby('ts_code')['close'].pct_change(5)
    df['ret_10d'] = df.groupby('ts_code')['close'].pct_change(10)
    df['ret_20d'] = df.groupby('ts_code')['close'].pct_change(20)

    df['log_ret_1d'] = np.log(df['close'] / df.groupby('ts_code')['close'].shift(1).clip(lower=0.01))
    df['log_ret_2d'] = np.log(df['close'] / df.groupby('ts_code')['close'].shift(2).clip(lower=0.01))

    for window in [5, 10, 20]:
        for col in ['close', 'vol', 'turnover_rate', 'chip_concentration', 'winner_rate']:
            if col not in df.columns:
                continue
            grp = df.groupby('ts_code')[col]
            df[f'{col}_mean_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).mean())
            df[f'{col}_std_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).std())

    for window in [5, 10, 20]:
        grp = df.groupby('ts_code')['close']
        df[f'close_min_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).min())
        df[f'close_max_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).max())

    for window in [5, 10, 20]:
        mean_col = f'close_mean_{window}'
        if mean_col in df.columns:
            df[f'ma{window}_dist'] = (df['close'] / df[mean_col].clip(lower=0.01)) - 1

    if 'vol_mean_5' in df.columns:
        df['vol_ratio_5d'] = df['vol'] / (df['vol_mean_5'].clip(lower=1))

    delta = df.groupby('ts_code')['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.groupby(df['ts_code']).transform(lambda x: x.rolling(14, min_periods=10).mean())
    avg_loss = loss.groupby(df['ts_code']).transform(lambda x: x.rolling(14, min_periods=10).mean())
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    if 'close_mean_20' in df.columns and 'close_std_20' in df.columns:
        df['boll_upper'] = df['close_mean_20'] + 2 * df['close_std_20']
        df['boll_lower'] = df['close_mean_20'] - 2 * df['close_std_20']
        df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'] + 1e-8)

    df['chip_conc_change_1d'] = df.groupby('ts_code')['chip_concentration'].diff(1)
    df['winner_rate_change_1d'] = df.groupby('ts_code')['winner_rate'].diff(1)

    if 'hot_rank_pct' in df.columns:
        for window in [5, 10]:
            df[f'hot_rank_pct_std_{window}'] = df.groupby('ts_code')['hot_rank_pct'].transform(
                lambda x: x.rolling(window, min_periods=3).std())

    df['day_of_week'] = df['date'].apply(lambda x: int_to_date(int(x)).weekday()).astype(np.float32)
    df['month'] = df['date'].apply(lambda x: int_to_date(int(x)).month).astype(np.float32)
    df['is_month_start'] = df['date'].apply(lambda x: 1.0 if int_to_date(int(x)).day <= 5 else 0.0)
    df['is_month_end'] = df['date'].apply(lambda x: 1.0 if int_to_date(int(x)).day >= 25 else 0.0)

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
    print("  TS Enhanced Backtest v4 - Day-by-Day Loading", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    # Load base model
    print("\n[Step 1] Loading base model...", flush=True)
    doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
    loaded = joblib.load(doubao_model_path)
    base_model = loaded[0] if isinstance(loaded, tuple) else loaded
    print("  Base model loaded", flush=True)

    # Load or train TS model
    print("\n[Step 2] Loading TS model...", flush=True)
    ts_model_path = os.path.join(THIS_DIR, 'models', 'ts_model_v3.joblib')
    ts_feat_path = os.path.join(THIS_DIR, 'models', 'ts_feat_cols_v3.joblib')
    os.makedirs(os.path.join(THIS_DIR, 'models'), exist_ok=True)

    if os.path.exists(ts_model_path) and os.path.exists(ts_feat_path):
        ts_model = joblib.load(ts_model_path)
        ts_feat_cols = joblib.load(ts_feat_path)
        print(f"  Loaded TS model ({len(ts_feat_cols)} features)", flush=True)
    else:
        print("  No existing TS model. Training new one...", flush=True)
        train_dates = [d for d in all_dates if int(TRAIN_START) <= int(d) < int(TEST_START)]
        sample_dates = train_dates[::3]
        print(f"  Training on {len(sample_dates)} dates", flush=True)

        train_records = []
        for i, d in enumerate(sample_dates):
            d_int = int(d)
            d_idx = all_dates.index(d)
            if d_idx < LOOKBACK:
                continue

            lookback_dates = all_dates[max(0, d_idx - LOOKBACK):d_idx + 1]
            records = []
            for ld in lookback_dates:
                day_data = load_day_data(int(ld))
                if day_data is not None:
                    day_data['date'] = int(ld)
                    records.append(day_data)

            if not records:
                continue
            panel = pd.concat(records, ignore_index=True)
            panel = compute_ts_features_vectorized(panel)

            # Add labels
            dt = int_to_date(d_int)
            t1, t2 = None, None
            for j in range(1, 10):
                nd = int((dt + timedelta(days=j)).strftime('%Y%m%d'))
                if nd in all_dates_set:
                    if t1 is None:
                        t1 = nd
                    elif t2 is None:
                        t2 = nd
                        break

            if t1 is None or t2 is None:
                continue

            p1 = os.path.join(PRICE_DIR, f"{t1}.parquet")
            p2 = os.path.join(PRICE_DIR, f"{t2}.parquet")
            if not os.path.exists(p1) or not os.path.exists(p2):
                continue

            price_t1 = pd.read_parquet(p1, columns=['ts_code', 'open'])
            price_t2 = pd.read_parquet(p2, columns=['ts_code', 'close'])
            labels = pd.merge(price_t1, price_t2, on='ts_code', how='inner')
            labels['label_ret'] = labels['close'] / labels['open'] - 1
            labels['label'] = (labels['label_ret'] > 0.04).astype(int)

            day_panel = panel[panel['date'] == d_int].copy()
            day_panel = pd.merge(day_panel, labels[['ts_code', 'label']], on='ts_code', how='left')
            day_panel = day_panel.dropna(subset=['label'])
            train_records.append(day_panel)

            del panel
            gc.collect()

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(sample_dates)} dates, {sum(len(r) for r in train_records)} samples", flush=True)

        train_df = pd.concat(train_records, ignore_index=True)
        print(f"  Training data: {len(train_df)} rows, pos_rate={train_df['label'].mean():.3f}", flush=True)

        exclude = {'ts_code', 'date', 'open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount',
                   'pre_close', 'circ_mv', 'turnover_rate', 'volume_ratio',
                   'cost_50pct', 'weight_avg', 'cost_15pct', 'cost_85pct',
                   'chip_concentration', 'winner_rate', 'hot_rank_pct', 'label', 'label_ret',
                   'news_market_impact', 'news_stock_impact'}
        ts_feat_cols = [c for c in train_df.columns if c not in exclude
                        and train_df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int8']]

        X = train_df[ts_feat_cols].fillna(0)
        y = train_df['label'].astype(int)
        pos_rate = y.mean()
        scale_pos = max(1, (1 - pos_rate) / max(pos_rate, 0.01))

        ts_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            scale_pos_weight=scale_pos,
            eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1
        )
        ts_model.fit(X, y)

        # Feature selection
        imp = pd.Series(ts_model.feature_importances_, index=ts_feat_cols).sort_values(ascending=False)
        selected = imp[imp > 0.005].index.tolist()
        if len(selected) > 30:
            corr = train_df[selected].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
            selected = [f for f in selected if f not in to_drop]

        ts_feat_cols = selected
        print(f"  Selected {len(ts_feat_cols)} features", flush=True)

        # Retrain
        X_sel = train_df[ts_feat_cols].fillna(0)
        ts_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            scale_pos_weight=scale_pos,
            eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1
        )
        ts_model.fit(X_sel, y)

        joblib.dump(ts_model, ts_model_path)
        joblib.dump(ts_feat_cols, ts_feat_path)
        print("  TS model trained and saved!", flush=True)

        del train_df
        gc.collect()

    # Phase 2: Generate trade pool
    print("\n[Step 3] Generating trade pool...", flush=True)
    trades_path = os.path.join(THIS_DIR, 'pool_ts_v4.csv')

    if os.path.exists(trades_path):
        print("  Loading existing trade pool...", flush=True)
        trades_df = pd.read_csv(trades_path)
    else:
        test_dates = [(idx, all_dates[idx]) for idx in range(LOOKBACK, len(all_dates) - 2)
                       if all_dates[idx] >= TEST_START and all_dates[idx] <= TEST_END]
        total = len(test_dates)
        print(f"  Test dates: {total}", flush=True)

        all_picks = []
        for i, (idx, d_t) in enumerate(test_dates):
            d_t_int = int(d_t)
            d_t1 = all_dates[idx + 1]
            d_t2 = all_dates[idx + 2]

            # Load lookback panel
            lookback_dates = all_dates[max(0, idx - LOOKBACK):idx + 1]
            records = []
            for ld in lookback_dates:
                day_data = load_day_data(int(ld))
                if day_data is not None:
                    day_data['date'] = int(ld)
                    records.append(day_data)

            if not records:
                continue

            panel = pd.concat(records, ignore_index=True)
            panel = compute_ts_features_vectorized(panel)

            # Get current day data
            current = panel[panel['date'] == d_t_int].copy()
            if current.empty:
                del panel
                continue

            # Add news features
            current = add_news_features(current, d_t, news_mkt, news_stk)

            # Base model predictions
            for feat in BASE_FEATS:
                if feat not in current.columns:
                    current[feat] = 0
            current = current.fillna({feat: 0 for feat in BASE_FEATS})
            current['base_prob'] = base_model.predict_proba(current[BASE_FEATS])[:, 1]

            # TS model predictions
            for feat in ts_feat_cols:
                if feat not in current.columns:
                    current[feat] = 0
            current['ts_prob'] = ts_model.predict_proba(current[ts_feat_cols].fillna(0))[:, 1]

            # Combined
            current['comb_prob'] = 0.5 * current['base_prob'] + 0.5 * current['ts_prob']

            # Get T+1/T+2 prices
            p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
            p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                del panel
                continue

            df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'pre_close'])
            df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
            df_t1 = df_t1.rename(columns={'open': 'open_t1', 'pre_close': 'pre_close_t1'})
            df_t2 = df_t2.rename(columns={'open': 'open_t2', 'high': 'high_t2', 'low': 'low_t2',
                                           'close': 'close_t2', 'pre_close': 'pre_close_t2'})

            merged = pd.merge(current[['ts_code', 'base_prob', 'ts_prob', 'comb_prob']],
                              df_t1, on='ts_code', how='inner')
            merged = pd.merge(merged, df_t2, on='ts_code', how='inner')

            # Filter: can't buy at limit-up
            merged['is_gem'] = merged['ts_code'].apply(is_gem_or_star)
            merged['up_limit'] = np.where(merged['is_gem'],
                                           (merged['pre_close_t1'] * 1.2).round(2),
                                           (merged['pre_close_t1'] * 1.1).round(2))
            valid = merged[~merged['open_t1'].isna() & (merged['open_t1'] < merged['up_limit'])].copy()

            if valid.empty:
                del panel
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

            del panel
            gc.collect()

            if (i + 1) % 20 == 0:
                n_trades = sum(len(b) for b in all_picks)
                print(f"  {i+1}/{total} days, {n_trades} trades", flush=True)

        trades_df = pd.concat(all_picks, ignore_index=True)
        trades_df.to_csv(trades_path, index=False)
        print(f"  Trade pool saved: {len(trades_df)} trades", flush=True)

    print(f"  Trade pool: {len(trades_df)} trades", flush=True)
    print(f"  base_prob: mean={trades_df['base_prob'].mean():.4f}, std={trades_df['base_prob'].std():.4f}", flush=True)
    print(f"  ts_prob:   mean={trades_df['ts_prob'].mean():.4f}, std={trades_df['ts_prob'].std():.4f}", flush=True)
    print(f"  comb_prob: mean={trades_df['comb_prob'].mean():.4f}, std={trades_df['comb_prob'].std():.4f}", flush=True)

    # Phase 3: Backtest
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

    # Sort by Sharpe
    sorted_r = sorted(results.items(), key=lambda x: x[1][1]['sharpe'], reverse=True)

    # Plot
    print("\n[Step 5] Plotting...", flush=True)
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
    plt.savefig(os.path.join(THIS_DIR, 'ts_backtest_v4_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Chart saved", flush=True)

    # Print final ranking
    print(f"\n{'Rank':>4} {'Scheme':<35} {'Total':>10} {'Ann':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'WinRate':>8} {'Trades':>7}")
    print('-' * 110)
    for rank, (sname, (eq, stats, tdf)) in enumerate(sorted_r, 1):
        print(f"{rank:>4} {sname:<35} {stats['total']:>9.2%} {stats['ann']:>9.2%} {stats['sharpe']:>7.2f} "
              f"{stats['mdd']:>9.2%} {stats['calmar']:>7.2f} {stats['win_rate']:>7.2%} {stats['trades']:>7}")

    # Save equity curves
    for sname, (eq, stats, tdf) in sorted_r[:5]:
        if not eq.empty:
            eq.to_csv(os.path.join(THIS_DIR, f'equity_ts_v4_{sname}.csv'), index=False)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
