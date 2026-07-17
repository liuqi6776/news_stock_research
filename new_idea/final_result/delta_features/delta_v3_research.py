"""
Delta Features Strategy v3 - Retrain with Recent Data Only

Key insight: Training data before 2023-08 lacks chip/rank data.
Solution: Only use 2023-08+ data for training delta model.
This gives us ~6 months of training data before test period.

Also: Try using delta features ONLY as a lightweight re-ranker
on top of the base model's top candidates.
"""
import os, sys
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import json
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
FINAL_DIR = os.path.join(os.path.dirname(THIS_DIR))

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']

DELTA_FEATS_ONLY = [
    'delta_winner_rate_1d', 'delta_chip_concentration_1d',
    'delta_cost_50pct_1d', 'delta_weight_avg_1d',
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_accel',
    'delta_turnover_rate_1d', 'delta_volume_ratio_1d',
    'delta_vol_1d', 'delta_amount_1d', 'delta_hot_1d',
    'ma5_dist', 'vol_price_diverge', 'chip_price_diverge',
    'intraday_range', 'upper_shadow', 'lower_shadow',
]

ALL_FEATS = BASE_FEATS + DELTA_FEATS_ONLY

CIRC_MV_LIMIT = 1000000
TRAIN_START = '20230821'
VAL_END = '20240601'
TEST_START = '20240601'
TEST_END = '20260324'

def is_gem_or_star(ts_code):
    return ('300' in ts_code) or ('301' in ts_code) or ('688' in ts_code) or ('689' in ts_code)

def int_to_date(date_int):
    s = str(date_int)
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
        df['news_market_impact'] = same_date['news_market_impact'].mean() if not same_date.empty else 0.0
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

def load_all_features(d_curr, prev_dates, news_mkt, news_stk):
    p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
    p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
    if not os.path.exists(p_chip) or not os.path.exists(p_price) or not os.path.exists(p_other):
        return None

    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount', 'pre_close'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'turnover_rate', 'volume_ratio', 'pe', 'pb', 'circ_mv'])
    rank_df = pd.read_parquet(p_rank) if os.path.exists(p_rank) else pd.DataFrame(columns=['ts_code', 'hot'])

    if len(rank_df) > 0:
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    else:
        rank_df['hot_rank_pct'] = 0.5

    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code') if len(rank_df) > 0 else price_df.copy().assign(hot_rank_pct=0.5)
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate', 'cost_50pct', 'weight_avg']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['circ_mv'] <= CIRC_MV_LIMIT]
    df = add_news_features(df, d_curr, news_mkt, news_stk)

    df['intraday_range'] = (df['high'] - df['low']) / (df['pre_close'] + 1e-8)
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['pre_close'] + 1e-8)
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['pre_close'] + 1e-8)

    for f in DELTA_FEATS_ONLY:
        df[f] = 0.0

    if len(prev_dates) >= 1:
        d_prev = prev_dates[0]
        p_chip_prev = os.path.join(CHIP_DIR, f"{d_prev}.parquet")
        p_price_prev = os.path.join(PRICE_DIR, f"{d_prev}.parquet")
        p_other_prev = os.path.join(OTHER_DIR, f"{d_prev}.parquet")
        p_rank_prev = os.path.join(RANK_DIR, f"{d_prev}.parquet")

        if os.path.exists(p_chip_prev):
            chip_prev = pd.read_parquet(p_chip_prev)
            chip_prev['chip_concentration'] = (chip_prev['cost_85pct'] - chip_prev['cost_15pct']) / (chip_prev['cost_50pct'] + 1e-8)
            mc = pd.merge(chip_df[['ts_code', 'chip_concentration', 'winner_rate', 'cost_50pct', 'weight_avg']],
                          chip_prev[['ts_code', 'chip_concentration', 'winner_rate', 'cost_50pct', 'weight_avg']],
                          on='ts_code', suffixes=('', '_prev'))
            df = pd.merge(df, mc[['ts_code', 'chip_concentration_prev', 'winner_rate_prev', 'cost_50pct_prev', 'weight_avg_prev']], on='ts_code', how='left')
            df['delta_winner_rate_1d'] = (df['winner_rate'] - df['winner_rate_prev']).fillna(0)
            df['delta_chip_concentration_1d'] = (df['chip_concentration'] - df['chip_concentration_prev']).fillna(0)
            df['delta_cost_50pct_1d'] = ((df['cost_50pct'] - df['cost_50pct_prev']) / (df['cost_50pct_prev'] + 1e-8)).fillna(0)
            df['delta_weight_avg_1d'] = ((df['weight_avg'] - df['weight_avg_prev']) / (df['weight_avg_prev'] + 1e-8)).fillna(0)
            df['chip_price_diverge'] = df['delta_cost_50pct_1d'] - (df['pct_chg'] / 100.0 if 'pct_chg' in df.columns else 0)

        if os.path.exists(p_price_prev):
            price_prev = pd.read_parquet(p_price_prev, columns=['ts_code', 'close', 'vol', 'amount'])
            mp = pd.merge(price_df[['ts_code', 'close', 'vol', 'amount']], price_prev, on='ts_code', suffixes=('', '_prev'))
            df = pd.merge(df, mp[['ts_code', 'close_prev', 'vol_prev', 'amount_prev']], on='ts_code', how='left')
            df['ret_1d'] = (df['close'] / (df['close_prev'] + 1e-8) - 1).fillna(0)
            df['delta_vol_1d'] = (df['vol'] / (df['vol_prev'] + 1e-8) - 1).fillna(0)
            df['delta_amount_1d'] = (df['amount'] / (df['amount_prev'] + 1e-8) - 1).fillna(0)
            df['vol_price_diverge'] = df['delta_vol_1d'] - df['ret_1d'].abs()

        if os.path.exists(p_other_prev):
            other_prev = pd.read_parquet(p_other_prev, columns=['ts_code', 'turnover_rate', 'volume_ratio'])
            mo = pd.merge(other_df[['ts_code', 'turnover_rate', 'volume_ratio']], other_prev, on='ts_code', suffixes=('', '_prev'))
            df = pd.merge(df, mo[['ts_code', 'turnover_rate_prev', 'volume_ratio_prev']], on='ts_code', how='left')
            df['delta_turnover_rate_1d'] = (df['turnover_rate'] - df['turnover_rate_prev']).fillna(0)
            df['delta_volume_ratio_1d'] = (df['volume_ratio'] - df['volume_ratio_prev']).fillna(0)

        if os.path.exists(p_rank_prev):
            rank_prev = pd.read_parquet(p_rank_prev)
            if len(rank_prev) > 0 and len(rank_df) > 0:
                mr = pd.merge(rank_df[['ts_code', 'hot']], rank_prev[['ts_code', 'hot']], on='ts_code', suffixes=('', '_prev'))
                df = pd.merge(df, mr[['ts_code', 'hot_prev']], on='ts_code', how='left')
                df['delta_hot_1d'] = (df['hot_rank_pct'] - df['hot_prev'].rank(pct=True)).fillna(0)

    if len(prev_dates) >= 2:
        d3 = prev_dates[min(2, len(prev_dates)-1)]
        p3 = os.path.join(PRICE_DIR, f"{d3}.parquet")
        if os.path.exists(p3):
            c3 = pd.read_parquet(p3, columns=['ts_code', 'close'])
            m3 = pd.merge(price_df[['ts_code', 'close']], c3, on='ts_code', suffixes=('', '_3d'))
            df['ret_3d'] = (df['close'] / (m3['close_3d'] + 1e-8) - 1).fillna(0)

    if len(prev_dates) >= 4:
        d5 = prev_dates[min(4, len(prev_dates)-1)]
        p5 = os.path.join(PRICE_DIR, f"{d5}.parquet")
        if os.path.exists(p5):
            c5 = pd.read_parquet(p5, columns=['ts_code', 'close'])
            m5 = pd.merge(price_df[['ts_code', 'close']], c5, on='ts_code', suffixes=('', '_5d'))
            df['ret_5d'] = (df['close'] / (m5['close_5d'] + 1e-8) - 1).fillna(0)

    if 'ret_1d' in df.columns and 'ret_3d' in df.columns:
        df['ret_accel'] = (df['ret_1d'] - df['ret_3d'] / 3.0).fillna(0)

    if len(prev_dates) >= 4:
        closes = [price_df[['ts_code', 'close']]]
        for i in range(min(4, len(prev_dates))):
            pp = os.path.join(PRICE_DIR, f"{prev_dates[i]}.parquet")
            if os.path.exists(pp):
                closes.append(pd.read_parquet(pp, columns=['ts_code', 'close']))
        if len(closes) >= 3:
            ac = closes[0].rename(columns={'close': 'c_0'})
            for i, c in enumerate(closes[1:], 1):
                ac = pd.merge(ac, c.rename(columns={'close': f'c_{i}'}), on='ts_code', how='outer')
            cc = [f'c_{i}' for i in range(len(closes)) if f'c_{i}' in ac.columns]
            if len(cc) >= 3:
                ac['ma5'] = ac[cc].mean(axis=1)
                ac['ma5_dist'] = (ac['c_0'] / ac['ma5'] - 1).fillna(0)
                df = pd.merge(df, ac[['ts_code', 'ma5_dist']], on='ts_code', how='left')

    for f in ALL_FEATS:
        if f not in df.columns:
            df[f] = 0.0
    return df

def backtest(trades_df, all_dates_set, take_profit=None):
    if trades_df.empty:
        return pd.DataFrame(), {}
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    total_trades = 0
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
            ldp = 0.8 if is_gem_or_star(ts_code) else 0.9
            ld_price = round(sell_pre_close * ldp, 2)
            if sell_high == ld_price:
                dt3 = get_next_trading_day(date_t2, all_dates_set)
                if dt3:
                    pt3 = os.path.join(PRICE_DIR, f"{dt3}.parquet")
                    if os.path.exists(pt3):
                        t3 = pd.read_parquet(pt3, columns=['ts_code', 'open'])
                        r3 = t3[t3['ts_code'] == ts_code]
                        sell_price = r3.iloc[0]['open'] if not r3.empty else sell_close
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
    dr = eq_df['nav'].pct_change()
    mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
    vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    return eq_df, {'total': total_ret, 'sharpe': sharpe, 'mdd': mdd, 'calmar': calmar,
                   'win_rate': (dr > 0).mean(), 'trades': total_trades}

def main():
    print("DELTA FEATURES v3 - Retrain with Recent Data (2023-08+)", flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    # Load base model
    doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
    loaded = joblib.load(doubao_model_path)
    base_model = loaded[0] if isinstance(loaded, tuple) else loaded
    print("Base model loaded", flush=True)

    # Step 1: Train delta model with recent data only
    print("\n--- Training delta model with recent data ---", flush=True)
    delta_model_path = os.path.join(THIS_DIR, 'models', 'delta_recent_model.joblib')

    if os.path.exists(delta_model_path):
        delta_model = joblib.load(delta_model_path)
        print("Loaded existing delta_recent_model", flush=True)
    else:
        X_all, y_all = [], []
        count = 0
        for idx in range(5, len(all_dates) - 2):
            prev_dates = [all_dates[idx - j] for j in range(1, min(6, idx))]
            d_curr = all_dates[idx]
            d_t1 = all_dates[idx + 1]
            d_t2 = all_dates[idx + 2]
            if d_curr < TRAIN_START or d_curr >= TEST_START:
                continue
            df = load_all_features(d_curr, prev_dates, None, None)
            if df is None:
                continue
            pt1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
            pt2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
            if not os.path.exists(pt1) or not os.path.exists(pt2):
                continue
            df_t1 = pd.read_parquet(pt1, columns=['ts_code', 'open'])
            df_t2 = pd.read_parquet(pt2, columns=['ts_code', 'close'])
            m = pd.merge(df_t1, df_t2, on='ts_code', suffixes=('_t1', '_t2'))
            m = pd.merge(df[['ts_code']], m, on='ts_code')
            m['label_ret'] = m['close'] / m['open'] - 1
            m['label'] = (m['label_ret'] > 0.04).astype(int)
            dl = pd.merge(df, m[['ts_code', 'label']], on='ts_code')
            X_all.append(dl[ALL_FEATS].fillna(0).values)
            y_all.append(dl['label'].values)
            count += 1
            if count % 20 == 0:
                print(f"  {count} days loaded", flush=True)

        if X_all:
            X = np.vstack(X_all)
            y = np.concatenate(y_all)
            print(f"  Training: {X.shape[0]} samples, pos_rate={y.mean():.3f}", flush=True)
            delta_model = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                scale_pos_weight=max(1, (1 - y.mean()) / max(y.mean(), 0.01)),
                eval_metric='logloss', verbosity=0, random_state=42
            )
            delta_model.fit(X, y)
            joblib.dump(delta_model, delta_model_path)
            print("  Delta model trained and saved!", flush=True)
        else:
            print("  No training data!", flush=True)
            return

    # Step 2: Generate trade pool for test period
    print("\n--- Generating trade pool ---", flush=True)
    trades_path = os.path.join(THIS_DIR, 'pool_delta_v3_trades.csv')

    if os.path.exists(trades_path):
        trades_df = pd.read_csv(trades_path)
        print(f"Loaded existing pool: {len(trades_df)} trades", flush=True)
    else:
        all_picks = []
        count = 0
        test_dates = [(idx, all_dates[idx]) for idx in range(5, len(all_dates) - 2)
                       if all_dates[idx] >= TEST_START and all_dates[idx] <= TEST_END]
        total = len(test_dates)

        for i, (idx, d_t) in enumerate(test_dates):
            prev_dates = [all_dates[idx - j] for j in range(1, min(6, idx))]
            d_t1 = all_dates[idx + 1]
            d_t2 = all_dates[idx + 2]
            try:
                df = load_all_features(d_t, prev_dates, news_mkt, news_stk)
            except Exception as e:
                print(f"  ERROR {d_t}: {e}", flush=True)
                continue
            if df is None:
                continue

            X_all_feats = df[ALL_FEATS].fillna(0)
            df['prob_base'] = base_model.predict_proba(df[BASE_FEATS].fillna(0))[:, 1]
            df['prob_delta'] = delta_model.predict_proba(X_all_feats)[:, 1]

            # Ensemble: weighted average of base and delta probabilities
            df['prob_ensemble_064'] = 0.6 * df['prob_base'] + 0.4 * df['prob_delta']
            df['prob_ensemble_073'] = 0.7 * df['prob_base'] + 0.3 * df['prob_delta']
            df['prob_ensemble_082'] = 0.8 * df['prob_base'] + 0.2 * df['prob_delta']
            df['prob_ensemble_091'] = 0.9 * df['prob_base'] + 0.1 * df['prob_delta']

            pt1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
            pt2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
            if not os.path.exists(pt1) or not os.path.exists(pt2):
                continue

            df_t1 = pd.read_parquet(pt1, columns=['ts_code', 'open', 'pre_close'])
            df_t2 = pd.read_parquet(pt2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
            df_t1 = df_t1.rename(columns={'open': 'open_t1', 'pre_close': 'pre_close_t1'})
            df_t2 = df_t2.rename(columns={'open': 'open_t2', 'high': 'high_t2', 'low': 'low_t2',
                                           'close': 'close_t2', 'pre_close': 'pre_close_t2'})

            cols = ['ts_code', 'prob_base', 'prob_delta',
                    'prob_ensemble_064', 'prob_ensemble_073', 'prob_ensemble_082', 'prob_ensemble_091',
                    'delta_cost_50pct_1d', 'delta_turnover_rate_1d', 'chip_price_diverge',
                    'delta_winner_rate_1d', 'ret_1d', 'delta_hot_1d']
            merged = pd.merge(df[[c for c in cols if c in df.columns]], df_t1, on='ts_code', how='inner')
            merged = pd.merge(merged, df_t2, on='ts_code', how='inner')

            merged['is_gem'] = merged['ts_code'].str.contains('300|301|688|689', regex=True)
            merged['up_limit'] = np.where(merged['is_gem'],
                                           (merged['pre_close_t1'] * 1.2).round(2),
                                           (merged['pre_close_t1'] * 1.1).round(2))
            valid = merged[~merged['open_t1'].isna() & (merged['open_t1'] < merged['up_limit'])].copy()

            batch = valid.rename(columns={'open_t1': 'buy_price', 'open_t2': 'sell_open',
                                           'high_t2': 'sell_high', 'close_t2': 'sell_close',
                                           'pre_close_t2': 'sell_pre_close'})
            batch['date_t'] = d_t
            batch['date_t1'] = d_t1
            batch['date_t2'] = d_t2
            all_picks.append(batch)

            count += 1
            if count % 20 == 0:
                n = sum(len(b) for b in all_picks)
                print(f"  {count}/{total} days, {n} trades", flush=True)

        trades_df = pd.concat(all_picks, ignore_index=True)
        trades_df.to_csv(trades_path, index=False)
        print(f"Pool saved: {len(trades_df)} trades", flush=True)

    # Step 3: Backtest
    print("\n--- Backtesting ---", flush=True)

    def select_top(trades, score_col, thresh, top_n):
        f = trades[trades[score_col] >= thresh].copy() if thresh > 0 else trades.copy()
        sel = []
        for dt, g in f.groupby('date_t', sort=True):
            sel.append(g.nlargest(top_n, score_col))
        return pd.concat(sel) if sel else pd.DataFrame()

    schemes = [
        ('Base_Top1_P04',          'prob_base',           0.4, 1),
        ('Base_Top1_P03',          'prob_base',           0.3, 1),
        ('Delta_Top1_P04',         'prob_delta',          0.4, 1),
        ('Delta_Top1_P03',         'prob_delta',          0.3, 1),
        ('Ens064_Top1_P04',        'prob_ensemble_064',   0.4, 1),
        ('Ens073_Top1_P04',        'prob_ensemble_073',   0.4, 1),
        ('Ens082_Top1_P04',        'prob_ensemble_082',   0.4, 1),
        ('Ens091_Top1_P04',        'prob_ensemble_091',   0.4, 1),
        ('Ens073_Top1_P03',        'prob_ensemble_073',   0.3, 1),
        ('Ens073_Top2_P04',        'prob_ensemble_073',   0.4, 2),
        ('Base_Top1_P04_full',     'prob_base',           0.4, 1),
    ]

    results = {}
    for sname, score_col, thresh, top_n in schemes:
        sel = select_top(trades_df, score_col, thresh, top_n)
        eq, stats = backtest(sel, all_dates_set)
        if stats:
            results[sname] = (eq, stats, sel)
            print(f"  {sname:<30} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  "
                  f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}  Trades={stats['trades']:>5d}", flush=True)

    # Also test with full period baseline (from enhanced)
    enhanced_pool = os.path.join(FINAL_DIR, 'enhanced', 'pool_base_trades.csv')
    if os.path.exists(enhanced_pool):
        bt = pd.read_csv(enhanced_pool)
        bs = select_top(bt, 'prob', 0.4, 1)
        eq_b, stats_b = backtest(bs, all_dates_set)
        if stats_b:
            results['Enhanced_Base_Full'] = (eq_b, stats_b, bs)
            print(f"  {'Enhanced_Base_Full':<30} Total={stats_b['total']:>9.2%}  Sharpe={stats_b['sharpe']:>6.2f}  "
                  f"MDD={stats_b['mdd']:>8.2%}  Calmar={stats_b['calmar']:>6.2f}  Trades={stats_b['trades']:>5d}", flush=True)

    # Ranking
    sorted_r = sorted(results.items(), key=lambda x: x[1][1]['sharpe'], reverse=True)
    print(f"\nFINAL RANKING:", flush=True)
    for rank, (sname, (eq, stats, _)) in enumerate(sorted_r, 1):
        m = " <-- BEST" if rank == 1 else ""
        print(f"  {rank:>2}. {sname:<30} Sharpe={stats['sharpe']:>6.2f}  Total={stats['total']:>9.2%}  "
              f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}{m}", flush=True)

    # Save best
    best_name, (best_eq, best_stats, best_trades) = sorted_r[0]
    best_eq.to_csv(os.path.join(THIS_DIR, 'best_equity_v3.csv'), index=False)
    best_trades.to_csv(os.path.join(THIS_DIR, 'best_trades_v3.csv'), index=False)

    # Plot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(16, 8))
    for sname, (eq, stats, _) in sorted_r[:8]:
        eq_norm = eq['nav'] / eq['nav'].iloc[0]
        label = f"{sname}: S={stats['sharpe']:.2f} R={stats['total']:.0%}"
        ax.plot(eq['date'], eq_norm, label=label, linewidth=1.5)

    doubao_eq_path = os.path.join(FINAL_DIR, 'doubao', 'equity.csv')
    if os.path.exists(doubao_eq_path):
        deq = pd.read_csv(doubao_eq_path)
        deq['date'] = pd.to_datetime(deq['date'])
        dn = deq['nav'] / deq['nav'].iloc[0]
        ax.plot(deq['date'], dn, label='doubao_result (ref)', linewidth=2.5, color='black', linestyle='--')

    ax.set_title('Delta v3: Ensemble with Recent-Trained Delta Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('NAV (normalized)')
    ax.set_xlabel('Date')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(THIS_DIR, 'delta_v3_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\nChart saved", flush=True)

    for sname, (eq, stats, tdf) in sorted_r:
        eq.to_csv(os.path.join(THIS_DIR, f'equity_v3_{sname}.csv'), index=False)
        tdf.to_csv(os.path.join(THIS_DIR, f'trades_v3_{sname}.csv'), index=False)

    print(f"All saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
