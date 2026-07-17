import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_MAJOR_DIR = r"C:\Users\liuqi\clowspace\toolstock\news_major1"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = THIS_DIR

DOUBAO_MODEL_DIR = os.path.join(FINAL_DIR, 'doubao', 'models')
NEWIDEA_MODEL_DIR_S2 = os.path.join(FINAL_DIR, 'NewIdea_S2', 'models')
NEWIDEA_MODEL_DIR_S3 = os.path.join(FINAL_DIR, 'NewIdea_S3', 'models')

DOUBAO_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
                'news_market_impact', 'news_stock_impact']

NEWIDEA_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
                 'news_market_impact', 'news_stock_impact',
                 'delta_cost_5pct', 'delta_cost_15pct', 'delta_cost_50pct',
                 'delta_cost_85pct', 'delta_cost_95pct', 'delta_winner_rate',
                 'delta_open', 'delta_high', 'delta_low', 'delta_close', 'delta_pct_chg']

CIRC_MV_LIMIT = 1000000

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
    market_records = []
    stock_records = []
    if not os.path.exists(news_dir):
        return pd.DataFrame(market_records), pd.DataFrame(stock_records)
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(news_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        date_str = data.get("article_date", "")
        if not date_str:
            continue
        trade_date = pd.to_datetime(date_str)
        market_impact = data.get("market_impact", 0)
        market_records.append({'trade_date': trade_date, 'news_market_impact': float(market_impact)})
        for s in data.get("stocks", []):
            code = s.get("stock_code")
            if not code:
                continue
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ" if (code.startswith('0') or code.startswith('3')) else code
            stock_records.append({'trade_date': trade_date, 'ts_code': ts_code, 'news_stock_impact': float(s.get("impact", 0))})
    return pd.DataFrame(market_records), pd.DataFrame(stock_records)

def add_news_features(df, d_curr, news_mkt, news_stk):
    if not news_mkt.empty:
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
    if not news_stk.empty:
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

def add_delta_features(df, d_curr, d_prev):
    p_chip_prev = os.path.join(CHIP_DIR, f"{d_prev}.parquet")
    p_price_prev = os.path.join(PRICE_DIR, f"{d_prev}.parquet")
    if not os.path.exists(p_chip_prev) or not os.path.exists(p_price_prev):
        for f in ['delta_cost_5pct', 'delta_cost_15pct', 'delta_cost_50pct',
                   'delta_cost_85pct', 'delta_cost_95pct', 'delta_winner_rate',
                   'delta_open', 'delta_high', 'delta_low', 'delta_close', 'delta_pct_chg']:
            df[f] = 0.0
        return df
    chip_curr_cols = ['ts_code', 'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct', 'winner_rate']
    p_chip_curr = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
    if not os.path.exists(p_chip_curr):
        for f in ['delta_cost_5pct', 'delta_cost_15pct', 'delta_cost_50pct',
                   'delta_cost_85pct', 'delta_cost_95pct', 'delta_winner_rate',
                   'delta_open', 'delta_high', 'delta_low', 'delta_close', 'delta_pct_chg']:
            df[f] = 0.0
        return df
    chip_curr = pd.read_parquet(p_chip_curr, columns=chip_curr_cols)
    chip_prev = pd.read_parquet(p_chip_prev, columns=chip_curr_cols)
    chip_merged = pd.merge(chip_curr, chip_prev, on='ts_code', suffixes=('', '_prev'))
    for col in ['cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct', 'winner_rate']:
        chip_merged[f'delta_{col}'] = chip_merged[col] - chip_merged[f'{col}_prev']
    df = pd.merge(df, chip_merged[['ts_code'] + [f'delta_{c}' for c in ['cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct', 'winner_rate']]], on='ts_code', how='left')
    price_curr_cols = ['ts_code', 'open', 'high', 'low', 'close', 'pct_chg']
    p_price_curr = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
    if not os.path.exists(p_price_curr):
        for f in ['delta_open', 'delta_high', 'delta_low', 'delta_close', 'delta_pct_chg']:
            df[f] = 0.0
        return df
    price_curr = pd.read_parquet(p_price_curr, columns=price_curr_cols)
    price_prev = pd.read_parquet(p_price_prev, columns=price_curr_cols)
    price_merged = pd.merge(price_curr, price_prev, on='ts_code', suffixes=('', '_prev'))
    for col in ['open', 'high', 'low', 'close']:
        price_merged[f'delta_{col}'] = (price_merged[col] - price_merged[f'{col}_prev']) / (price_merged[f'{col}_prev'] + 1e-8)
    price_merged['delta_pct_chg'] = price_merged['pct_chg'] - price_merged['pct_chg_prev']
    df = pd.merge(df, price_merged[['ts_code'] + [f'delta_{c}' for c in ['open', 'high', 'low', 'close', 'pct_chg']]], on='ts_code', how='left')
    delta_cols = [f'delta_{c}' for c in ['cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct', 'winner_rate', 'open', 'high', 'low', 'close', 'pct_chg']]
    for col in delta_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df

def load_base_features(d_curr):
    p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other]):
        return None
    rank_df = pd.read_parquet(p_rank)
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close', 'open', 'high', 'low'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['circ_mv'] <= CIRC_MV_LIMIT]
    return df

def prepare_doubao_training(train_dates, news_mkt, news_stk):
    all_data = []
    for i in tqdm(range(len(train_dates) - 2), desc="doubao train", leave=False):
        d_curr = train_dates[i]
        d_t1 = train_dates[i + 1]
        d_t2 = train_dates[i + 2]
        p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
        if not os.path.exists(p_t1) or not os.path.exists(p_t2):
            continue
        df = load_base_features(d_curr)
        if df is None:
            continue
        df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open'])
        df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'close'])
        label_df = pd.merge(df_t1[['ts_code', 'open']].rename(columns={'open': 't1_open'}),
                           df_t2[['ts_code', 'close']].rename(columns={'close': 't2_close'}),
                           on='ts_code')
        label_df['label_ret'] = label_df['t2_close'] / label_df['t1_open'] - 1
        label_df['label'] = (label_df['label_ret'] > 0.04).astype(int)
        m = pd.merge(df, label_df[['ts_code', 'label']], on='ts_code')
        m['trade_date'] = d_t1
        m = add_news_features(m, d_curr, news_mkt, news_stk)
        all_data.append(m)
    if not all_data:
        return None
    return pd.concat(all_data, ignore_index=True)

def prepare_newidea_training(train_dates, news_mkt, news_stk):
    all_data = []
    for i in tqdm(range(1, len(train_dates) - 2), desc="newidea train", leave=False):
        d_prev = train_dates[i - 1]
        d_curr = train_dates[i]
        d_t1 = train_dates[i + 1]
        d_t2 = train_dates[i + 2]
        p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
        if not os.path.exists(p_t1) or not os.path.exists(p_t2):
            continue
        df = load_base_features(d_curr)
        if df is None:
            continue
        df = add_delta_features(df, d_curr, d_prev)
        df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open'])
        df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'close'])
        label_df = pd.merge(df_t1[['ts_code', 'open']].rename(columns={'open': 't1_open'}),
                           df_t2[['ts_code', 'close']].rename(columns={'close': 't2_close'}),
                           on='ts_code')
        label_df['label_ret'] = label_df['t2_close'] / label_df['t1_open'] - 1
        label_df['label'] = (label_df['label_ret'] > 0.04).astype(int)
        m = pd.merge(df, label_df[['ts_code', 'label']], on='ts_code')
        m['trade_date'] = d_t1
        m = add_news_features(m, d_curr, news_mkt, news_stk)
        all_data.append(m)
    if not all_data:
        return None
    return pd.concat(all_data, ignore_index=True)

def run_backtest(trades_df, take_profit, all_dates_set):
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
            sell_high = trade['sell_high']
            sell_close = trade['sell_close']
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
                        if not t3_row.empty:
                            sell_price = t3_row.iloc[0]['open']
                        else:
                            sell_price = sell_close
                    else:
                        sell_price = sell_close
                else:
                    sell_price = sell_close
            elif sell_high >= buy_price * (1 + take_profit):
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
    return eq_df, {'total': total_ret, 'ann': ann_ret, 'sharpe': sharpe, 'mdd': mdd,
                     'trades': total_trades, 'cannot_sell': cannot_sell_trades, 'final_nav': capital}

def main():
    print("=" * 80)
    print(f"  RETRAINING WITH CIRC_MV_LIMIT={CIRC_MV_LIMIT}")
    print("=" * 80)

    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    dates_2022 = [d for d in dates if d >= '20220101']
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    monthly = {}
    for d in dates_2022:
        ym = d[:6]
        if ym not in monthly:
            monthly[ym] = []
        monthly[ym].append(d)
    yms = sorted(monthly.keys())

    os.makedirs(DOUBAO_MODEL_DIR, exist_ok=True)
    os.makedirs(NEWIDEA_MODEL_DIR_S2, exist_ok=True)
    os.makedirs(NEWIDEA_MODEL_DIR_S3, exist_ok=True)

    all_train_dates = [d for d in dates if '20200101' <= d <= '20260324']

    print("\n" + "=" * 80)
    print("  1. TRAINING DOUBAO MODEL")
    print("=" * 80)
    df_db = prepare_doubao_training(all_train_dates, news_mkt, news_stk)
    X_db = df_db[DOUBAO_FEATS].fillna(0)
    y_db = df_db['label']
    print(f"  Samples: {len(X_db)}, Positive: {y_db.sum()} ({y_db.mean():.2%})")

    db_model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  eval_metric='auc', n_jobs=-1, tree_method='hist')
    db_model.fit(X_db, y_db)
    joblib.dump((db_model, DOUBAO_FEATS), os.path.join(DOUBAO_MODEL_DIR, 'doubao_t1t2_model.joblib'))
    imp = pd.DataFrame({'f': DOUBAO_FEATS, 'imp': db_model.feature_importances_}).sort_values('imp', ascending=False)
    print("  Feature importance:")
    for _, row in imp.iterrows():
        print(f"    {row['f']}: {row['imp']:.4f}")

    print("\n" + "=" * 80)
    print("  2. TRAINING NEWIDEA ROLLING MODELS")
    print("=" * 80)

    for i in range(len(yms)):
        train_ym = yms[i]
        if train_ym < '202301':
            continue
        if i + 1 >= len(yms):
            break
        test_ym = yms[i + 1]

        train_dates = []
        for j in range(i + 1):
            train_dates.extend(monthly[yms[j]])

        df_ni = prepare_newidea_training(train_dates, news_mkt, news_stk)
        if df_ni is None or len(df_ni) == 0:
            print(f"\n  Train: ...-{train_ym} | Predict: {test_ym}")
            print(f"    SKIP: No training data")
            continue

        X_ni = df_ni[NEWIDEA_FEATS].fillna(0)
        y_ni = df_ni['label']

        print(f"\n  Train: 202201-{train_ym} | Predict: {test_ym}")
        print(f"    Samples: {len(X_ni)}, Positive: {y_ni.sum()} ({y_ni.mean():.2%})")

        ni_model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42,
                                      eval_metric='auc', n_jobs=-1, tree_method='hist')
        ni_model.fit(X_ni, y_ni)

        joblib.dump((ni_model, NEWIDEA_FEATS), os.path.join(NEWIDEA_MODEL_DIR_S2, f'model_{test_ym}.joblib'))
        joblib.dump((ni_model, NEWIDEA_FEATS), os.path.join(NEWIDEA_MODEL_DIR_S3, f'model_{test_ym}.joblib'))
        print(f"    Saved model_{test_ym}")

    print("\n  Done training")

    print("\n" + "=" * 80)
    print("  3. BACKTESTING")
    print("=" * 80)

    all_test_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_test_dates)

    strategies = {}

    db_data = joblib.load(os.path.join(DOUBAO_MODEL_DIR, 'doubao_t1t2_model.joblib'))

    def load_d(d_t, d_prev):
        df = load_base_features(d_t)
        if df is None:
            return None
        df = add_news_features(df, d_t, news_mkt, news_stk)
        for f in DOUBAO_FEATS:
            if f not in df.columns:
                df[f] = 0.0
        return db_data[0], db_data[1], df

    def load_ni(d_t, d_prev):
        month_key = d_t[:6]
        mp = os.path.join(NEWIDEA_MODEL_DIR_S2, f'model_{month_key}.joblib')
        if not os.path.exists(mp):
            return None
        model, feats = joblib.load(mp)
        df = load_base_features(d_t)
        if df is None:
            return None
        df = add_delta_features(df, d_t, d_prev)
        df = add_news_features(df, d_t, news_mkt, news_stk)
        for f in NEWIDEA_FEATS:
            if f not in df.columns:
                df[f] = 0.0
        return model, feats, df

    for sname, loader_fn, sel_type in [('doubao_result', load_d, 's1'), ('NewIdea S2', load_ni, 's2'), ('NewIdea S3', load_ni, 's3')]:
        print(f"\n  Backtesting {sname}...")
        trades = []
        for idx in tqdm(range(1, len(all_test_dates) - 2), desc=f"  {sname}", leave=False):
            d_prev = all_test_dates[idx - 1]
            d_t = all_test_dates[idx]
            d_t1 = all_test_dates[idx + 1]
            d_t2 = all_test_dates[idx + 2]
            if d_t < '20230101' or d_t > '20260324':
                continue
            result = loader_fn(d_t, d_prev)
            if result is None:
                continue
            model, feats, df = result
            X = df[feats].fillna(0)
            if len(X) == 0:
                continue
            df['prob'] = model.predict_proba(X)[:, 1]

            if sel_type == 's1':
                picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
                if picks.empty:
                    picks = df.sort_values('prob', ascending=False).head(1)
            elif sel_type == 's2':
                picks = df.sort_values('prob', ascending=False).head(3)
            elif sel_type == 's3':
                df['wr_norm'] = df['winner_rate'] / 100.0
                df['cc_norm'] = df['chip_concentration'].clip(0, 5) / 5.0
                df['score'] = df['prob'] * 0.7 + df['wr_norm'] * 0.2 + (1 - df['cc_norm']) * 0.1
                picks = df.sort_values('score', ascending=False).head(3)

            if picks.empty:
                continue

            p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
            p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                continue
            df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'pre_close'])
            df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])

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
                trades.append({
                    'date_t': d_t, 'date_t1': d_t1, 'date_t2': d_t2,
                    'ts_code': ts_code, 'buy_price': t1['open'],
                    'sell_open': t2['open'], 'sell_high': t2['high'],
                    'sell_close': t2['close'], 'sell_pre_close': t2['pre_close'],
                    'prob': pick.get('prob', 0),
                })

        if not trades:
            print(f"    No trades!")
            strategies[sname] = (pd.DataFrame(), {})
            continue

        td = pd.DataFrame(trades)
        eq, stats = run_backtest(td, 0.08, all_dates_set)
        strategies[sname] = (eq, stats)
        print(f"    Trades: {stats['trades']}, Return: {stats['total']:.2%}, Sharpe: {stats['sharpe']:.2f}, MDD: {stats['mdd']:.2%}")

        sub = {'doubao_result': 'doubao', 'NewIdea S2': 'NewIdea_S2', 'NewIdea S3': 'NewIdea_S3'}[sname]
        if len(eq) > 0:
            eq.to_csv(os.path.join(FINAL_DIR, sub, 'equity.csv'), index=False)
            td.to_csv(os.path.join(FINAL_DIR, sub, 'trades.csv'), index=False)

    print("\n" + "=" * 80)
    print("  4. PLOTTING")
    print("=" * 80)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = {'doubao_result': '#1f77b4', 'NewIdea S2': '#2ca02c', 'NewIdea S3': '#d62728'}
    for name, (eq, stats) in strategies.items():
        if len(eq) > 0:
            eq_norm = eq['nav'] / eq['nav'].iloc[0]
            label = f"{name}: {stats['total']:.1%} (Sharpe={stats['sharpe']:.2f})"
            ax.plot(eq['date'], eq_norm, label=label, linewidth=2, color=colors[name])
    ax.set_title(f'Strategy Comparison (circ_mv<={CIRC_MV_LIMIT/10000:.0f}亿, Label: T+2close/T+1open>4%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Date'); ax.set_ylabel('NAV (normalized)')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_DIR, 'strategy_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Chart saved")

    print("\n" + "=" * 80)
    print("  FINAL RESULTS")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Total':>10} {'Annual':>10} {'Sharpe':>8} {'MDD':>10} {'Trades':>8}")
    print("-" * 70)
    for name, (eq, stats) in strategies.items():
        if stats:
            print(f"{name:<20} {stats['total']:>9.2%} {stats['ann']:>9.2%} {stats['sharpe']:>7.2f} {stats['mdd']:>9.2%} {stats['trades']:>7d}")

if __name__ == "__main__":
    main()
