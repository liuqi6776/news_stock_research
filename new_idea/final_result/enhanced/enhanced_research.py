"""
Enhanced Strategy Research v2 - Efficient Multi-Scheme Optimization
Step 1: Generate ALL trades with base model (once)
Step 2: Apply different selection/TP/sizing schemes on the same trade pool
Step 3: Compare all schemes
"""
import os
import sys
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
MODEL_DIR = os.path.join(THIS_DIR, 'models')
FINAL_DIR = os.path.join(os.path.dirname(THIS_DIR))

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']

EXTENDED_FEATS = BASE_FEATS + [
    'turnover_rate', 'volume_ratio', 'pe', 'pb',
    'momentum_1d', 'volatility_proxy', 'intraday_range',
    'upper_shadow', 'lower_shadow', 'vol_change', 'amount_change',
    'chip_width', 'chip_skew',
]

CIRC_MV_LIMIT = 1000000
TRAIN_START = '20200101'
TEST_START = '20230101'
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

def load_features(d_curr, d_prev, news_mkt, news_stk, extended=False):
    p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
    if not os.path.exists(p_chip) or not os.path.exists(p_price) or not os.path.exists(p_other):
        return None

    rank_df = pd.read_parquet(p_rank) if os.path.exists(p_rank) else pd.DataFrame(columns=['ts_code', 'hot'])
    if len(rank_df) > 0:
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    else:
        rank_df['hot_rank_pct'] = 0.5
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount', 'pre_close'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'turnover_rate', 'volume_ratio', 'pe', 'pb', 'circ_mv'])

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

    if extended:
        df['chip_width'] = (chip_df['cost_95pct'] - chip_df['cost_5pct']) / (chip_df['cost_50pct'] + 1e-8)
        chip_skew = (chip_df['cost_85pct'] - chip_df['cost_50pct']) - (chip_df['cost_50pct'] - chip_df['cost_15pct'])
        df['chip_skew'] = chip_skew / (chip_df['cost_50pct'] + 1e-8)
        df['intraday_range'] = (df['high'] - df['low']) / (df['pre_close'] + 1e-8)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['pre_close'] + 1e-8)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['pre_close'] + 1e-8)

        if d_prev and os.path.exists(os.path.join(PRICE_DIR, f"{d_prev}.parquet")):
            price_prev = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_prev}.parquet"), columns=['ts_code', 'close', 'vol', 'amount'])
            merged = pd.merge(price_df[['ts_code', 'close', 'vol', 'amount']], price_prev, on='ts_code', suffixes=('', '_prev'))
            df['momentum_1d'] = (merged['close'] / (merged['close_prev'] + 1e-8) - 1).fillna(0)
            df['vol_change'] = (merged['vol'] / (merged['vol_prev'] + 1e-8) - 1).fillna(0)
            df['amount_change'] = (merged['amount'] / (merged['amount_prev'] + 1e-8) - 1).fillna(0)
            pct_prev = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_prev}.parquet"), columns=['ts_code', 'pct_chg'])
            merged2 = pd.merge(price_df[['ts_code', 'pct_chg']], pct_prev, on='ts_code', suffixes=('', '_prev'))
            df['volatility_proxy'] = (merged2['pct_chg'] - merged2['pct_chg_prev']).abs().fillna(0)

        for f in EXTENDED_FEATS:
            if f not in df.columns:
                df[f] = 0.0

    for f in BASE_FEATS:
        if f not in df.columns:
            df[f] = 0.0

    return df

def train_model_fast(all_dates, feats, label_thresh=0.04):
    print(f"  Training model with {len(feats)} features...", flush=True)
    X_all, y_all = [], []
    extended = (len(feats) > len(BASE_FEATS))
    count = 0
    for idx in range(2, len(all_dates) - 2):
        d_prev = all_dates[idx - 1]
        d_curr = all_dates[idx]
        d_t1 = all_dates[idx + 1]
        d_t2 = all_dates[idx + 2]
        if d_curr < TRAIN_START:
            continue
        if d_curr >= TEST_START:
            break

        df = load_features(d_curr, d_prev, None, None, extended=extended)
        if df is None:
            continue

        p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
        if not os.path.exists(p_t1) or not os.path.exists(p_t2):
            continue

        df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open'])
        df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'close'])
        merged = pd.merge(df_t1, df_t2, on='ts_code', suffixes=('_t1', '_t2'))
        merged = pd.merge(df[['ts_code']], merged, on='ts_code')
        merged['label_ret'] = merged['close'] / merged['open'] - 1
        merged['label'] = (merged['label_ret'] > label_thresh).astype(int)

        df_labeled = pd.merge(df, merged[['ts_code', 'label']], on='ts_code')
        X_all.append(df_labeled[feats].fillna(0).values)
        y_all.append(df_labeled['label'].values)
        count += 1
        if count % 100 == 0:
            print(f"    ...{count} days loaded", flush=True)

    if not X_all:
        return None
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f"  Training data: {X.shape[0]} samples, positive rate: {y.mean():.3f}", flush=True)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=max(1, (1 - y.mean()) / max(y.mean(), 0.01)),
        eval_metric='logloss', verbosity=0, random_state=42
    )
    model.fit(X, y)
    print(f"  Model trained!", flush=True)
    return model

def generate_all_trades(all_dates, all_dates_set, news_mkt, news_stk, model, feats, extended):
    print(f"  Generating trades...", flush=True)
    all_picks = []
    count = 0
    for idx in range(2, len(all_dates) - 2):
        d_prev = all_dates[idx - 1]
        d_t = all_dates[idx]
        d_t1 = all_dates[idx + 1]
        d_t2 = all_dates[idx + 2]
        if d_t < TEST_START or d_t > TEST_END:
            continue

        df = load_features(d_t, d_prev, news_mkt, news_stk, extended=extended)
        if df is None:
            continue

        X = df[feats].fillna(0)
        if len(X) == 0:
            continue
        df['prob'] = model.predict_proba(X)[:, 1]

        p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
        if not os.path.exists(p_t1) or not os.path.exists(p_t2):
            continue

        df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'pre_close'])
        df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])

        for _, row in df.iterrows():
            ts_code = row['ts_code']
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
            all_picks.append({
                'date_t': d_t, 'date_t1': d_t1, 'date_t2': d_t2,
                'ts_code': ts_code, 'buy_price': t1['open'],
                'sell_open': t2['open'], 'sell_high': t2['high'],
                'sell_close': t2['close'], 'sell_pre_close': t2['pre_close'],
                'prob': row['prob'],
                'hot_rank_pct': row.get('hot_rank_pct', 0),
                'chip_concentration': row.get('chip_concentration', 0),
                'winner_rate': row.get('winner_rate', 0),
                'turnover_rate': row.get('turnover_rate', 0),
                'volume_ratio': row.get('volume_ratio', 0),
                'pct_chg': row.get('pct_chg', 0),
            })

        count += 1
        if count % 50 == 0:
            print(f"    ...{count} days processed, {len(all_picks)} trades", flush=True)

    print(f"  Total: {len(all_picks)} trades from {count} days", flush=True)
    return pd.DataFrame(all_picks)

def apply_selection(all_trades, prob_thresh=0.0, top_n=3):
    if prob_thresh > 0:
        filtered = all_trades[all_trades['prob'] >= prob_thresh].copy()
    else:
        filtered = all_trades.copy()

    daily_groups = filtered.groupby('date_t', sort=True)
    selected = []
    for date_t, group in daily_groups:
        top = group.nlargest(top_n, 'prob')
        selected.append(top)
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected)

def backtest_scheme(trades_df, all_dates_set, take_profit=None, position_sizing='equal'):
    if trades_df.empty:
        return pd.DataFrame(), {}
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    total_trades = 0
    cannot_sell_trades = 0

    for date_t2, group in trades_df.groupby('date_t2', sort=True):
        if position_sizing == 'prob_weighted':
            weights = group['prob'].values
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(group)) / len(group)

        day_pnl = 0.0
        for i, (_, trade) in enumerate(group.iterrows()):
            total_trades += 1
            ts_code = trade['ts_code']
            buy_price = trade['buy_price']
            sell_close = trade['sell_close']
            sell_high = trade['sell_high']
            sell_pre_close = trade['sell_pre_close']
            alloc = capital * weights[i]

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
    print("  ENHANCED STRATEGY RESEARCH v2 - Multi-Scheme Optimization", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    # Step 1: Train models
    print("\n[Step 1] Training models...", flush=True)

    doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
    base_model_path = os.path.join(MODEL_DIR, 'enhanced_base_model.joblib')
    ext_model_path = os.path.join(MODEL_DIR, 'enhanced_ext_model.joblib')

    if os.path.exists(doubao_model_path):
        print("  Loading doubao_result base model...", flush=True)
        loaded = joblib.load(doubao_model_path)
        base_model = loaded[0] if isinstance(loaded, tuple) else loaded
    elif os.path.exists(base_model_path):
        print("  Loading existing base model...", flush=True)
        loaded = joblib.load(base_model_path)
        base_model = loaded[0] if isinstance(loaded, tuple) else loaded
    else:
        print("  Training base model (5 features)...", flush=True)
        base_model = train_model_fast(all_dates, BASE_FEATS)
        if base_model is None:
            print("  Failed!", flush=True)
            return
        joblib.dump(base_model, base_model_path)

    if os.path.exists(ext_model_path):
        print("  Loading existing extended model...", flush=True)
        loaded = joblib.load(ext_model_path)
        ext_model = loaded[0] if isinstance(loaded, tuple) else loaded
    else:
        print("  Training extended model (19 features)...", flush=True)
        ext_model = train_model_fast(all_dates, EXTENDED_FEATS)
        if ext_model is not None:
            joblib.dump(ext_model, ext_model_path)
            print(f"  Extended model saved.", flush=True)
        else:
            print("  Extended model failed, using base model", flush=True)
            ext_model = base_model

    # Step 2: Generate trade pools
    print("\n[Step 2] Generating trade pools...", flush=True)

    base_trades_path = os.path.join(THIS_DIR, 'pool_base_trades.csv')
    ext_trades_path = os.path.join(THIS_DIR, 'pool_ext_trades.csv')

    if os.path.exists(base_trades_path):
        print("  Loading existing base trade pool...", flush=True)
        base_trades = pd.read_csv(base_trades_path)
    else:
        base_trades = generate_all_trades(all_dates, all_dates_set, news_mkt, news_stk, base_model, BASE_FEATS, False)
        base_trades.to_csv(base_trades_path, index=False)

    if os.path.exists(ext_trades_path):
        print("  Loading existing extended trade pool...", flush=True)
        ext_trades = pd.read_csv(ext_trades_path)
    else:
        ext_trades = generate_all_trades(all_dates, all_dates_set, news_mkt, news_stk, ext_model, EXTENDED_FEATS, True)
        ext_trades.to_csv(ext_trades_path, index=False)

    print(f"  Base pool: {len(base_trades)} trades", flush=True)
    print(f"  Extended pool: {len(ext_trades)} trades", flush=True)

    # Step 3: Run all schemes
    print(f"\n{'='*90}", flush=True)
    print(f"  RUNNING ALL SCHEMES", flush=True)
    print(f"{'='*90}", flush=True)

    schemes = [
        # (name, pool, prob_thresh, top_n, take_profit, position_sizing)
        ('S0_Baseline',            base_trades, 0.0, 3, None,  'equal'),
        ('S0_Baseline_P08',        base_trades, 0.8, 3, None,  'equal'),
        ('S1_ExtFeats',            ext_trades,  0.0, 3, None,  'equal'),
        ('S1_ExtFeats_P04',        ext_trades,  0.4, 3, None,  'equal'),
        ('S2_TP15',                base_trades, 0.0, 3, 0.15,  'equal'),
        ('S2_TP18',                base_trades, 0.0, 3, 0.18,  'equal'),
        ('S2_TP20',                base_trades, 0.0, 3, 0.20,  'equal'),
        ('S3_ProbWeighted',        base_trades, 0.0, 3, None,  'prob_weighted'),
        ('S4_Top1_P04',            base_trades, 0.4, 1, None,  'equal'),
        ('S4_Top1_P04_TP18',       base_trades, 0.4, 1, 0.18,  'equal'),
        ('S4_Top1_P04_TP20',       base_trades, 0.4, 1, 0.20,  'equal'),
        ('S4_Top2_P04',            base_trades, 0.4, 2, None,  'equal'),
        ('S4_Top2_P04_TP18',       base_trades, 0.4, 2, 0.18,  'equal'),
        ('S5_Ext_Top1_P04_PW',     ext_trades,  0.4, 1, None,  'prob_weighted'),
        ('S5_Ext_Top1_P04_TP18_PW',ext_trades,  0.4, 1, 0.18,  'prob_weighted'),
        ('S5_Ext_Top2_P04_PW',     ext_trades,  0.4, 2, None,  'prob_weighted'),
        ('S5_Ext_Top2_P04_TP18_PW',ext_trades,  0.4, 2, 0.18,  'prob_weighted'),
        ('S5_Ext_Top1_P05_TP18_PW',ext_trades,  0.5, 1, 0.18,  'prob_weighted'),
        ('S5_Ext_Top1_P03_PW',     ext_trades,  0.3, 1, None,  'prob_weighted'),
    ]

    results = {}
    for sname, pool, p_thresh, top_n, tp, pos_sizing in schemes:
        selected = apply_selection(pool, prob_thresh=p_thresh, top_n=top_n)
        eq, stats = backtest_scheme(selected, all_dates_set, take_profit=tp, position_sizing=pos_sizing)
        if stats:
            results[sname] = (eq, stats, selected)
            print(f"  {sname:<30} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  "
                  f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}  Trades={stats['trades']:>5d}  "
                  f"Win={stats['win_rate']:.2%}", flush=True)

    # Sort by Sharpe
    sorted_results = sorted(results.items(), key=lambda x: x[1][1]['sharpe'], reverse=True)

    print(f"\n{'='*90}", flush=True)
    print(f"  FINAL RANKING (by Sharpe)", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'Rank':>4} {'Scheme':<30} {'Total':>10} {'Annual':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'Trades':>8} {'WinRate':>8}", flush=True)
    print("-" * 100, flush=True)
    for rank, (sname, (eq, stats, _)) in enumerate(sorted_results, 1):
        print(f"{rank:>4} {sname:<30} {stats['total']:>9.2%} {stats['ann']:>9.2%} {stats['sharpe']:>7.2f} "
              f"{stats['mdd']:>9.2%} {stats['calmar']:>7.2f} {stats['trades']:>7d} {stats['win_rate']:>7.2%}", flush=True)

    # Also rank by total return
    sorted_by_return = sorted(results.items(), key=lambda x: x[1][1]['total'], reverse=True)
    print(f"\n  TOP 5 by Total Return:", flush=True)
    for rank, (sname, (eq, stats, _)) in enumerate(sorted_by_return[:5], 1):
        print(f"  {rank}. {sname:<30} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}", flush=True)

    best_name = sorted_results[0][0]
    best_eq, best_stats, best_trades = sorted_results[0][1]
    print(f"\n  BEST (by Sharpe): {best_name} -> Sharpe={best_stats['sharpe']:.2f}, Total={best_stats['total']:.2%}", flush=True)

    best_ret_name = sorted_by_return[0][0]
    best_ret_eq, best_ret_stats, _ = sorted_by_return[0][1]
    print(f"  BEST (by Return): {best_ret_name} -> Total={best_ret_stats['total']:.2%}, Sharpe={best_ret_stats['sharpe']:.2f}", flush=True)

    # Save best results
    best_eq.to_csv(os.path.join(THIS_DIR, 'best_equity_sharpe.csv'), index=False)
    best_trades.to_csv(os.path.join(THIS_DIR, 'best_trades_sharpe.csv'), index=False)
    best_ret_eq.to_csv(os.path.join(THIS_DIR, 'best_equity_return.csv'), index=False)

    # Plot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(3, 1, figsize=(18, 20))

    # Plot 1: Top 10 by Sharpe
    for i, (sname, (eq, stats, _)) in enumerate(sorted_results[:10]):
        eq_norm = eq['nav'] / eq['nav'].iloc[0]
        label = f"#{i+1} {sname}: S={stats['sharpe']:.2f} R={stats['total']:.0%}"
        axes[0].plot(eq['date'], eq_norm, label=label, linewidth=1.5)
    axes[0].set_title('Top 10 by Sharpe Ratio', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('NAV (normalized)')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Top 5 by Return
    for i, (sname, (eq, stats, _)) in enumerate(sorted_by_return[:5]):
        eq_norm = eq['nav'] / eq['nav'].iloc[0]
        label = f"#{i+1} {sname}: R={stats['total']:.0%} S={stats['sharpe']:.2f}"
        axes[1].plot(eq['date'], eq_norm, label=label, linewidth=2)
    axes[1].set_title('Top 5 by Total Return', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('NAV (normalized)')
    axes[1].legend(fontsize=8, loc='upper left')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Category comparison (best from each category)
    categories = {}
    for sname, (eq, stats, _) in sorted_results:
        cat = sname.split('_')[0] + '_' + sname.split('_')[1] if '_' in sname else sname
        if cat not in categories or stats['sharpe'] > categories[cat][1]['sharpe']:
            categories[cat] = (sname, stats, eq)

    for cat, (sname, stats, eq) in categories.items():
        eq_norm = eq['nav'] / eq['nav'].iloc[0]
        label = f"{sname}: S={stats['sharpe']:.2f} R={stats['total']:.0%}"
        axes[2].plot(eq['date'], eq_norm, label=label, linewidth=1.5)
    axes[2].set_title('Best from Each Category', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('NAV (normalized)')
    axes[2].legend(fontsize=8, loc='upper left')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(THIS_DIR, 'enhanced_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved to enhanced_comparison.png", flush=True)

    # Save all equity curves
    for sname, (eq, stats, trades_df) in sorted_results:
        safe_name = sname.replace(' ', '_')
        eq.to_csv(os.path.join(THIS_DIR, f'equity_{safe_name}.csv'), index=False)
        trades_df.to_csv(os.path.join(THIS_DIR, f'trades_{safe_name}.csv'), index=False)

    print(f"\n  All results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
