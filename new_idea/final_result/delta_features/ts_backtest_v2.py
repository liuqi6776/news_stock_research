"""
TS Enhanced Backtest - All-in-one approach.
No need to load the huge labeled panel. Instead:
1. Load base model
2. For each test date, compute TS features on-the-fly
3. Use pre-trained TS model for prediction
4. Backtest and compare

This avoids the memory issue of loading 3GB+ parquet files.
"""
import os, sys, json
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
INDEX_DIR = os.path.join(DATA_DIR, 'index_day1')
NEWS_MAJOR_DIR = r"C:\Users\liuqi\clowspace\toolstock\news_major1"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.dirname(THIS_DIR)
DOUBAO_DIR = os.path.join(FINAL_DIR, 'doubao')

CIRC_MV_LIMIT = 1000000
TEST_START = '20230101'
TEST_END = '20260324'
TRAIN_START = '20200801'
LOOKBACK = 60

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']

def is_main_board(ts_code):
    code = ts_code[:6]
    if code.startswith(('60',)) and ts_code.endswith('.SH'):
        return True
    if code.startswith(('00',)) and ts_code.endswith('.SZ'):
        return True
    return False

def int_to_date(date_int):
    s = str(int(date_int))
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))

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
    """Load all data for a single day, filtered to main board stocks."""
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

def compute_ts_features_for_stock(history_df, current_date_int):
    """Compute TS features for all stocks on a given date using history_df.
    
    history_df: DataFrame with columns [ts_code, date, open, close, high, low, vol, amount,
                pct_chg, pre_close, turnover_rate, volume_ratio, chip_concentration, 
                winner_rate, cost_50pct, weight_avg, hot_rank_pct]
    current_date_int: the date to compute features for
    
    All features use data up to current_date_int (inclusive).
    Target is T+2 close / T+1 open, so current date data is safe to use.
    """
    features_list = []

    for ts_code, group in history_df.groupby('ts_code'):
        group = group.sort_values('date').reset_index(drop=True)

        if len(group) < 5:
            continue

        current = group.iloc[-1]
        if int(current['date']) != current_date_int:
            continue

        f = {'ts_code': ts_code}

        # LAG FEATURES
        for lag in [1, 2, 3, 5, 10, 20]:
            if len(group) > lag:
                prev = group.iloc[-(lag+1)]
                f[f'close_lag{lag}'] = prev['close']
                f[f'vol_lag{lag}'] = prev['vol']
                f[f'turnover_rate_lag{lag}'] = prev.get('turnover_rate', np.nan)
                f[f'chip_concentration_lag{lag}'] = prev.get('chip_concentration', np.nan)
                f[f'winner_rate_lag{lag}'] = prev.get('winner_rate', np.nan)
                f[f'hot_rank_pct_lag{lag}'] = prev.get('hot_rank_pct', np.nan)

        # ROLLING STATS
        for window in [5, 10, 20, 60]:
            if len(group) >= window:
                w = group.tail(window)
                f[f'close_mean_{window}'] = w['close'].mean()
                f[f'close_std_{window}'] = w['close'].std()
                f[f'vol_mean_{window}'] = w['vol'].mean()
                f[f'vol_std_{window}'] = w['vol'].std()
                f[f'turnover_rate_mean_{window}'] = w.get('turnover_rate', pd.Series(dtype=float)).mean()
                f[f'chip_concentration_mean_{window}'] = w.get('chip_concentration', pd.Series(dtype=float)).mean()
                f[f'winner_rate_mean_{window}'] = w.get('winner_rate', pd.Series(dtype=float)).mean()
                f[f'hot_rank_pct_std_{window}'] = w.get('hot_rank_pct', pd.Series(dtype=float)).std()
                f[f'close_min_{window}'] = w['close'].min()
                f[f'close_max_{window}'] = w['close'].max()
            elif len(group) >= 3:
                w = group.tail(min(len(group), window))
                f[f'close_mean_{window}'] = w['close'].mean()
                f[f'close_std_{window}'] = w['close'].std()
                f[f'vol_mean_{window}'] = w['vol'].mean()

        # RETURNS
        if len(group) >= 2:
            f['ret_1d'] = current['pct_chg'] / 100
        if len(group) >= 3:
            f['ret_2d'] = (current['close'] / group.iloc[-3]['close']) - 1
        if len(group) >= 6:
            f['ret_5d'] = (current['close'] / group.iloc[-6]['close']) - 1
        if len(group) >= 11:
            f['ret_10d'] = (current['close'] / group.iloc[-11]['close']) - 1
        if len(group) >= 21:
            f['ret_20d'] = (current['close'] / group.iloc[-21]['close']) - 1

        # LOG RETURNS
        if len(group) >= 2 and current['close'] > 0 and group.iloc[-2]['close'] > 0:
            f['log_ret_1d'] = np.log(current['close'] / group.iloc[-2]['close'])
        if len(group) >= 3 and current['close'] > 0 and group.iloc[-3]['close'] > 0:
            f['log_ret_2d'] = np.log(current['close'] / group.iloc[-3]['close'])

        # MA DISTANCE
        for window in [5, 10, 20, 60]:
            if len(group) >= window and f.get(f'close_mean_{window}', 0) != 0:
                f[f'ma{window}_dist'] = (current['close'] / f[f'close_mean_{window}']) - 1

        # RSI
        if len(group) >= 15:
            changes = group['close'].diff().dropna()
            if len(changes) >= 14:
                gains = changes.where(changes > 0, 0).tail(14)
                losses = (-changes.where(changes < 0, 0)).tail(14)
                avg_gain = gains.mean()
                avg_loss = losses.mean()
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    f['rsi_14'] = 100 - (100 / (1 + rs))
                else:
                    f['rsi_14'] = 100

        # BOLLINGER
        if len(group) >= 20:
            w = group.tail(20)
            mean20 = w['close'].mean()
            std20 = w['close'].std()
            if std20 > 0:
                f['boll_upper'] = mean20 + 2 * std20
                f['boll_lower'] = mean20 - 2 * std20
                f['boll_position'] = (current['close'] - f['boll_lower']) / (f['boll_upper'] - f['boll_lower'])

        # VOLUME RATIO
        if len(group) >= 6 and f.get('vol_mean_5', 0) > 0:
            f['vol_ratio_5d'] = current['vol'] / f['vol_mean_5']

        # CHIP CHANGE
        if len(group) >= 2:
            f['chip_conc_change_1d'] = current.get('chip_concentration', np.nan) - group.iloc[-2].get('chip_concentration', np.nan)
            f['winner_rate_change_1d'] = current.get('winner_rate', np.nan) - group.iloc[-2].get('winner_rate', np.nan)

        # CALENDAR FEATURES
        dt = int_to_date(current_date_int)
        f['day_of_week'] = dt.weekday()
        f['month'] = dt.month
        f['is_month_start'] = 1 if dt.day <= 5 else 0
        f['is_month_end'] = 1 if dt.day >= 25 else 0

        features_list.append(f)

    if not features_list:
        return pd.DataFrame()
    return pd.DataFrame(features_list)

def backtest_scheme(trades_df, initial_capital=1000000, max_positions=3, take_profit=0.15):
    """Backtest with position management."""
    if trades_df.empty:
        return pd.DataFrame(), {'total': 0, 'sharpe': 0, 'mdd': 0, 'win_rate': 0, 'calmar': 0, 'n_trades': 0}

    trades = trades_df.sort_values('entry_date').copy()
    capital = initial_capital
    equity_records = []

    for _, trade in trades.iterrows():
        if len([e for e in equity_records if e.get('active')]) >= max_positions:
            continue

        ret = trade['return']
        position_size = capital / max_positions
        pnl = position_size * ret
        capital += pnl

        equity_records.append({
            'date': trade['exit_date'],
            'capital': capital,
            'return': ret,
            'pnl': pnl
        })

    if not equity_records:
        return pd.DataFrame(), {'total': 0, 'sharpe': 0, 'mdd': 0, 'win_rate': 0, 'calmar': 0, 'n_trades': 0}

    eq = pd.DataFrame(equity_records).sort_values('date').reset_index(drop=True)
    eq['equity'] = eq['capital']

    total_ret = (capital / initial_capital) - 1
    daily_rets = eq.groupby('date')['return'].mean()
    sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * np.sqrt(252) if len(daily_rets) > 1 else 0

    eq['peak'] = eq['equity'].cummax()
    eq['drawdown'] = (eq['equity'] - eq['peak']) / eq['peak']
    mdd = eq['drawdown'].min()
    win_rate = (eq['return'] > 0).mean()
    calmar = total_ret / abs(mdd) if mdd != 0 else 0

    return eq, {'total': total_ret, 'sharpe': sharpe, 'mdd': mdd, 'win_rate': win_rate, 'calmar': calmar, 'n_trades': len(eq)}

def main():
    print("=" * 90, flush=True)
    print("  TS Enhanced Backtest - All-in-One (Memory Efficient)", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_int = [int(d) for d in all_dates]
    all_dates_set = set(all_dates_int)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    # Load base model
    print("\n[Step 1] Loading base model...", flush=True)
    doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
    loaded = joblib.load(doubao_model_path)
    base_model = loaded[0] if isinstance(loaded, tuple) else loaded
    print("  Base model loaded", flush=True)

    # Phase 1: Train TS model using training period data
    print("\n[Step 2] Training TS model...", flush=True)
    ts_model_path = os.path.join(THIS_DIR, 'models', 'ts_model_v2.joblib')
    os.makedirs(os.path.join(THIS_DIR, 'models'), exist_ok=True)

    if os.path.exists(ts_model_path):
        ts_model = joblib.load(ts_model_path)
        ts_feat_cols = joblib.load(os.path.join(THIS_DIR, 'models', 'ts_feat_cols_v2.joblib'))
        print(f"  Loaded existing TS model ({len(ts_feat_cols)} features)", flush=True)
    else:
        # Build training data from training period
        print("  Building training data...", flush=True)
        train_dates = [d for d in all_dates_int if int(TRAIN_START) <= d < int(TEST_START)]
        print(f"  Training dates: {len(train_dates)}", flush=True)

        # Sample training dates (every 5th day to save time)
        sample_dates = train_dates[::5]
        print(f"  Sampled training dates: {len(sample_dates)}", flush=True)

        train_records = []
        # Pre-load history for training
        history_cache = {}

        for i, d in enumerate(sample_dates):
            d_idx = all_dates_int.index(d)
            if d_idx < LOOKBACK:
                continue

            # Load lookback data
            lookback_dates = all_dates_int[max(0, d_idx-LOOKBACK):d_idx+1]
            for ld in lookback_dates:
                if ld not in history_cache:
                    day_data = load_day_data(ld)
                    if day_data is not None:
                        day_data['date'] = ld
                        history_cache[ld] = day_data

            # Build history DataFrame
            hist_frames = [history_cache[ld] for ld in lookback_dates if ld in history_cache]
            if not hist_frames:
                continue
            history_df = pd.concat(hist_frames, ignore_index=True)

            # Compute features
            ts_feats = compute_ts_features_for_stock(history_df, d)
            if ts_feats.empty:
                continue

            # Add labels
            dt = int_to_date(d)
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

            p1_path = os.path.join(PRICE_DIR, f"{t1}.parquet")
            p2_path = os.path.join(PRICE_DIR, f"{t2}.parquet")
            if not os.path.exists(p1_path) or not os.path.exists(p2_path):
                continue

            price_t1 = pd.read_parquet(p1_path, columns=['ts_code', 'open'])
            price_t2 = pd.read_parquet(p2_path, columns=['ts_code', 'close'])

            labels = pd.merge(ts_feats[['ts_code']], price_t1, on='ts_code', how='left')
            labels = pd.merge(labels, price_t2, on='ts_code', how='left')
            labels['label_ret'] = labels['close'] / labels['open'] - 1
            labels['label'] = (labels['label_ret'] > 0.04).astype(int)

            ts_feats = pd.merge(ts_feats, labels[['ts_code', 'label']], on='ts_code', how='left')
            ts_feats = ts_feats.dropna(subset=['label'])
            train_records.append(ts_feats)

            # Clean cache periodically
            if len(history_cache) > 100:
                old_dates = sorted(history_cache.keys())[:50]
                for od in old_dates:
                    del history_cache[od]

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(sample_dates)} dates, {sum(len(r) for r in train_records)} samples", flush=True)

        train_df = pd.concat(train_records, ignore_index=True)
        print(f"  Training data: {len(train_df)} rows, pos_rate={train_df['label'].mean():.3f}", flush=True)

        # Get feature columns
        ts_feat_cols = [c for c in train_df.columns if c not in ['ts_code', 'label']]
        print(f"  TS features: {len(ts_feat_cols)}", flush=True)

        # Train
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
        joblib.dump(ts_model, ts_model_path)
        joblib.dump(ts_feat_cols, os.path.join(THIS_DIR, 'models', 'ts_feat_cols_v2.joblib'))
        print("  TS model trained and saved!", flush=True)

    # Phase 2: Generate trades and backtest
    print("\n[Step 3] Generating trades for backtest...", flush=True)
    test_dates = [d for d in all_dates_int if int(TEST_START) <= d <= int(TEST_END)]
    print(f"  Test dates: {len(test_dates)}", flush=True)

    all_trades = []
    history_cache = {}

    for i, d in enumerate(test_dates):
        d_idx = all_dates_int.index(d)
        if d_idx < LOOKBACK:
            continue

        # Load lookback data
        lookback_dates = all_dates_int[max(0, d_idx-LOOKBACK):d_idx+1]
        for ld in lookback_dates:
            if ld not in history_cache:
                day_data = load_day_data(ld)
                if day_data is not None:
                    day_data['date'] = ld
                    history_cache[ld] = day_data

        hist_frames = [history_cache[ld] for ld in lookback_dates if ld in history_cache]
        if not hist_frames:
            continue
        history_df = pd.concat(hist_frames, ignore_index=True)

        # Current day data for base features
        current_data = history_cache.get(d)
        if current_data is None:
            continue

        # Compute TS features
        ts_feats = compute_ts_features_for_stock(history_df, d)
        if ts_feats.empty:
            continue

        # Base model predictions
        base_df = current_data[['ts_code'] + [c for c in BASE_FEATS if c in current_data.columns]].copy()
        for feat in BASE_FEATS:
            if feat not in base_df.columns:
                base_df[feat] = 0
        base_df = base_df.fillna(0)
        base_df['base_prob'] = base_model.predict_proba(base_df[BASE_FEATS])[:, 1]

        # TS model predictions
        ts_pred_df = ts_feats[['ts_code']].copy()
        for feat in ts_feat_cols:
            if feat not in ts_feats.columns:
                ts_pred_df[feat] = 0
        ts_pred_df = ts_pred_df.fillna(0)
        ts_feats['ts_prob'] = ts_model.predict_proba(ts_pred_df[ts_feat_cols].fillna(0))[:, 1]

        # Merge predictions
        merged = pd.merge(base_df[['ts_code', 'base_prob']], ts_feats[['ts_code', 'ts_prob']], on='ts_code', how='inner')

        # Combined score
        merged['comb_prob'] = 0.5 * merged['base_prob'] + 0.5 * merged['ts_prob']

        # Get T+1/T+2 prices
        dt = int_to_date(d)
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

        p1_path = os.path.join(PRICE_DIR, f"{t1}.parquet")
        p2_path = os.path.join(PRICE_DIR, f"{t2}.parquet")
        if not os.path.exists(p1_path) or not os.path.exists(p2_path):
            continue

        price_t1 = pd.read_parquet(p1_path, columns=['ts_code', 'open'])
        price_t2 = pd.read_parquet(p2_path, columns=['ts_code', 'close', 'high'])

        merged = pd.merge(merged, price_t1, on='ts_code', how='left')
        merged = pd.merge(merged, price_t2, on='ts_code', how='left')
        merged = merged.dropna(subset=['open', 'close'])

        if merged.empty:
            continue

        merged['actual_ret'] = merged['close'] / merged['open'] - 1
        merged['high_ret'] = merged['high'] / merged['open'] - 1

        for _, row in merged.iterrows():
            all_trades.append({
                'date': d,
                'ts_code': row['ts_code'],
                'entry_date': t1,
                'exit_date': t2,
                'entry_price': row['open'],
                'exit_price': row['close'],
                'high_price': row['high'],
                'return': row['actual_ret'],
                'high_return': row['high_ret'],
                'base_prob': row['base_prob'],
                'ts_prob': row['ts_prob'],
                'comb_prob': row['comb_prob'],
            })

        # Clean cache
        if len(history_cache) > 100:
            old_dates = sorted(history_cache.keys())[:50]
            for od in old_dates:
                del history_cache[od]

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_dates)} dates, {len(all_trades)} trades", flush=True)

    trades_df = pd.DataFrame(all_trades)
    print(f"  Total trades: {len(trades_df)}", flush=True)

    # Save trades
    trades_df.to_parquet(os.path.join(THIS_DIR, 'ts_trades_v2.parquet'), index=False)

    # Backtest schemes
    print("\n[Step 4] Backtesting schemes...", flush=True)

    schemes = {
        'base_top3_p04': {'prob_col': 'base_prob', 'prob_thresh': 0.4, 'top_n': 3, 'tp': 0.15},
        'ts_top3_p04': {'prob_col': 'ts_prob', 'prob_thresh': 0.4, 'top_n': 3, 'tp': 0.15},
        'comb_top3_p04': {'prob_col': 'comb_prob', 'prob_thresh': 0.4, 'top_n': 3, 'tp': 0.15},
        'base_top3_p05': {'prob_col': 'base_prob', 'prob_thresh': 0.5, 'top_n': 3, 'tp': 0.15},
        'comb_top3_p05': {'prob_col': 'comb_prob', 'prob_thresh': 0.5, 'top_n': 3, 'tp': 0.15},
        'base_top2_p04': {'prob_col': 'base_prob', 'prob_thresh': 0.4, 'top_n': 2, 'tp': 0.15},
        'comb_top2_p04': {'prob_col': 'comb_prob', 'prob_thresh': 0.4, 'top_n': 2, 'tp': 0.15},
        'base_top3_p04_tp18': {'prob_col': 'base_prob', 'prob_thresh': 0.4, 'top_n': 3, 'tp': 0.18},
        'comb_top3_p04_tp18': {'prob_col': 'comb_prob', 'prob_thresh': 0.4, 'top_n': 3, 'tp': 0.18},
        'rerank_base_ts': {'prob_col': 'base_prob', 'prob_thresh': 0.3, 'top_n': 3, 'tp': 0.15, 'rerank_col': 'ts_prob', 'rerank_top': 10},
        'rerank_base_comb': {'prob_col': 'base_prob', 'prob_thresh': 0.3, 'top_n': 3, 'tp': 0.15, 'rerank_col': 'comb_prob', 'rerank_top': 10},
    }

    results = {}
    for sname, params in schemes.items():
        prob_col = params['prob_col']
        prob_thresh = params['prob_thresh']
        top_n = params['top_n']
        take_profit = params['tp']
        rerank_col = params.get('rerank_col')
        rerank_top = params.get('rerank_top')

        daily_picks = []
        for date, group in trades_df.groupby('date'):
            if rerank_col and rerank_top:
                candidates = group[group[prob_col] >= prob_thresh].nlargest(rerank_top, prob_col)
                picks = candidates.nlargest(top_n, rerank_col)
            else:
                picks = group[group[prob_col] >= prob_thresh].nlargest(top_n, prob_col)

            for _, pick in picks.iterrows():
                entry_price = pick['entry_price']
                high_price = pick.get('high_price', pick['exit_price'])
                exit_price = pick['exit_price']

                if take_profit and high_price / entry_price - 1 > take_profit:
                    actual_ret = take_profit
                else:
                    actual_ret = exit_price / entry_price - 1

                daily_picks.append({
                    'entry_date': pick['entry_date'],
                    'exit_date': pick['exit_date'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': actual_ret,
                    'ts_code': pick['ts_code']
                })

        picks_df = pd.DataFrame(daily_picks)
        eq, stats = backtest_scheme(picks_df, take_profit=take_profit)
        results[sname] = (eq, stats, picks_df)
        print(f"  {sname:<30} Total={stats['total']:>8.2%}  Sharpe={stats['sharpe']:>6.2f}  "
              f"MDD={stats['mdd']:>8.2%}  WinRate={stats['win_rate']:>6.2%}  Trades={stats['n_trades']}", flush=True)

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
            ax.plot(eq['date'], eq['equity'], label=f"{sname} (Sharpe={stats['sharpe']:.2f})")
    ax.set_title('Equity Curves - Top 6 Schemes')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
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
    plt.savefig(os.path.join(THIS_DIR, 'ts_backtest_v2_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Chart saved", flush=True)

    # Print final ranking
    print(f"\n{'Rank':>4} {'Scheme':<35} {'Total':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'WinRate':>8} {'Trades':>7}")
    print('-' * 100)
    for rank, (sname, (eq, stats, tdf)) in enumerate(sorted_r, 1):
        print(f"{rank:>4} {sname:<35} {stats['total']:>9.2%} {stats['sharpe']:>7.2f} "
              f"{stats['mdd']:>9.2%} {stats['calmar']:>7.2f} {stats['win_rate']:>7.2%} {stats['n_trades']:>7}")

    # Save equity curves
    for sname, (eq, stats, tdf) in sorted_r[:5]:
        if not eq.empty:
            eq.to_csv(os.path.join(THIS_DIR, f'equity_ts_v2_{sname}.csv'), index=False)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
