"""
TS Enhanced Backtest v3 - Vectorized feature computation.
Uses groupby + rolling instead of per-stock loops.
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
NEWS_MAJOR_DIR = r"C:\Users\liuqi\clowspace\toolstock\news_major1"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.dirname(THIS_DIR)

CIRC_MV_LIMIT = 1000000
TEST_START = 20230101
TEST_END = 20260324
TRAIN_START = 20200801
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

def compute_vectorized_ts_features(panel):
    """Compute TS features using vectorized groupby+rolling operations."""
    df = panel.copy()
    df = df.sort_values(['ts_code', 'date']).reset_index(drop=True)

    # RETURNS
    df['ret_1d'] = df.groupby('ts_code')['close'].pct_change(1)
    df['ret_2d'] = df.groupby('ts_code')['close'].pct_change(2)
    df['ret_5d'] = df.groupby('ts_code')['close'].pct_change(5)
    df['ret_10d'] = df.groupby('ts_code')['close'].pct_change(10)
    df['ret_20d'] = df.groupby('ts_code')['close'].pct_change(20)

    # LOG RETURNS
    df['log_ret_1d'] = np.log(df['close'] / df.groupby('ts_code')['close'].shift(1))
    df['log_ret_2d'] = np.log(df['close'] / df.groupby('ts_code')['close'].shift(2))

    # ROLLING MEAN/STD
    for window in [5, 10, 20]:
        for col in ['close', 'vol', 'turnover_rate', 'chip_concentration', 'winner_rate']:
            if col not in df.columns:
                continue
            grp = df.groupby('ts_code')[col]
            df[f'{col}_mean_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).mean())
            df[f'{col}_std_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).std())

    # CLOSE MIN/MAX
    for window in [5, 10, 20]:
        grp = df.groupby('ts_code')['close']
        df[f'close_min_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).min())
        df[f'close_max_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).max())

    # MA DISTANCE
    for window in [5, 10, 20]:
        mean_col = f'close_mean_{window}'
        if mean_col in df.columns:
            df[f'ma{window}_dist'] = (df['close'] / df[mean_col]) - 1

    # VOLUME RATIO
    if 'vol_mean_5' in df.columns:
        df['vol_ratio_5d'] = df['vol'] / (df['vol_mean_5'] + 1e-8)

    # RSI 14
    delta = df.groupby('ts_code')['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.groupby(df['ts_code']).transform(lambda x: x.rolling(14, min_periods=10).mean())
    avg_loss = loss.groupby(df['ts_code']).transform(lambda x: x.rolling(14, min_periods=10).mean())
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # BOLLINGER
    if 'close_mean_20' in df.columns and 'close_std_20' in df.columns:
        df['boll_upper'] = df['close_mean_20'] + 2 * df['close_std_20']
        df['boll_lower'] = df['close_mean_20'] - 2 * df['close_std_20']
        df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'] + 1e-8)

    # CHIP CHANGE
    df['chip_conc_change_1d'] = df.groupby('ts_code')['chip_concentration'].diff(1)
    df['winner_rate_change_1d'] = df.groupby('ts_code')['winner_rate'].diff(1)

    # HOT RANK STD
    if 'hot_rank_pct' in df.columns:
        for window in [5, 10]:
            df[f'hot_rank_pct_std_{window}'] = df.groupby('ts_code')['hot_rank_pct'].transform(
                lambda x: x.rolling(window, min_periods=3).std())

    # CALENDAR
    df['day_of_week'] = df['date'].apply(lambda x: int_to_date(int(x)).weekday())
    df['month'] = df['date'].apply(lambda x: int_to_date(int(x)).month)
    df['is_month_start'] = df['date'].apply(lambda x: 1 if int_to_date(int(x)).day <= 5 else 0)
    df['is_month_end'] = df['date'].apply(lambda x: 1 if int_to_date(int(x)).day >= 25 else 0)

    return df

def backtest_scheme(trades_df, initial_capital=1000000, max_positions=3, take_profit=0.15):
    if trades_df.empty:
        return pd.DataFrame(), {'total': 0, 'sharpe': 0, 'mdd': 0, 'win_rate': 0, 'calmar': 0, 'n_trades': 0}

    trades = trades_df.sort_values('entry_date').copy()
    capital = initial_capital
    equity_records = []

    for _, trade in trades.iterrows():
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
    print("  TS Enhanced Backtest v3 - Vectorized Features", flush=True)
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

    # Phase 1: Build panel and train TS model
    print("\n[Step 2] Building training panel...", flush=True)
    ts_model_path = os.path.join(THIS_DIR, 'models', 'ts_model_v3.joblib')
    ts_feat_path = os.path.join(THIS_DIR, 'models', 'ts_feat_cols_v3.joblib')
    os.makedirs(os.path.join(THIS_DIR, 'models'), exist_ok=True)

    if os.path.exists(ts_model_path) and os.path.exists(ts_feat_path):
        ts_model = joblib.load(ts_model_path)
        ts_feat_cols = joblib.load(ts_feat_path)
        print(f"  Loaded existing TS model ({len(ts_feat_cols)} features)", flush=True)
    else:
        # Build training panel: load LOOKBACK days of data for sampled training dates
        train_dates = [d for d in all_dates_int if TRAIN_START <= d < TEST_START]
        # Sample every 3rd date
        sample_dates = train_dates[::3]
        print(f"  Training dates: {len(sample_dates)}", flush=True)

        # Build panel: load all dates in range
        panel_start = all_dates_int[max(0, all_dates_int.index(train_dates[0]) - LOOKBACK)]
        panel_dates = [d for d in all_dates_int if panel_start <= d < TEST_START]
        print(f"  Panel dates: {len(panel_dates)}", flush=True)

        records = []
        for i, d in enumerate(panel_dates):
            day_data = load_day_data(d)
            if day_data is None:
                continue
            day_data['date'] = d
            records.append(day_data)
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i+1}/{len(panel_dates)} dates", flush=True)

        panel = pd.concat(records, ignore_index=True)
        panel = panel.sort_values(['ts_code', 'date']).reset_index(drop=True)
        print(f"  Panel: {len(panel)} rows", flush=True)

        # Compute features
        print("  Computing TS features...", flush=True)
        panel = compute_vectorized_ts_features(panel)
        print(f"  After features: {len(panel.columns)} columns", flush=True)

        # Add labels for sampled dates only
        print("  Adding labels...", flush=True)
        panel['label'] = np.nan
        panel['label_ret'] = np.nan

        for d in sample_dates:
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

            p1 = os.path.join(PRICE_DIR, f"{t1}.parquet")
            p2 = os.path.join(PRICE_DIR, f"{t2}.parquet")
            if not os.path.exists(p1) or not os.path.exists(p2):
                continue

            price_t1 = pd.read_parquet(p1, columns=['ts_code', 'open'])
            price_t2 = pd.read_parquet(p2, columns=['ts_code', 'close'])

            labels = pd.merge(price_t1, price_t2, on='ts_code', how='inner')
            labels['label_ret'] = labels['close'] / labels['open'] - 1
            labels['label'] = (labels['label_ret'] > 0.04).astype(int)
            labels = labels[['ts_code', 'label', 'label_ret']]

            mask = panel['date'] == d
            day_panel = panel.loc[mask, ['ts_code']].copy()
            day_labels = pd.merge(day_panel, labels, on='ts_code', how='left')
            panel.loc[mask, 'label'] = day_labels['label'].values
            panel.loc[mask, 'label_ret'] = day_labels['label_ret'].values

        train_df = panel[panel['label'].notna()].copy()
        print(f"  Training data: {len(train_df)} rows, pos_rate={train_df['label'].mean():.3f}", flush=True)

        # Get feature columns
        exclude = {'ts_code', 'date', 'open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount',
                   'pre_close', 'circ_mv', 'turnover_rate', 'volume_ratio',
                   'cost_50pct', 'weight_avg', 'cost_15pct', 'cost_85pct',
                   'chip_concentration', 'winner_rate', 'hot_rank_pct', 'label', 'label_ret',
                   'news_market_impact', 'news_stock_impact'}
        ts_feat_cols = [c for c in train_df.columns if c not in exclude and train_df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int8']]
        print(f"  TS features: {len(ts_feat_cols)}", flush=True)

        # Train model
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

        # Feature importance
        imp = pd.Series(ts_model.feature_importances_, index=ts_feat_cols).sort_values(ascending=False)
        print(f"  Top 20 features:", flush=True)
        for feat, score in imp.head(20).items():
            print(f"    {feat:<50} imp={score:.4f}", flush=True)

        # Select top features (importance > 0.005)
        selected = imp[imp > 0.005].index.tolist()
        # Also remove highly correlated
        if len(selected) > 30:
            corr = train_df[selected].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
            selected = [f for f in selected if f not in to_drop]

        ts_feat_cols = selected
        print(f"  Selected features: {len(ts_feat_cols)}", flush=True)

        # Retrain with selected features
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

        del panel, train_df

    # Phase 2: Generate trades
    print("\n[Step 3] Generating trades for backtest...", flush=True)
    test_dates = [d for d in all_dates_int if TEST_START <= d <= TEST_END]
    print(f"  Test dates: {len(test_dates)}", flush=True)

    # Build test panel with lookback
    first_test_idx = all_dates_int.index(test_dates[0])
    panel_start = all_dates_int[max(0, first_test_idx - LOOKBACK)]
    panel_dates = [d for d in all_dates_int if panel_start <= d <= TEST_END]
    print(f"  Panel dates (with lookback): {len(panel_dates)}", flush=True)

    # Process in chunks to manage memory
    CHUNK_SIZE = 60
    all_trades = []

    for chunk_start in range(0, len(panel_dates), CHUNK_SIZE // 2):
        chunk_dates = panel_dates[chunk_start:chunk_start + CHUNK_SIZE]
        if not chunk_dates:
            break

        # Only process if chunk contains test dates
        test_in_chunk = [d for d in chunk_dates if d >= TEST_START]
        if not test_in_chunk:
            continue

        # Load chunk data
        records = []
        for d in chunk_dates:
            day_data = load_day_data(d)
            if day_data is None:
                continue
            day_data['date'] = d
            records.append(day_data)

        if not records:
            continue

        chunk_panel = pd.concat(records, ignore_index=True)
        chunk_panel = chunk_panel.sort_values(['ts_code', 'date']).reset_index(drop=True)

        # Compute features
        chunk_panel = compute_vectorized_ts_features(chunk_panel)

        # Process each test date in chunk
        for d in test_in_chunk:
            day_data = chunk_panel[chunk_panel['date'] == d].copy()
            if day_data.empty:
                continue

            # Base features
            for feat in BASE_FEATS:
                if feat not in day_data.columns:
                    day_data[feat] = 0
            day_data = day_data.fillna({feat: 0 for feat in BASE_FEATS})
            day_data['base_prob'] = base_model.predict_proba(day_data[BASE_FEATS])[:, 1]

            # TS features
            for feat in ts_feat_cols:
                if feat not in day_data.columns:
                    day_data[feat] = 0
            day_data['ts_prob'] = ts_model.predict_proba(day_data[ts_feat_cols].fillna(0))[:, 1]

            # Combined
            day_data['comb_prob'] = 0.5 * day_data['base_prob'] + 0.5 * day_data['ts_prob']

            # Get T+1/T+2
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

            day_data = pd.merge(day_data, price_t1, on='ts_code', how='left', suffixes=('', '_t1'))
            day_data = pd.merge(day_data, price_t2, on='ts_code', how='left', suffixes=('', '_t2'))
            day_data = day_data.dropna(subset=['open_t1', 'close_t2'])

            if day_data.empty:
                continue

            day_data['actual_ret'] = day_data['close_t2'] / day_data['open_t1'] - 1
            day_data['high_ret'] = day_data['high_t2'] / day_data['open_t1'] - 1

            for _, row in day_data.iterrows():
                all_trades.append({
                    'date': d,
                    'ts_code': row['ts_code'],
                    'entry_date': t1,
                    'exit_date': t2,
                    'entry_price': row['open_t1'],
                    'exit_price': row['close_t2'],
                    'high_price': row.get('high_t2', row['close_t2']),
                    'return': row['actual_ret'],
                    'high_return': row.get('high_ret', row['actual_ret']),
                    'base_prob': row['base_prob'],
                    'ts_prob': row['ts_prob'],
                    'comb_prob': row['comb_prob'],
                })

        print(f"  Chunk {chunk_start//30+1}: {len(all_trades)} trades so far", flush=True)

        del chunk_panel

    trades_df = pd.DataFrame(all_trades)
    print(f"  Total trades: {len(trades_df)}", flush=True)
    trades_df.to_parquet(os.path.join(THIS_DIR, 'ts_trades_v3.parquet'), index=False)

    # Backtest
    print("\n[Step 4] Backtesting...", flush=True)

    schemes = {
        'base_top3_p04': {'prob_col': 'base_prob', 'thresh': 0.4, 'top_n': 3, 'tp': 0.15},
        'ts_top3_p04': {'prob_col': 'ts_prob', 'thresh': 0.4, 'top_n': 3, 'tp': 0.15},
        'comb_top3_p04': {'prob_col': 'comb_prob', 'thresh': 0.4, 'top_n': 3, 'tp': 0.15},
        'base_top3_p05': {'prob_col': 'base_prob', 'thresh': 0.5, 'top_n': 3, 'tp': 0.15},
        'comb_top3_p05': {'prob_col': 'comb_prob', 'thresh': 0.5, 'top_n': 3, 'tp': 0.15},
        'base_top2_p04': {'prob_col': 'base_prob', 'thresh': 0.4, 'top_n': 2, 'tp': 0.15},
        'comb_top2_p04': {'prob_col': 'comb_prob', 'thresh': 0.4, 'top_n': 2, 'tp': 0.15},
        'base_top3_p04_tp18': {'prob_col': 'base_prob', 'thresh': 0.4, 'top_n': 3, 'tp': 0.18},
        'comb_top3_p04_tp18': {'prob_col': 'comb_prob', 'thresh': 0.4, 'top_n': 3, 'tp': 0.18},
        'rerank_base_ts': {'prob_col': 'base_prob', 'thresh': 0.3, 'top_n': 3, 'tp': 0.15, 'rerank': 'ts_prob', 'rerank_top': 10},
        'rerank_base_comb': {'prob_col': 'base_prob', 'thresh': 0.3, 'top_n': 3, 'tp': 0.15, 'rerank': 'comb_prob', 'rerank_top': 10},
    }

    results = {}
    for sname, params in schemes.items():
        prob_col = params['prob_col']
        thresh = params['thresh']
        top_n = params['top_n']
        tp = params['tp']
        rerank = params.get('rerank')
        rerank_top = params.get('rerank_top')

        daily_picks = []
        for date, group in trades_df.groupby('date'):
            if rerank and rerank_top:
                cands = group[group[prob_col] >= thresh].nlargest(rerank_top, prob_col)
                picks = cands.nlargest(top_n, rerank)
            else:
                picks = group[group[prob_col] >= thresh].nlargest(top_n, prob_col)

            for _, pick in picks.iterrows():
                ep = pick['entry_price']
                hp = pick.get('high_price', pick['exit_price'])
                xp = pick['exit_price']
                ret = tp if (hp / ep - 1 > tp) else (xp / ep - 1)
                daily_picks.append({
                    'entry_date': pick['entry_date'],
                    'exit_date': pick['exit_date'],
                    'entry_price': ep,
                    'exit_price': xp,
                    'return': ret,
                    'ts_code': pick['ts_code']
                })

        picks_df = pd.DataFrame(daily_picks)
        eq, stats = backtest_scheme(picks_df, take_profit=tp)
        results[sname] = (eq, stats, picks_df)
        print(f"  {sname:<30} Total={stats['total']:>8.2%}  Sharpe={stats['sharpe']:>6.2f}  "
              f"MDD={stats['mdd']:>8.2%}  WinRate={stats['win_rate']:>6.2%}  Trades={stats['n_trades']}", flush=True)

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
    plt.savefig(os.path.join(THIS_DIR, 'ts_backtest_v3_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Chart saved", flush=True)

    print(f"\n{'Rank':>4} {'Scheme':<35} {'Total':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'WinRate':>8} {'Trades':>7}")
    print('-' * 100)
    for rank, (sname, (eq, stats, tdf)) in enumerate(sorted_r, 1):
        print(f"{rank:>4} {sname:<35} {stats['total']:>9.2%} {stats['sharpe']:>7.2f} "
              f"{stats['mdd']:>9.2%} {stats['calmar']:>7.2f} {stats['win_rate']:>7.2%} {stats['n_trades']:>7}")

    for sname, (eq, stats, tdf) in sorted_r[:5]:
        if not eq.empty:
            eq.to_csv(os.path.join(THIS_DIR, f'equity_ts_v3_{sname}.csv'), index=False)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
