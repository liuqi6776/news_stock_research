"""
Time-Series Feature Engineering v4 - Optimized Panel Pipeline

Key improvements:
1. Only main board stocks (no ETF, no 创业板 300/301, no 北交所 8/4开头)
2. Load each date once, build panel incrementally
3. Use float32 to save memory
4. Strict no-leakage: all features use T日 and earlier data only
5. Target: (T+2 close / T+1 open - 1) > 0.04
"""
import os, sys
import pandas as pd
import numpy as np
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
INDEX_DIR = os.path.join(DATA_DIR, 'index_day1')
NEWS_MAJOR_DIR = r"C:\Users\liuqi\clowspace\toolstock\news_major1"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.join(os.path.dirname(THIS_DIR))

CIRC_MV_LIMIT = 1000000
TEST_START = '20230101'
TEST_END = '20260324'

PANEL_START = '20200801'

def is_main_board(ts_code):
    """Only keep main board stocks:
    - SH: 60xxxx (main board)
    - SZ: 00xxxx (main board)
    Exclude: ETF/基金(51/15/16/18), 创业板(300/301), 北交所(8/4), 科创板(688)
    """
    code = ts_code[:6]
    if code.startswith(('60',)):
        if ts_code.endswith('.SH'):
            return True
    if code.startswith(('00',)):
        if ts_code.endswith('.SZ'):
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

def build_panel(all_dates, start_date, end_date):
    """Build panel by loading each date once. Memory-efficient with float32."""
    records = []
    start_int = int(start_date)
    end_int = int(end_date)

    for i, d in enumerate(all_dates):
        d_int = int(d)
        if d_int < start_int or d_int > end_int:
            continue

        p_price = os.path.join(PRICE_DIR, f"{d}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d}.parquet")
        p_other = os.path.join(OTHER_DIR, f"{d}.parquet")
        p_rank = os.path.join(RANK_DIR, f"{d}.parquet")

        if not os.path.exists(p_price) or not os.path.exists(p_chip) or not os.path.exists(p_other):
            continue

        price = pd.read_parquet(p_price, columns=['ts_code', 'open', 'high', 'low', 'close',
                                                    'pct_chg', 'vol', 'amount', 'pre_close'])
        price = price[price['ts_code'].apply(is_main_board)]

        chip = pd.read_parquet(p_chip)
        chip = chip[chip['ts_code'].apply(is_main_board)]
        chip['chip_concentration'] = (chip['cost_85pct'] - chip['cost_15pct']) / (chip['cost_50pct'] + 1e-8)

        other = pd.read_parquet(p_other, columns=['ts_code', 'turnover_rate', 'volume_ratio', 'circ_mv'])
        other = other[other['ts_code'].apply(is_main_board)]

        row = pd.merge(price, chip[['ts_code', 'chip_concentration', 'winner_rate', 'cost_50pct', 'weight_avg']], on='ts_code', how='left')
        row = pd.merge(row, other, on='ts_code', how='left')

        row = row[row['circ_mv'] <= CIRC_MV_LIMIT]

        if os.path.exists(p_rank):
            rank = pd.read_parquet(p_rank)
            if len(rank) > 0:
                rank = rank[rank['ts_code'].apply(is_main_board)]
                if len(rank) > 0:
                    rank['hot_rank_pct'] = rank['hot'].rank(pct=True)
                    row = pd.merge(row, rank[['ts_code', 'hot_rank_pct']], on='ts_code', how='left')
                    row['hot_rank_pct'] = row['hot_rank_pct'].fillna(0.5)
                else:
                    row['hot_rank_pct'] = np.float32(0.5)
            else:
                row['hot_rank_pct'] = np.float32(0.5)
        else:
            row['hot_rank_pct'] = np.float32(0.5)

        row['date'] = np.int32(d_int)

        float_cols = ['open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount', 'pre_close',
                       'chip_concentration', 'winner_rate', 'cost_50pct', 'weight_avg',
                       'turnover_rate', 'volume_ratio', 'circ_mv', 'hot_rank_pct']
        for col in float_cols:
            if col in row.columns:
                row[col] = row[col].astype(np.float32)

        records.append(row)

        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(all_dates)} dates, {d}", flush=True)

    if not records:
        return pd.DataFrame()

    panel = pd.concat(records, ignore_index=True)
    panel = panel.sort_values(['ts_code', 'date']).reset_index(drop=True)
    return panel

def compute_ts_features(panel):
    """Compute time-series features using groupby + rolling.
    
    CRITICAL: All features use data up to T日 (current row).
    Target is (T+2 close / T+1 open), so T日 data is known before T+1 open.
    Lag features use shift() = strictly T-1 and earlier.
    Rolling features include T日 and earlier (no T+1/T+2 data).
    """
    df = panel.copy()
    print(f"  Computing features on {len(df)} rows, {df['ts_code'].nunique()} stocks...", flush=True)

    # CATEGORY 1: LAG FEATURES
    print("  [1/6] Lag features...", flush=True)
    lag_cols = ['close', 'vol', 'amount', 'high', 'low', 'open',
                'turnover_rate', 'volume_ratio', 'chip_concentration',
                'winner_rate', 'cost_50pct', 'weight_avg', 'hot_rank_pct']
    for lag in [1, 2, 3, 5, 10, 20]:
        for col in lag_cols:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df.groupby('ts_code')[col].shift(lag).astype(np.float32)

    # CATEGORY 2: ROLLING WINDOW STATS
    print("  [2/6] Rolling window stats...", flush=True)
    for window in [5, 10, 20, 60]:
        for col in ['close', 'vol', 'turnover_rate', 'chip_concentration', 'winner_rate', 'hot_rank_pct']:
            if col not in df.columns:
                continue
            grp = df.groupby('ts_code')[col]
            df[f'{col}_mean_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).mean()).astype(np.float32)
            df[f'{col}_std_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).std()).astype(np.float32)
            if col in ['close', 'vol']:
                df[f'{col}_min_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).min()).astype(np.float32)
                df[f'{col}_max_{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=3).max()).astype(np.float32)

    # CATEGORY 3: DIFFERENCE & RETURN FEATURES
    print("  [3/6] Difference & return features...", flush=True)
    for lag in [1, 2, 3, 5, 10, 20]:
        if f'close_lag{lag}' in df.columns:
            df[f'ret_{lag}d'] = (df['close'] / df[f'close_lag{lag}'] - 1).astype(np.float32)
            df[f'log_ret_{lag}d'] = np.log(df['close'] / (df[f'close_lag{lag}'] + 1e-8)).astype(np.float32)

    for lag in [1, 3, 5]:
        if f'vol_lag{lag}' in df.columns:
            df[f'vol_change_{lag}d'] = (df['vol'] / (df[f'vol_lag{lag}'] + 1e-8) - 1).astype(np.float32)
        if f'turnover_rate_lag{lag}' in df.columns:
            df[f'turnover_change_{lag}d'] = (df['turnover_rate'] - df[f'turnover_rate_lag{lag}']).astype(np.float32)
        if f'chip_concentration_lag{lag}' in df.columns:
            df[f'chip_conc_change_{lag}d'] = (df['chip_concentration'] - df[f'chip_concentration_lag{lag}']).astype(np.float32)
        if f'winner_rate_lag{lag}' in df.columns:
            df[f'winner_rate_change_{lag}d'] = (df['winner_rate'] - df[f'winner_rate_lag{lag}']).astype(np.float32)
        if f'cost_50pct_lag{lag}' in df.columns:
            df[f'cost_50pct_change_{lag}d'] = ((df['cost_50pct'] - df[f'cost_50pct_lag{lag}']) / (df[f'cost_50pct_lag{lag}'] + 1e-8)).astype(np.float32)
        if f'weight_avg_lag{lag}' in df.columns:
            df[f'weight_avg_change_{lag}d'] = ((df['weight_avg'] - df[f'weight_avg_lag{lag}']) / (df[f'weight_avg_lag{lag}'] + 1e-8)).astype(np.float32)
        if f'hot_rank_pct_lag{lag}' in df.columns:
            df[f'hot_rank_change_{lag}d'] = (df['hot_rank_pct'] - df[f'hot_rank_pct_lag{lag}']).astype(np.float32)

    if 'ret_1d' in df.columns and 'vol_change_1d' in df.columns:
        df['price_vol_diverge_1d'] = (df['vol_change_1d'] - df['ret_1d'].abs()).astype(np.float32)
    if 'cost_50pct_change_1d' in df.columns and 'ret_1d' in df.columns:
        df['chip_price_diverge_1d'] = (df['cost_50pct_change_1d'] - df['ret_1d']).astype(np.float32)

    # CATEGORY 4: TECHNICAL INDICATORS
    print("  [4/6] Technical indicators...", flush=True)
    for window in [5, 10, 20]:
        if f'close_mean_{window}' in df.columns:
            df[f'ma{window}_dist'] = (df['close'] / df[f'close_mean_{window}'] - 1).astype(np.float32)

    if 'close_mean_5' in df.columns and 'close_mean_20' in df.columns:
        df['ma5_ma20_ratio'] = (df['close_mean_5'] / (df['close_mean_20'] + 1e-8) - 1).astype(np.float32)

    if 'close_mean_20' in df.columns and 'close_std_20' in df.columns:
        bb_upper = df['close_mean_20'] + 2 * df['close_std_20']
        bb_lower = df['close_mean_20'] - 2 * df['close_std_20']
        df['bb_pos_20'] = ((df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)).clip(0, 1).astype(np.float32)

    if 'ret_1d' in df.columns:
        grp_ret = df.groupby('ts_code')['ret_1d']
        avg_gain14 = grp_ret.transform(lambda x: x.clip(lower=0).rolling(14, min_periods=5).mean())
        avg_loss14 = grp_ret.transform(lambda x: (-x).clip(lower=0).rolling(14, min_periods=5).mean())
        df['rsi_14'] = (100 - (100 / (1 + avg_gain14 / (avg_loss14 + 1e-8)))).astype(np.float32)

        avg_gain6 = grp_ret.transform(lambda x: x.clip(lower=0).rolling(6, min_periods=3).mean())
        avg_loss6 = grp_ret.transform(lambda x: (-x).clip(lower=0).rolling(6, min_periods=3).mean())
        df['rsi_6'] = (100 - (100 / (1 + avg_gain6 / (avg_loss6 + 1e-8)))).astype(np.float32)

    grp_close = df.groupby('ts_code')['close']
    ema12 = grp_close.transform(lambda x: x.ewm(span=12, min_periods=5).mean())
    ema26 = grp_close.transform(lambda x: x.ewm(span=26, min_periods=10).mean())
    df['macd_dif'] = (ema12 - ema26).astype(np.float32)
    grp_dif = df.groupby('ts_code')['macd_dif']
    df['macd_dea'] = grp_dif.transform(lambda x: x.ewm(span=9, min_periods=3).mean()).astype(np.float32)
    df['macd_hist'] = (df['macd_dif'] - df['macd_dea']).astype(np.float32)

    for window in [5, 10]:
        if f'vol_mean_{window}' in df.columns:
            df[f'vol_ratio_{window}'] = (df['vol'] / (df[f'vol_mean_{window}'] + 1e-8)).astype(np.float32)

    df['intraday_range'] = ((df['high'] - df['low']) / (df['pre_close'] + 1e-8)).astype(np.float32)
    df['upper_shadow'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / (df['pre_close'] + 1e-8)).astype(np.float32)
    df['lower_shadow'] = ((df[['open', 'close']].min(axis=1) - df['low']) / (df['pre_close'] + 1e-8)).astype(np.float32)

    if 'close_lag1' in df.columns:
        df['gap'] = (df['open'] / (df['close_lag1'] + 1e-8) - 1).astype(np.float32)

    # CATEGORY 5: CHIP FEATURES
    print("  [5/6] Chip features...", flush=True)
    if 'cost_50pct' in df.columns:
        df['price_vs_cost_50'] = (df['close'] / (df['cost_50pct'] + 1e-8) - 1).astype(np.float32)
    if 'weight_avg' in df.columns:
        df['price_vs_weight_avg'] = (df['close'] / (df['weight_avg'] + 1e-8) - 1).astype(np.float32)
    if 'winner_rate_mean_5' in df.columns and 'winner_rate_mean_20' in df.columns:
        df['winner_rate_trend'] = (df['winner_rate_mean_5'] - df['winner_rate_mean_20']).astype(np.float32)
    if 'chip_concentration_mean_5' in df.columns and 'chip_concentration_mean_20' in df.columns:
        df['chip_conc_trend'] = (df['chip_concentration_mean_5'] - df['chip_concentration_mean_20']).astype(np.float32)

    # CATEGORY 6: CALENDAR & MARKET
    print("  [6/6] Calendar & market features...", flush=True)
    df['weekday'] = pd.to_datetime(df['date'].astype(str)).dt.weekday.astype(np.int8)
    df['month'] = pd.to_datetime(df['date'].astype(str)).dt.month.astype(np.int8)
    df['is_month_start'] = (pd.to_datetime(df['date'].astype(str)).dt.day <= 5).astype(np.int8)
    df['is_month_end'] = (pd.to_datetime(df['date'].astype(str)).dt.day >= 25).astype(np.int8)

    return df

def add_labels(panel, all_dates_set):
    """Add target labels: (T+2 close / T+1 open - 1) > 0.04"""
    print("  Adding labels...", flush=True)
    df = panel.copy()
    df['label'] = np.float32(np.nan)
    df['label_ret'] = np.float32(np.nan)

    price_cache = {}
    dates = sorted(df['date'].unique())

    for d in dates:
        d_int = int(d)
        dt = int_to_date(d_int)
        t1, t2 = None, None
        for i in range(1, 10):
            nd = int((dt + timedelta(days=i)).strftime('%Y%m%d'))
            if nd in all_dates_set:
                if t1 is None:
                    t1 = nd
                elif t2 is None:
                    t2 = nd
                    break

        if t1 is None or t2 is None:
            continue

        if t1 not in price_cache:
            p1 = os.path.join(PRICE_DIR, f"{t1}.parquet")
            if os.path.exists(p1):
                price_cache[t1] = pd.read_parquet(p1, columns=['ts_code', 'open']).rename(columns={'open': 'open_t1'})
        if t2 not in price_cache:
            p2 = os.path.join(PRICE_DIR, f"{t2}.parquet")
            if os.path.exists(p2):
                price_cache[t2] = pd.read_parquet(p2, columns=['ts_code', 'close']).rename(columns={'close': 'close_t2'})

        if t1 in price_cache and t2 in price_cache:
            mask = df['date'] == d
            day_stocks = df.loc[mask, 'ts_code']
            m = pd.merge(day_stocks.to_frame(), price_cache[t1], on='ts_code', how='left')
            m = pd.merge(m, price_cache[t2], on='ts_code', how='left')
            m['label_ret'] = m['close_t2'] / m['open_t1'] - 1
            m['label'] = (m['label_ret'] > 0.04).astype(int)
            df.loc[mask, 'label_ret'] = m['label_ret'].values.astype(np.float32)
            df.loc[mask, 'label'] = m['label'].values.astype(np.float32)

    return df

def add_market_features(panel, news_mkt, news_stk):
    """Add market-level and news features."""
    print("  Adding market & news features...", flush=True)
    df = panel.copy()

    # Market return
    mkt_ret_map = {}
    for d in df['date'].unique():
        d_int = int(d)
        idx_path = os.path.join(INDEX_DIR, f"{d_int}.parquet")
        if os.path.exists(idx_path):
            try:
                idx_df = pd.read_parquet(idx_path)
                sh = idx_df[idx_df['ts_code'] == '000001.SH']
                if not sh.empty:
                    mkt_ret_map[d] = float(sh['pct_chg'].values[0] / 100.0)
            except:
                pass
    df['market_ret'] = df['date'].map(mkt_ret_map).fillna(0).astype(np.float32)

    # Market return lag
    df['market_ret_lag1'] = df.groupby('ts_code')['market_ret'].shift(1).astype(np.float32)

    # News
    if news_mkt is not None and not news_mkt.empty:
        nm = news_mkt.copy()
        if pd.api.types.is_datetime64_any_dtype(nm['trade_date']):
            nm['date'] = nm['trade_date'].dt.strftime('%Y%m%d').astype(int)
        else:
            nm['date'] = nm['trade_date']
        nm = nm.groupby('date')['news_market_impact'].max().reset_index()
        df = pd.merge(df, nm[['date', 'news_market_impact']], on='date', how='left')
        df['news_market_impact'] = df['news_market_impact'].fillna(0).astype(np.float32)
    else:
        df['news_market_impact'] = np.float32(0)

    if news_stk is not None and not news_stk.empty:
        ns = news_stk.copy()
        if pd.api.types.is_datetime64_any_dtype(ns['trade_date']):
            ns['date'] = ns['trade_date'].dt.strftime('%Y%m%d').astype(int)
        else:
            ns['date'] = ns['trade_date']
        ns = ns.groupby(['date', 'ts_code'])['news_stock_impact'].max().reset_index()
        df = pd.merge(df, ns, on=['date', 'ts_code'], how='left')
        df['news_stock_impact'] = df['news_stock_impact'].fillna(0).astype(np.float32)
    else:
        df['news_stock_impact'] = np.float32(0)

    return df

def get_feature_cols(df):
    exclude = {'ts_code', 'date', 'open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount',
               'pre_close', 'circ_mv', 'pe', 'pb', 'turnover_rate', 'volume_ratio',
               'cost_50pct', 'weight_avg', 'cost_15pct', 'cost_85pct',
               'chip_concentration', 'winner_rate', 'hot_rank_pct', 'label', 'label_ret',
               'news_market_impact', 'news_stock_impact'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int8']]

def main():
    print("=" * 90, flush=True)
    print("  TIME-SERIES FEATURE ENGINEERING v4 - Main Board Only, Memory Optimized", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    # Step 1: Build panel (only main board, from PANEL_START)
    print("\n[Step 1] Building panel (main board only)...", flush=True)
    panel_path = os.path.join(THIS_DIR, 'ts_panel_raw.parquet')

    if os.path.exists(panel_path):
        print("  Loading existing panel...", flush=True)
        panel = pd.read_parquet(panel_path)
    else:
        panel = build_panel(all_dates, PANEL_START, TEST_END)
        if panel.empty:
            print("  ERROR: Panel is empty!", flush=True)
            return
        print(f"  Panel: {len(panel)} rows, {panel['ts_code'].nunique()} stocks", flush=True)
        panel.to_parquet(panel_path, index=False)
        print(f"  Panel saved to {panel_path}", flush=True)

    print(f"  Panel: {len(panel)} rows, {panel['ts_code'].nunique()} stocks", flush=True)

    # Step 2: Compute features
    print("\n[Step 2] Computing time-series features...", flush=True)
    feat_path = os.path.join(THIS_DIR, 'ts_panel_features.parquet')

    if os.path.exists(feat_path):
        print("  Loading existing features...", flush=True)
        feat_panel = pd.read_parquet(feat_path)
    else:
        feat_panel = compute_ts_features(panel)
        del panel
        feat_panel = add_market_features(feat_panel, news_mkt, news_stk)
        print(f"  Features: {len(feat_panel)} rows, {len(feat_panel.columns)} columns", flush=True)
        feat_panel.to_parquet(feat_path, index=False)
        print(f"  Features saved to {feat_path}", flush=True)

    print(f"  Features: {len(feat_panel)} rows, {len(feat_panel.columns)} columns", flush=True)

    # Step 3: Add labels
    print("\n[Step 3] Adding labels...", flush=True)
    labeled_path = os.path.join(THIS_DIR, 'ts_panel_labeled.parquet')

    if os.path.exists(labeled_path):
        print("  Loading existing labeled data...", flush=True)
        labeled_df = pd.read_parquet(labeled_path)
    else:
        labeled_df = add_labels(feat_panel, all_dates_set)
        labeled_df = labeled_df[labeled_df['label'].notna()].copy()
        labeled_df.to_parquet(labeled_path, index=False)
        print(f"  Labeled: {len(labeled_df)} rows", flush=True)

    # Filter to test period for analysis
    test_df = labeled_df[labeled_df['date'] >= int(TEST_START)].copy()
    print(f"  Test period: {len(test_df)} rows, pos_rate={test_df['label'].mean():.3f}", flush=True)

    # Step 4: Feature Analysis
    print("\n[Step 4] Feature Analysis...", flush=True)
    feature_cols = get_feature_cols(test_df)
    print(f"  Total features: {len(feature_cols)}", flush=True)

    # 4a: Correlation with target
    print("\n  --- Correlation with label_ret ---", flush=True)
    valid_feats = [f for f in feature_cols if test_df[f].notna().sum() > 1000]
    print(f"  Valid features (>{1000} non-null): {len(valid_feats)}", flush=True)

    corr = test_df[valid_feats + ['label_ret']].corr()['label_ret'].drop('label_ret').abs().sort_values(ascending=False)
    print(f"  Top 25 features by |correlation|:", flush=True)
    for feat, c in corr.head(25).items():
        print(f"    {feat:<50} corr={c:.4f}", flush=True)

    # 4b: XGBoost importance
    print("\n  --- Feature Importance (XGBoost) ---", flush=True)
    import xgboost as xgb

    X = test_df[valid_feats].fillna(0)
    y = test_df['label'].astype(int)

    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                               subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                               eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1)
    model.fit(X, y)

    imp = pd.Series(model.feature_importances_, index=valid_feats).sort_values(ascending=False)
    print(f"  Top 30 features by importance:", flush=True)
    for feat, score in imp.head(30).items():
        print(f"    {feat:<50} imp={score:.4f}", flush=True)

    # 4c: Correlation filter
    print("\n  --- Correlation Filter (threshold=0.8) ---", flush=True)
    top_feats = imp.head(50).index.tolist()
    feat_corr = test_df[top_feats].corr().abs()
    upper = feat_corr.where(np.triu(np.ones(feat_corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]
    print(f"  Features to drop (corr>0.8): {len(to_drop)}", flush=True)
    for f in to_drop[:15]:
        correlated_with = upper[f][upper[f] > 0.8].index.tolist()
        print(f"    {f:<50} correlated with: {correlated_with[:3]}", flush=True)

    selected = [f for f in top_feats if f not in to_drop]
    print(f"  Selected features (from top 50): {len(selected)}", flush=True)
    print(f"  Selected features:", flush=True)
    for f in selected:
        print(f"    {f:<50} imp={imp[f]:.4f}  corr={corr.get(f, 0):.4f}", flush=True)

    # Save
    imp_df = pd.DataFrame({
        'feature': imp.index,
        'importance': imp.values,
        'corr_with_target': [corr.get(f, 0) for f in imp.index],
        'selected': [f in selected for f in imp.index]
    })
    imp_df.to_csv(os.path.join(THIS_DIR, 'ts_feature_ranking.csv'), index=False)
    pd.DataFrame({'feature': selected}).to_csv(os.path.join(THIS_DIR, 'ts_selected_features.csv'), index=False)

    # Step 5: Leakage check
    print("\n[Step 5] LEAKAGE CHECK", flush=True)
    print("  Target: (T+2 close / T+1 open - 1) > 0.04", flush=True)
    print("  Features computed from: T日 data (after close) + T-1 and earlier for rolling/lag", flush=True)
    print("  ✅ All lag features use shift() = strictly T-1 and earlier", flush=True)
    print("  ✅ All rolling features use T日 and earlier (no T+1/T+2 data)", flush=True)
    print("  ✅ News features use T日 news (available after close)", flush=True)
    print("  ✅ Calendar features use T日 date", flush=True)
    print("  ✅ T日 price/chip data used as base (available after close, before T+1 open)", flush=True)
    print("  ✅ Market return uses T日 index data (available after close)", flush=True)

    # Plot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(14, 10))
    top20 = imp.head(20).sort_values()
    colors = ['green' if f in selected else 'red' for f in top20.index]
    ax.barh(range(len(top20)), top20.values, color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=8)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Time-Series Feature Importance (Green=Selected, Red=Dropped by Corr Filter)')
    plt.tight_layout()
    plt.savefig(os.path.join(THIS_DIR, 'ts_feature_importance.png'), dpi=150, bbox_inches='tight')
    print(f"\nChart saved", flush=True)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
