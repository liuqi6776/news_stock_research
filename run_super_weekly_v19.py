"""
Super-Weekly V19: Enhanced with Multi-Source Features
=====================================================
Base: super_weekly (313% return, best version)
New Features:
1. moneyflow1 - 主力资金净流入 (大单/超大单净流入, 连续流入天数)
2. ths_rank1 - 个股热度排名 (热度值, 热度变化率)
3. money1 - 北向资金 (净买入额, 5日/20日均值变化)
4. skill1 - 技术指标 (RSI, MACD, KDJ, BOLL偏离度)
5. vix1 - 恐慌指数 (当日值, 5日变化率)
6. margin1 - 融资融券 (融资余额变化)
Memory: sampling 10k + gc after each train
Progress: real-time to v19_progress.log
"""
import os
import sys
import time
import gc
import warnings
import pandas as pd
import numpy as np
import tushare as ts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
import joblib

warnings.filterwarnings('ignore')

TUSHARE_TOKEN = '421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa'
DATA_DIR      = r'D:\iquant_data\data_v2\data_day1'
BASIC_DIR     = r'D:\iquant_data\data_v2\other_day1'
CHIP_DIR      = r'D:\iquant_data\data_v2\cyq1'
VIX_DIR       = r'D:\iquant_data\data_v2\vix1'
MARGIN_DIR    = r'D:\iquant_data\data_v2\margin1'
MONEYFLOW_DIR = r'D:\iquant_data\data_v2\moneyflow1'
THS_RANK_DIR  = r'D:\iquant_data\data_v2\ths_rank1'
MONEY_DIR     = r'D:\iquant_data\data_v2\money1'
SKILL_DIR     = r'D:\iquant_data\data_v2\skill1'
OUT_DIR       = r'C:\Users\liuqi\quant_system_v2'

INITIAL_CAP   = 100_000.0
TOP_N         = 3
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005
SLIPPAGE      = 0.001
REBAL_FREQ    = 5

PROGRESS_LOG  = os.path.join(OUT_DIR, 'v19_progress.log')


def plog(msg):
    """Write progress log."""
    with open(PROGRESS_LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)


# ==================== Data Loaders ====================

def load_market_data(start, end):
    """Load price + basic + chip data (same as super_weekly)."""
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    for ds in tqdm(date_strs, desc="Load market"):
        if not (start <= ds <= end):
            continue
        try:
            p = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}.parquet"),
                                columns=['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'pre_close', 'vol'])
            b = pd.read_parquet(os.path.join(BASIC_DIR, f"{ds}.parquet"),
                                columns=['ts_code', 'pe', 'pb', 'circ_mv'])
            chip_path = os.path.join(CHIP_DIR, f"{ds}.parquet")
            if os.path.exists(chip_path):
                c = pd.read_parquet(chip_path, columns=['ts_code', 'winner_rate', 'cost_15pct', 'cost_50pct', 'cost_85pct'])
            else:
                c = pd.DataFrame(columns=['ts_code', 'winner_rate', 'cost_15pct', 'cost_50pct', 'cost_85pct'])
            files.append(pd.merge(pd.merge(p, b, on='ts_code'), c, on='ts_code', how='left'))
        except:
            continue
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    return df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)


def load_moneyflow(start, end):
    """Load moneyflow - only aggregate stats to save memory."""
    # Per-stock daily: net_mf_amount, net_lg_amount, net_elg_amount
    # Instead of loading all rows, we only keep the merged result per day
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(MONEYFLOW_DIR) if f.endswith('.parquet')])
    dfs = []
    for ds in date_strs:
        if not (start <= ds <= end):
            continue
        try:
            df = pd.read_parquet(os.path.join(MONEYFLOW_DIR, f"{ds}.parquet"),
                                 columns=['ts_code', 'buy_lg_amount', 'sell_lg_amount',
                                          'buy_elg_amount', 'sell_elg_amount', 'net_mf_amount',
                                          'buy_md_amount', 'sell_md_amount'])
            df['trade_date'] = pd.to_datetime(ds)
            df['net_lg_amount'] = df['buy_lg_amount'] - df['sell_lg_amount']
            df['net_elg_amount'] = df['buy_elg_amount'] - df['sell_elg_amount']
            df['net_md_amount'] = df['buy_md_amount'] - df['sell_md_amount']
            dfs.append(df[['ts_code', 'trade_date', 'net_mf_amount', 'net_lg_amount', 'net_elg_amount', 'net_md_amount']])
        except:
            continue
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        return result
    return pd.DataFrame(columns=['ts_code', 'trade_date', 'net_mf_amount', 'net_lg_amount', 'net_elg_amount', 'net_md_amount'])


def load_ths_rank(start, end):
    """Load THS stock heat ranking (TOP 100)."""
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(THS_RANK_DIR) if f.endswith('.parquet')])
    for ds in date_strs:
        if not (start <= ds <= end):
            continue
        try:
            df = pd.read_parquet(os.path.join(THS_RANK_DIR, f"{ds}.parquet"),
                                 columns=['ts_code', 'hot'])
            df['trade_date'] = ds
            files.append(df)
        except:
            continue
    if files:
        df = pd.concat(files, ignore_index=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df[['ts_code', 'trade_date', 'hot']]
    return pd.DataFrame(columns=['ts_code', 'trade_date', 'hot'])


def load_north_money(start, end):
    """Load north-bound money flow (market-level)."""
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(MONEY_DIR) if f.endswith('.parquet')])
    for ds in date_strs:
        if not (start <= ds <= end):
            continue
        try:
            df = pd.read_parquet(os.path.join(MONEY_DIR, f"{ds}.parquet"),
                                 columns=['trade_date', 'north_money'])
            df['trade_date'] = ds
            files.append(df)
        except:
            continue
    if files:
        df = pd.concat(files, ignore_index=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['north_5d'] = df['north_money'].rolling(5).mean()
        df['north_20d'] = df['north_money'].rolling(20).mean()
        df['north_change_5d'] = (df['north_money'] - df['north_5d']) / (df['north_5d'].abs() + 1e-8)
        return df[['trade_date', 'north_money', 'north_5d', 'north_20d', 'north_change_5d']]
    return pd.DataFrame(columns=['trade_date', 'north_money', 'north_5d', 'north_20d', 'north_change_5d'])


def load_skill(start, end):
    """Load technical indicators."""
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(SKILL_DIR) if f.endswith('.parquet')])
    dfs = []
    for ds in date_strs:
        if not (start <= ds <= end):
            continue
        try:
            df = pd.read_parquet(os.path.join(SKILL_DIR, f"{ds}.parquet"),
                                 columns=['ts_code', 'rsi_6', 'rsi_12', 'macd',
                                          'kdj_k', 'kdj_j', 'boll_upper', 'boll_mid', 'boll_lower'])
            df['trade_date'] = pd.to_datetime(ds)
            boll_range = (df['boll_upper'] - df['boll_lower'])
            df['boll_dev'] = (df['boll_mid'] - df['boll_lower']) / (boll_range + 1e-8)
            df['macd_signal'] = np.where(df['macd'] > 0, 1, -1)
            df['rsi_signal'] = np.where(df['rsi_6'] < 30, 1, np.where(df['rsi_6'] > 70, -1, 0))
            dfs.append(df[['ts_code', 'trade_date', 'rsi_6', 'rsi_12', 'macd', 'macd_signal',
                           'kdj_k', 'kdj_j', 'boll_dev', 'rsi_signal']])
        except:
            continue
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        return result
    return pd.DataFrame(columns=['ts_code', 'trade_date', 'rsi_6', 'rsi_12', 'macd', 'macd_signal',
                                 'kdj_k', 'kdj_j', 'boll_dev', 'rsi_signal'])


def load_vix(start, end):
    """Load VIX data."""
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(VIX_DIR) if f.endswith('.parquet')])
    for ds in date_strs:
        if not (start <= ds <= end):
            continue
        try:
            df = pd.read_parquet(os.path.join(VIX_DIR, f"{ds}.parquet"),
                                 columns=['ts_code', 'trade_date', 'close', 'pct_chg'])
            df['trade_date'] = ds
            files.append(df)
        except:
            continue
    if files:
        df = pd.concat(files, ignore_index=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        # Use 000001.SH as market VIX
        idx = df[df['ts_code'] == '000001.SH'].copy()
        if len(idx) > 0:
            idx = idx.sort_values('trade_date')
            idx = idx.rename(columns={'close': 'vix_close', 'pct_chg': 'vix_pct_chg'})
            idx['vix_5d_chg'] = idx['vix_close'].pct_change(5)
            return idx[['trade_date', 'vix_close', 'vix_pct_chg', 'vix_5d_chg']]
    return pd.DataFrame(columns=['trade_date', 'vix_close', 'vix_pct_chg', 'vix_5d_chg'])


# ==================== Feature Engineering ====================

def build_all_features(df, mf_df, rank_df, north_df, skill_df, vix_df):
    """Build original + new features."""
    plog("Building features...")
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')

    # === Original super_weekly features ===
    gc = g['close']
    for w in [5, 20]:
        df[f'mom_{w}'] = gc.transform(lambda x: x / x.shift(w) - 1)
        df[f'bias_{w}'] = (df['close'] - gc.transform(lambda x: x.rolling(w).mean())) / (gc.transform(lambda x: x.rolling(w).mean()) + 1e-8)
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    df['chip_score'] = df['winner_rate'] * (df['close'] > df['cost_50pct']).astype(int)
    df['chip_bottom_heavy'] = (df['cost_85pct'] - df['cost_50pct']) / (df['cost_50pct'] - df['cost_15pct'] + 1e-8)

    # === NEW: Volatility ===
    ret = df.groupby('ts_code')['close'].pct_change()
    df['vol_20'] = ret.groupby(df['ts_code']).transform(lambda x: x.rolling(20).std())
    df['vol_ratio'] = df['vol'] / (df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5).mean()) + 1e-8)

    # === NEW: Money Flow features ===
    if not mf_df.empty:
        df = df.merge(mf_df, on=['ts_code', 'trade_date'], how='left')
        # Consecutive inflow days
        df['mf_positive'] = (df['net_mf_amount'] > 0).astype(int)
        df['mf_consec_inflow'] = df.groupby('ts_code')['mf_positive'].apply(
            lambda x: x.groupby((x == 0).cumsum()).cumcount()
        ).reset_index(level=0, drop=True)
        # Cross-sectional rank
        for col in ['net_mf_amount', 'net_lg_amount', 'net_elg_amount', 'mf_consec_inflow']:
            if col in df.columns:
                df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    else:
        plog("  WARNING: moneyflow data empty, skipping")

    # === NEW: THS Rank (stock heat) ===
    if not rank_df.empty:
        df = df.merge(rank_df, on=['ts_code', 'trade_date'], how='left')
        df['hot'] = df['hot'].fillna(0)
        # Log-transform hot (very skewed)
        df['log_hot'] = np.log(df['hot'] + 1)
        # Cross-sectional rank
        df['hot_rank'] = df.groupby('trade_date')['hot'].rank(pct=True)
        # Heat change (need lag)
        df['hot_5d_ago'] = df.groupby('ts_code')['hot'].shift(5)
        df['hot_change_5d'] = (df['hot'] - df['hot_5d_ago']) / (df['hot_5d_ago'].abs() + 1e-8)
    else:
        plog("  WARNING: ths_rank data empty, skipping")

    # === NEW: North-bound money (market-level, broadcast) ===
    if not north_df.empty:
        north_map = north_df.set_index('trade_date')[['north_money', 'north_5d', 'north_20d', 'north_change_5d']].to_dict('index')
        for col in ['north_money', 'north_5d', 'north_20d', 'north_change_5d']:
            df[col] = df['trade_date'].map(lambda d: north_map.get(d, {}).get(col, 0))
    else:
        plog("  WARNING: north money data empty, skipping")

    # === NEW: Technical indicators ===
    if not skill_df.empty:
        df = df.merge(skill_df, on=['ts_code', 'trade_date'], how='left')
        for col in ['rsi_6', 'rsi_12', 'kdj_k', 'boll_dev', 'macd_signal']:
            if col in df.columns:
                df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    else:
        plog("  WARNING: skill data empty, skipping")

    # === NEW: VIX (market-level) ===
    if not vix_df.empty:
        vix_map = vix_df.set_index('trade_date')[['vix_close', 'vix_pct_chg', 'vix_5d_chg']].to_dict('index')
        for col in ['vix_close', 'vix_pct_chg', 'vix_5d_chg']:
            df[col] = df['trade_date'].map(lambda d: vix_map.get(d, {}).get(col, 0))
    else:
        plog("  WARNING: vix data empty, skipping")

    # === Cross-sectional ranking for original features ===
    for col in ['mom_5', 'mom_20', 'bias_5', 'ep', 'bp', 'chip_score', 'chip_bottom_heavy']:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)

    plog("  Feature engineering done")
    return df


def add_labels(df, horizon=5):
    df = df.sort_values(['ts_code', 'trade_date'])
    entry = df.groupby('ts_code')['open'].shift(-1)
    exit_ = df.groupby('ts_code')['open'].shift(-1 - horizon)
    df['ret'] = (exit_ - entry) / (entry + 1e-8)
    df['label'] = (df['ret'] > 0.02).astype(int)
    label_counts = df['label'].value_counts()
    plog(f"  Label: bull(>2%)={label_counts.get(1, 0)}, bear/flat={label_counts.get(0, 0)}")
    return df


# ==================== Feature Columns ====================

ORIGINAL_FEATURES = [
    'mom_5', 'mom_20', 'bias_5', 'ep', 'bp', 'log_mv',
    'chip_score', 'chip_bottom_heavy',
    'mom_5_rank', 'mom_20_rank', 'ep_rank', 'bp_rank',
    'chip_score_rank', 'chip_bottom_heavy_rank',
]

NEW_FEATURES = [
    # Money flow
    'net_mf_amount_rank', 'net_lg_amount_rank', 'net_elg_amount_rank', 'mf_consec_inflow_rank',
    # Stock heat
    'hot_rank', 'hot_change_5d',
    # Technical
    'rsi_6_rank', 'rsi_12_rank', 'kdj_k_rank', 'boll_dev_rank', 'macd_signal',
    # Volatility
    'vol_20', 'vol_ratio',
    # North money (market level)
    'north_change_5d',
    # VIX (market level)
    'vix_5d_chg',
]

ALL_FEATURES = ORIGINAL_FEATURES + NEW_FEATURES


# ==================== Model ====================

def train_model(train_df):
    valid_cols = [c for c in ALL_FEATURES if c in train_df.columns]
    sub = train_df.dropna(subset=valid_cols + ['label']).copy()
    X = sub[valid_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    pos = sub[y == 1].sample(min(len(sub[y == 1]), 10000), random_state=42)
    neg = sub[y == 0].sample(min(len(sub[y == 0]), 10000), random_state=42)
    bal = pd.concat([pos, neg])
    scaler = RobustScaler()
    X_bal = scaler.fit_transform(bal[valid_cols])
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                              subsample=0.8, eval_metric='logloss', random_state=42, n_jobs=1)
    model.fit(X_bal, bal['label'], verbose=False)
    del bal, X_bal, sub
    gc.collect()
    return model, scaler, valid_cols


# ==================== Backtest ====================

def run_backtest(df):
    plog("Starting backtest...")
    print("\n" + "!" * 60)
    print("  Super-Weekly V19: Enhanced Multi-Source Features")
    print("!" * 60)

    test_dates = sorted(df[df['trade_date'] >= '2023-01-01']['trade_date'].unique())
    rebal_dates = test_dates[::REBAL_FREQ]
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'pre_close']].to_dict('index')

    valid_cols = [c for c in ALL_FEATURES if c in df.columns]
    plog(f"Valid features: {len(valid_cols)}")
    print(f"  Valid features: {len(valid_cols)}")

    start_time = time.time()
    capital = INITIAL_CAP
    holdings = []
    equity = []
    cur_model, cur_scaler = None, None
    last_month = None
    total_rebals = len(rebal_dates[:-1])

    for i, date in enumerate(tqdm(rebal_dates[:-1])):
        d_signal = date
        d_trade = test_dates[test_dates.index(date) + 1]

        # Monthly model retrain
        month = date.month
        if month != last_month:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365 * 3))]
            if len(train_data) > 500:
                cur_model, cur_scaler, _ = train_model(train_data)
                plog(f"{date.strftime('%Y-%m')}: model trained ({len(train_data)} rows)")
                print(f"  {date.strftime('%Y-%m')}: model trained ({len(train_data)} rows)")
                del train_data
                gc.collect()
            last_month = month

        # Sell all
        for pos in list(holdings):
            key_sell = (d_trade, pos['ts_code'])
            if key_sell in prices:
                px_sell = prices[key_sell]
                down_limit = get_limit_price(pos['ts_code'], px_sell['pre_close'], 'down')
                if px_sell['open'] <= down_limit:
                    continue
                exit_px = px_sell['open'] * (1 - SLIPPAGE)
                revenue = pos['shares'] * exit_px
                capital += (revenue - max(5, revenue * COMMISSION) - revenue * STAMP_DUTY)
                holdings.remove(pos)

        # Buy
        day_data = df[df['trade_date'] == d_signal].dropna(subset=valid_cols)
        if cur_model and not day_data.empty:
            X = cur_scaler.transform(day_data[valid_cols].fillna(0))
            day_data['prob'] = cur_model.predict_proba(X)[:, 1]
            picks = day_data.sort_values('prob', ascending=False).head(TOP_N)

            if not picks.empty:
                cash_per = capital / TOP_N
                for _, row in picks.iterrows():
                    key_buy = (d_trade, row['ts_code'])
                    if key_buy in prices:
                        px_buy = prices[key_buy]
                        up_limit = get_limit_price(row['ts_code'], px_buy['pre_close'], 'up')
                        if px_buy['open'] >= up_limit:
                            continue
                        buy_px = px_buy['open'] * (1 + SLIPPAGE)
                        shares = int(cash_per / buy_px / 100) * 100
                        if shares >= 100:
                            capital -= (shares * buy_px + max(5, shares * buy_px * COMMISSION))
                            holdings.append({'ts_code': row['ts_code'], 'shares': shares, 'buy_px': buy_px})

        mv = sum(p['shares'] * prices.get((d_trade, p['ts_code']), {'close': p['buy_px']})['close']
                 for p in holdings)
        nav = capital + mv
        equity.append({'date': d_trade, 'nav': nav})

        # Progress log every 5 iterations
        if (i + 1) % 5 == 0 or i == total_rebals - 1:
            pct = (i + 1) / total_rebals * 100
            elapsed = time.time() - start_time
            plog(f"{i + 1}/{total_rebals} ({pct:.1f}%) | {d_signal.strftime('%Y-%m-%d')} | NAV={nav:.0f} | holdings={len(holdings)} | {elapsed:.0f}s")

    plog(f"=== V19 DONE in {time.time() - start_time:.0f}s ===")
    return pd.DataFrame(equity)


# ==================== Main ====================

if __name__ == "__main__":
    # Init progress log
    with open(PROGRESS_LOG, 'w', encoding='utf-8') as f:
        f.write(f"V19 Progress - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

    plog("Loading market data...")
    df = load_market_data('20200101', '20260101')
    plog(f"Market data: {len(df)} rows")

    plog("Loading moneyflow data...")
    mf_df = load_moneyflow('20200101', '20260101')
    plog(f"Moneyflow: {len(mf_df)} rows")

    plog("Loading THS rank data...")
    rank_df = load_ths_rank('20200101', '20260101')
    plog(f"THS rank: {len(rank_df)} rows")

    plog("Loading north money data...")
    north_df = load_north_money('20200101', '20260101')
    plog(f"North money: {len(north_df)} rows")

    plog("Loading skill data...")
    skill_df = load_skill('20200101', '20260101')
    plog(f"Skill: {len(skill_df)} rows")

    plog("Loading VIX data...")
    vix_df = load_vix('20200101', '20260101')
    plog(f"VIX: {len(vix_df)} rows")

    # Free raw loaders
    gc.collect()

    plog("Merging and building features...")
    df = build_all_features(df, mf_df, rank_df, north_df, skill_df, vix_df)
    del mf_df, rank_df, north_df, skill_df, vix_df
    gc.collect()
    plog(f"After merge + features: {len(df)} rows, {len(df.columns)} cols")
    gc.collect()

    df = add_labels(df, horizon=5)
    df = df.drop_duplicates(subset=['ts_code', 'trade_date']).reset_index(drop=True)

    eq_df = run_backtest(df)

    # Performance metrics
    total_ret = (eq_df['nav'].iloc[-1] / INITIAL_CAP - 1) * 100
    days = (eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days
    annual_ret = ((eq_df['nav'].iloc[-1] / INITIAL_CAP) ** (365 / days) - 1) * 100
    peak = eq_df['nav'].cummax()
    drawdown = (eq_df['nav'] - peak) / peak
    max_dd = drawdown.min() * 100
    daily_ret = eq_df['nav'].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (daily_ret > 0).sum() / len(daily_ret) * 100 if len(daily_ret) > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  V19 Performance (vs super_weekly: +313.6%)")
    print(f"{'=' * 60}")
    print(f"  Total Return: {total_ret:+.2f}%")
    print(f"  Annual Return: {annual_ret:+.2f}%")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Calmar: {calmar:.2f}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Days: {days}")
    print(f"{'=' * 60}")

    # Yearly returns
    eq_df['year'] = eq_df['date'].dt.year
    yearly = eq_df.groupby('year').apply(lambda x: (x['nav'].iloc[-1] / x['nav'].iloc[0] - 1) * 100)
    print(f"\n  Yearly Returns:")
    for y, ret in yearly.items():
        print(f"    {y}: {ret:+.1f}%")

    plog(f"RESULT: Total={total_ret:.1f}%, Annual={annual_ret:.1f}%, MaxDD={max_dd:.1f}%, Sharpe={sharpe:.2f}, Calmar={calmar:.2f}")
    for y, ret in yearly.items():
        plog(f"  {y}: {ret:+.1f}%")

    # Save equity curve
    eq_df[['date', 'nav']].to_csv(os.path.join(OUT_DIR, 'super_weekly_v19_equity.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1 = axes[0, 0]
    ax1.plot(eq_df['date'], eq_df['nav'], linewidth=1.5, color='#E91E63', label='V19 (Multi-Source)')
    ax1.set_title('V19 Equity Curve', fontsize=13, fontweight='bold')
    ax1.set_ylabel('NAV')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.fill_between(eq_df['date'], drawdown * 100, 0, alpha=0.4, color='red')
    ax2.set_title('Drawdown (%)', fontsize=12)
    ax2.set_ylabel('Drawdown %')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(eq_df['date'], eq_df['nav'], linewidth=1.5, color='black')
    ax3.set_title('NAV Curve', fontsize=10)
    ax3.set_ylabel('NAV')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in yearly]
    ax4.bar(yearly.index, yearly.values, color=colors, alpha=0.8)
    ax4.set_title('Yearly Returns (%)', fontsize=12)
    ax4.set_ylabel('Return %')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    for y, v in yearly.items():
        ax4.text(y, v + (2 if v > 0 else -5), f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'weekly_v19_curve.png'), dpi=150)
    print(f"\n  Saved: weekly_v19_curve.png, super_weekly_v19_equity.csv")
