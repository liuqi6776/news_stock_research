"""
Step 1: 构建全量特征数据

输入: D:\\iquant_data\\data_v2\\ 下的原始数据
输出: data/all_features_v2.parquet

耗时: 约2-4小时 (取决于数据量)
运行频率: 每月1次 或 有新数据时
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from shared.data_loader import get_all_dates, PRICE_DIR

from config import (
    RAW_PRICE_DIR, RAW_OTHER_DIR, RAW_NEWS_DIR, RAW_RANK_DIR,
    RAW_MONEYFLOW_DIR, RAW_THS_NEWS_DIR, RAW_INCOME_DIR,
    TRAIN_START, MIN_LISTING_DAYS, TARGET_RETURN_THRESHOLD,
    TARGET_HORIZON_DAYS, FEATURES_FILE, DATA_DIR
)


def is_main_board(ts_code: str) -> bool:
    return ts_code.startswith(('60', '00', '002', '003'))


def build_listing_date_map():
    first_seen = {}
    all_dates = get_all_dates()
    for d in tqdm(all_dates, desc="构建上市日期映射"):
        p = os.path.join(RAW_PRICE_DIR, f"{d}.parquet")
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_parquet(p, columns=['ts_code'])
            for code in df['ts_code'].unique():
                if code not in first_seen:
                    first_seen[code] = d
        except:
            pass
    return first_seen, all_dates


def calc_price_features(df):
    df = df.copy()
    df['pct_chg'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['amplitude'] = (df['high'] - df['low']) / df['pre_close']
    df['body_size'] = abs(df['close'] - df['open']) / df['pre_close']
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['pre_close']
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['pre_close']
    df['is_yang'] = (df['close'] > df['open']).astype(int)
    df['gap'] = (df['open'] - df['pre_close']) / df['pre_close']
    df['close_to_high'] = (df['high'] - df['close']) / df['pre_close']
    df['close_to_low'] = (df['close'] - df['low']) / df['pre_close']
    return df


def calc_volume_features(df, hist_vol_mean=None):
    df = df.copy()
    if hist_vol_mean:
        df['hist_vol_mean'] = df['ts_code'].map(hist_vol_mean)
        df['vol_ratio'] = df['vol'] / (df['hist_vol_mean'] + 1e-8)
    else:
        df['vol_ratio'] = 1.0
    df['vol_amount'] = df['close'] * df['vol']
    return df


def calc_momentum_features(df, price_hist):
    df = df.copy()
    for w in [5, 10, 20, 60]:
        mom_vals = []
        for _, row in df.iterrows():
            ts_code = row['ts_code']
            if ts_code in price_hist and len(price_hist[ts_code]) >= w:
                hist = price_hist[ts_code]
                mom_vals.append((row['close'] - hist[-w]) / hist[-w])
            else:
                mom_vals.append(np.nan)
        df[f'mom_{w}d'] = mom_vals
    return df


def calc_volatility_features(df, price_hist):
    df = df.copy()
    for w in [5, 10, 20]:
        vol_vals = []
        for _, row in df.iterrows():
            ts_code = row['ts_code']
            if ts_code in price_hist and len(price_hist[ts_code]) >= w + 1:
                hist = price_hist[ts_code]
                returns = [(hist[i] - hist[i-1]) / hist[i-1] for i in range(1, len(hist))]
                vol_vals.append(np.std(returns[-w:]) * np.sqrt(252))
            else:
                vol_vals.append(np.nan)
        df[f'vol_{w}d'] = vol_vals
    return df


def calc_technical_indicators(df, price_hist):
    df = df.copy()
    for _, row in df.iterrows():
        ts_code = row['ts_code']
        if ts_code not in price_hist or len(price_hist[ts_code]) < 20:
            continue
        hist = price_hist[ts_code]
        if len(hist) >= 15:
            prices = np.array(hist[-15:])
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                df.loc[df['ts_code'] == ts_code, 'rsi_14'] = 100 - (100 / (1 + rs))
        if len(hist) >= 30:
            prices = pd.Series(hist)
            ema12 = prices.ewm(span=12).mean().iloc[-1]
            ema26 = prices.ewm(span=26).mean().iloc[-1]
            macd = ema12 - ema26
            df.loc[df['ts_code'] == ts_code, 'macd'] = macd
        if len(hist) >= 10:
            prices = np.array(hist[-10:])
            lowest = np.min(prices)
            highest = np.max(prices)
            if highest > lowest:
                rsv = (prices[-1] - lowest) / (highest - lowest) * 100
                df.loc[df['ts_code'] == ts_code, 'kdj_k'] = rsv
        if len(hist) >= 20:
            prices = np.array(hist[-20:])
            ma20 = np.mean(prices)
            std20 = np.std(prices)
            df.loc[df['ts_code'] == ts_code, 'bb_position'] = (prices[-1] - ma20) / (2 * std20 + 1e-8)
        if len(hist) >= 15:
            prices = np.array(hist[-15:])
            atr = np.mean(np.abs(np.diff(prices)))
            df.loc[df['ts_code'] == ts_code, 'atr_14'] = atr / (prices[-1] + 1e-8)
    return df


def load_fundamental_features(date):
    p = os.path.join(RAW_OTHER_DIR, f"{date}.parquet")
    if os.path.exists(p):
        try:
            df = pd.read_parquet(p)
            cols = ['ts_code']
            for c in ['pe', 'pb', 'circ_mv', 'turnover_rate', 'volume_ratio']:
                if c in df.columns:
                    cols.append(c)
            if len(cols) > 1:
                result = df[cols].copy()
                if 'circ_mv' in result.columns:
                    result['log_circ_mv'] = np.log1p(result['circ_mv'])
                if 'pe' in result.columns:
                    result['pe'] = result['pe'].replace([np.inf, -np.inf], np.nan)
                    result['log_pe'] = np.log1p(result['pe'].abs())
                if 'pb' in result.columns:
                    result['pb'] = result['pb'].replace([np.inf, -np.inf], np.nan)
                    result['log_pb'] = np.log1p(result['pb'].abs())
                return result
        except:
            pass
    return pd.DataFrame()


def load_news_features(date):
    date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    p = os.path.join(RAW_NEWS_DIR, f"analysis_{date_fmt}.json")
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        import json
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rows = []
        market_impact = data.get('market_impact', 0)
        for stock in data.get('stocks', []):
            code = stock.get('stock_code', '')
            if len(code) == 6:
                ts_code = code + '.SH' if code.startswith('6') else code + '.SZ'
                rows.append({
                    'ts_code': ts_code,
                    'news_stock_impact': stock.get('impact', 0),
                    'news_market_impact': market_impact,
                    'news_has_mention': 1,
                })
        if rows:
            return pd.DataFrame(rows)
    except:
        pass
    return pd.DataFrame()


def load_rank_features(date):
    p = os.path.join(RAW_RANK_DIR, f"{date}.parquet")
    if os.path.exists(p):
        try:
            df = pd.read_parquet(p)
            if 'hot' in df.columns and len(df) > 0:
                result = df[['ts_code', 'hot']].copy()
                result = result.rename(columns={'hot': 'ths_hot'})
                result['ths_hot_rank'] = result['ths_hot'].rank(ascending=False, method='min')
                return result
        except:
            pass
    return pd.DataFrame()


def load_moneyflow_features(date):
    p = os.path.join(RAW_MONEYFLOW_DIR, f"{date}.parquet")
    if os.path.exists(p):
        try:
            df = pd.read_parquet(p)
            cols = ['ts_code']
            for c in ['net_mf_amount', 'net_mf_vol', 'buy_lg_amount', 'sell_lg_amount',
                       'buy_elg_amount', 'sell_elg_amount']:
                if c in df.columns:
                    cols.append(c)
            if len(cols) > 1:
                result = df[cols].copy()
                if 'net_mf_amount' in result.columns:
                    result['net_mf_amount_norm'] = result['net_mf_amount'] / (
                        result['net_mf_amount'].abs().quantile(0.95) + 1e-8)
                return result
        except:
            pass
    return pd.DataFrame()


def load_ths_news_features(date):
    p = os.path.join(RAW_THS_NEWS_DIR, f"{date}.parquet")
    if os.path.exists(p):
        try:
            df = pd.read_parquet(p)
            cols = ['ts_code']
            for c in ['new_gs', 'new_bs', 'new_gi']:
                if c in df.columns:
                    cols.append(c)
            if len(cols) > 1:
                return df[cols]
        except:
            pass
    return pd.DataFrame()


def run():
    print("=" * 80)
    print("Step 1: 构建全量特征数据")
    print("=" * 80)

    if os.path.exists(FEATURES_FILE):
        existing = pd.read_parquet(FEATURES_FILE)
        existing_max_date = existing['trade_date'].astype(str).max()
        print(f"已有特征数据: {len(existing)} 行, 最新日期: {existing_max_date}")
        all_dates = get_all_dates()
        raw_max_date = all_dates[-1]
        if existing_max_date >= raw_max_date:
            print("特征数据已是最新，跳过")
            return existing

    first_seen, all_dates_raw = build_listing_date_map()
    print(f"上市日期映射: {len(first_seen)} 只股票")

    all_dates = get_all_dates()
    date_idx = {d: i for i, d in enumerate(all_dates)}
    available_dates = [d for d in all_dates if d >= TRAIN_START]
    print(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    print(f"总交易日: {len(available_dates)}")

    price_hist = {}
    vol_hist = {}
    hist_vol_mean = {}
    training_data = []

    for i in tqdm(range(len(available_dates) - 30), desc="准备数据"):
        d_curr = available_dates[i]

        p_curr = os.path.join(RAW_PRICE_DIR, f"{d_curr}.parquet")
        if not os.path.exists(p_curr):
            continue
        try:
            df_curr = pd.read_parquet(p_curr)
        except:
            continue

        df_curr = df_curr[df_curr['ts_code'].apply(is_main_board)]
        if df_curr.empty:
            continue

        curr_idx = date_idx.get(d_curr, 0)
        codes_to_keep = []
        for code in df_curr['ts_code'].unique():
            if code in first_seen:
                list_date = first_seen[code]
                list_idx = date_idx.get(list_date, 0)
                days_since_list = curr_idx - list_idx
                if days_since_list >= MIN_LISTING_DAYS:
                    codes_to_keep.append(code)
        df_curr = df_curr[df_curr['ts_code'].isin(codes_to_keep)]
        if df_curr.empty:
            continue

        for _, row in df_curr.iterrows():
            ts_code = row['ts_code']
            if ts_code not in price_hist:
                price_hist[ts_code] = []
                vol_hist[ts_code] = []
            price_hist[ts_code].append(row['close'])
            vol_hist[ts_code].append(row['vol'])
            if len(price_hist[ts_code]) > 100:
                price_hist[ts_code] = price_hist[ts_code][-100:]
                vol_hist[ts_code] = vol_hist[ts_code][-100:]

        for ts_code in df_curr['ts_code'].unique():
            if ts_code in vol_hist and len(vol_hist[ts_code]) > 0:
                hist_vol_mean[ts_code] = np.mean(vol_hist[ts_code])

        features = calc_price_features(df_curr)
        features = calc_volume_features(features, hist_vol_mean)
        features = calc_momentum_features(features, price_hist)
        features = calc_volatility_features(features, price_hist)
        features = calc_technical_indicators(features, price_hist)

        fund_df = load_fundamental_features(d_curr)
        if not fund_df.empty:
            features = pd.merge(features, fund_df, on='ts_code', how='left')

        news_df = load_news_features(d_curr)
        if not news_df.empty:
            features = pd.merge(features, news_df, on='ts_code', how='left')
            features['news_stock_impact'] = features['news_stock_impact'].fillna(0)
            features['news_market_impact'] = features['news_market_impact'].fillna(0)
            features['news_has_mention'] = features['news_has_mention'].fillna(0)

        rank_df = load_rank_features(d_curr)
        if not rank_df.empty:
            features = pd.merge(features, rank_df, on='ts_code', how='left')
            features['ths_hot'] = features['ths_hot'].fillna(0)
            features['ths_hot_rank'] = features['ths_hot_rank'].fillna(9999)

        mf_df = load_moneyflow_features(d_curr)
        if not mf_df.empty:
            features = pd.merge(features, mf_df, on='ts_code', how='left')

        tn_df = load_ths_news_features(d_curr)
        if not tn_df.empty:
            features = pd.merge(features, tn_df, on='ts_code', how='left')

        d_t1 = available_dates[i + 1]
        d_exit = available_dates[i + 2]

        p_t1 = os.path.join(RAW_PRICE_DIR, f"{d_t1}.parquet")
        if os.path.exists(p_t1):
            try:
                df_t1 = pd.read_parquet(p_t1)
                t1_data = df_t1[['ts_code', 'open']].rename(columns={'open': 'entry_price'})
                features = pd.merge(features, t1_data, on='ts_code', how='left')
            except:
                pass

        p_exit = os.path.join(RAW_PRICE_DIR, f"{d_exit}.parquet")
        if os.path.exists(p_exit):
            try:
                df_exit = pd.read_parquet(p_exit)
                exit_data = df_exit[['ts_code', 'close']].rename(columns={'close': 'exit_price_1d'})
                features = pd.merge(features, exit_data, on='ts_code', how='left')
                if 'exit_price_1d' in features.columns and 'entry_price' in features.columns:
                    features['actual_return'] = (
                        features['exit_price_1d'] / features['entry_price'] - 1
                    )
            except:
                pass

        features['trade_date'] = d_curr
        training_data.append(features)

    if training_data:
        result = pd.concat(training_data, ignore_index=True)
        result.to_parquet(FEATURES_FILE)
        print(f"\n特征数据已保存: {FEATURES_FILE}")
        print(f"数据: {len(result)} 行, {len(result.columns)} 列")
        return result

    print("错误: 未能生成特征数据")
    return None


if __name__ == '__main__':
    run()
