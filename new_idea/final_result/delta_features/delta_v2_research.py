"""
Delta Features Strategy v2 - Secondary Filtering Approach

Problem: Pure delta model fails because training data (2020-2022) lacks
chip/rank data, so delta features are all zeros during training.

Solution: Use doubao_result's 5-feature model for probability prediction,
then use delta features as SECONDARY signals for:
  1. Re-ranking: Sort by prob * delta_signal instead of just prob
  2. Filtering: Remove stocks with adverse delta signals
  3. Composite score: prob * (1 + alpha * delta_score)
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
NEWS_MAJOR_DIR = r"C:\Users\liuqi\clowspace\toolstock\news_major1"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.join(os.path.dirname(THIS_DIR))

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']

CIRC_MV_LIMIT = 1000000
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

def load_features_with_delta(d_curr, prev_dates, news_mkt, news_stk):
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

    if len(rank_df) > 0:
        df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    else:
        df = price_df.copy()
        df['hot_rank_pct'] = 0.5

    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate', 'cost_50pct', 'weight_avg']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['circ_mv'] <= CIRC_MV_LIMIT]
    df = add_news_features(df, d_curr, news_mkt, news_stk)

    # Delta features
    df['delta_winner_rate_1d'] = 0.0
    df['delta_chip_concentration_1d'] = 0.0
    df['delta_cost_50pct_1d'] = 0.0
    df['ret_1d'] = 0.0
    df['ret_3d'] = 0.0
    df['ret_5d'] = 0.0
    df['delta_turnover_rate_1d'] = 0.0
    df['delta_volume_ratio_1d'] = 0.0
    df['delta_vol_1d'] = 0.0
    df['delta_hot_1d'] = 0.0
    df['chip_price_diverge'] = 0.0
    df['vol_price_diverge'] = 0.0
    df['ma5_dist'] = 0.0

    if len(prev_dates) >= 1:
        d_prev = prev_dates[0]
        p_chip_prev = os.path.join(CHIP_DIR, f"{d_prev}.parquet")
        p_price_prev = os.path.join(PRICE_DIR, f"{d_prev}.parquet")
        p_other_prev = os.path.join(OTHER_DIR, f"{d_prev}.parquet")
        p_rank_prev = os.path.join(RANK_DIR, f"{d_prev}.parquet")

        if os.path.exists(p_chip_prev):
            chip_prev = pd.read_parquet(p_chip_prev)
            chip_prev['chip_concentration'] = (chip_prev['cost_85pct'] - chip_prev['cost_15pct']) / (chip_prev['cost_50pct'] + 1e-8)
            merged_chip = pd.merge(
                chip_df[['ts_code', 'chip_concentration', 'winner_rate', 'cost_50pct', 'weight_avg']],
                chip_prev[['ts_code', 'chip_concentration', 'winner_rate', 'cost_50pct', 'weight_avg']],
                on='ts_code', suffixes=('', '_prev')
            )
            df = pd.merge(df, merged_chip[['ts_code', 'chip_concentration_prev', 'winner_rate_prev',
                                            'cost_50pct_prev', 'weight_avg_prev']], on='ts_code', how='left')
            df['delta_winner_rate_1d'] = (df['winner_rate'] - df['winner_rate_prev']).fillna(0)
            df['delta_chip_concentration_1d'] = (df['chip_concentration'] - df['chip_concentration_prev']).fillna(0)
            df['delta_cost_50pct_1d'] = ((df['cost_50pct'] - df['cost_50pct_prev']) / (df['cost_50pct_prev'] + 1e-8)).fillna(0)
            df['chip_price_diverge'] = df['delta_cost_50pct_1d'] - (df['pct_chg'] / 100.0 if 'pct_chg' in df.columns else 0)

        if os.path.exists(p_price_prev):
            price_prev = pd.read_parquet(p_price_prev, columns=['ts_code', 'close', 'vol', 'amount'])
            merged_price = pd.merge(
                price_df[['ts_code', 'close', 'vol', 'amount']],
                price_prev, on='ts_code', suffixes=('', '_prev')
            )
            df = pd.merge(df, merged_price[['ts_code', 'close_prev', 'vol_prev', 'amount_prev']], on='ts_code', how='left')
            df['ret_1d'] = (df['close'] / (df['close_prev'] + 1e-8) - 1).fillna(0)
            df['delta_vol_1d'] = (df['vol'] / (df['vol_prev'] + 1e-8) - 1).fillna(0)
            df['vol_price_diverge'] = df['delta_vol_1d'] - df['ret_1d'].abs()

        if os.path.exists(p_other_prev):
            other_prev = pd.read_parquet(p_other_prev, columns=['ts_code', 'turnover_rate', 'volume_ratio'])
            merged_other = pd.merge(
                other_df[['ts_code', 'turnover_rate', 'volume_ratio']],
                other_prev, on='ts_code', suffixes=('', '_prev')
            )
            df = pd.merge(df, merged_other[['ts_code', 'turnover_rate_prev', 'volume_ratio_prev']], on='ts_code', how='left')
            df['delta_turnover_rate_1d'] = (df['turnover_rate'] - df['turnover_rate_prev']).fillna(0)
            df['delta_volume_ratio_1d'] = (df['volume_ratio'] - df['volume_ratio_prev']).fillna(0)

        if os.path.exists(p_rank_prev):
            rank_prev = pd.read_parquet(p_rank_prev)
            if len(rank_prev) > 0 and len(rank_df) > 0:
                merged_rank = pd.merge(rank_df[['ts_code', 'hot']], rank_prev[['ts_code', 'hot']], on='ts_code', suffixes=('', '_prev'))
                df = pd.merge(df, merged_rank[['ts_code', 'hot_prev']], on='ts_code', how='left')
                df['delta_hot_1d'] = (df['hot_rank_pct'] - df['hot_prev'].rank(pct=True)).fillna(0)

    if len(prev_dates) >= 2:
        d_prev3 = prev_dates[min(2, len(prev_dates)-1)]
        p_price_3d = os.path.join(PRICE_DIR, f"{d_prev3}.parquet")
        if os.path.exists(p_price_3d):
            price_3d = pd.read_parquet(p_price_3d, columns=['ts_code', 'close'])
            merged_3d = pd.merge(price_df[['ts_code', 'close']], price_3d, on='ts_code', suffixes=('', '_3d'))
            df['ret_3d'] = (df['close'] / (merged_3d['close_3d'] + 1e-8) - 1).fillna(0)

    if len(prev_dates) >= 4:
        d_prev5 = prev_dates[min(4, len(prev_dates)-1)]
        p_price_5d = os.path.join(PRICE_DIR, f"{d_prev5}.parquet")
        if os.path.exists(p_price_5d):
            price_5d = pd.read_parquet(p_price_5d, columns=['ts_code', 'close'])
            merged_5d = pd.merge(price_df[['ts_code', 'close']], price_5d, on='ts_code', suffixes=('', '_5d'))
            df['ret_5d'] = (df['close'] / (merged_5d['close_5d'] + 1e-8) - 1).fillna(0)

    if len(prev_dates) >= 4:
        closes = [price_df[['ts_code', 'close']]]
        for i in range(min(4, len(prev_dates))):
            pp = os.path.join(PRICE_DIR, f"{prev_dates[i]}.parquet")
            if os.path.exists(pp):
                c = pd.read_parquet(pp, columns=['ts_code', 'close'])
                closes.append(c)
        if len(closes) >= 3:
            all_c = closes[0].rename(columns={'close': 'c_0'})
            for i, c in enumerate(closes[1:], 1):
                all_c = pd.merge(all_c, c.rename(columns={'close': f'c_{i}'}), on='ts_code', how='outer')
            close_cols = [f'c_{i}' for i in range(len(closes)) if f'c_{i}' in all_c.columns]
            if len(close_cols) >= 3:
                all_c['ma5'] = all_c[close_cols].mean(axis=1)
                all_c['ma5_dist'] = (all_c['c_0'] / all_c['ma5'] - 1).fillna(0)
                df = pd.merge(df, all_c[['ts_code', 'ma5_dist']], on='ts_code', how='left')

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

def main():
    print("DELTA FEATURES v2 - Secondary Filtering with Base Model", flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    # Load base model from doubao_result
    doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
    loaded = joblib.load(doubao_model_path)
    base_model = loaded[0] if isinstance(loaded, tuple) else loaded
    print(f"Base model loaded from doubao_result", flush=True)

    # Load existing trade pool from enhanced (5-feature model)
    enhanced_pool_path = os.path.join(FINAL_DIR, 'enhanced', 'pool_base_trades.csv')
    if os.path.exists(enhanced_pool_path):
        print(f"Loading enhanced trade pool...", flush=True)
        base_trades = pd.read_csv(enhanced_pool_path)
    else:
        print(f"No enhanced trade pool found!", flush=True)
        return

    print(f"Base trade pool: {len(base_trades)} trades", flush=True)

    # Now we need to add delta features to each trade
    # For efficiency, we'll regenerate with delta features
    print(f"\nAdding delta features to trade pool...", flush=True)

    trades_path = os.path.join(THIS_DIR, 'pool_delta_v2_trades.csv')
    if os.path.exists(trades_path):
        print(f"Loading existing delta v2 trade pool...", flush=True)
        trades_df = pd.read_csv(trades_path)
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
                df = load_features_with_delta(d_t, prev_dates, news_mkt, news_stk)
            except Exception as e:
                print(f"  ERROR {d_t}: {e}", flush=True)
                continue
            if df is None:
                continue

            # Base model probability
            X_base = df[BASE_FEATS].fillna(0)
            df['prob'] = base_model.predict_proba(X_base)[:, 1]

            # Delta composite score
            # Based on feature analysis: delta_cost_50pct_1d is most important
            # Negative delta_cost_50pct means cost decreasing = bullish signal
            # Positive delta_turnover = more activity = potential
            # Negative chip_price_diverge = cost moving down while price up = healthy
            df['delta_score'] = (
                -df['delta_cost_50pct_1d'] * 2.0 +     # cost decreasing = bullish
                df['delta_turnover_rate_1d'] * 0.5 +     # more turnover = attention
                -df['delta_chip_concentration_1d'] * 1.0 + # concentration decreasing = distribution
                df['delta_volume_ratio_1d'] * 0.3 +      # volume ratio increasing
                -df['chip_price_diverge'] * 1.5 +         # chip-price divergence negative = healthy
                -df['ret_1d'].abs() * 0.5 +               # less extreme moves
                df['delta_hot_1d'] * 0.3                   # hot rank increasing
            )

            # Composite score: prob * (1 + alpha * delta_score_normalized)
            delta_score_z = (df['delta_score'] - df['delta_score'].mean()) / (df['delta_score'].std() + 1e-8)
            df['composite_score'] = df['prob'] * (1 + 0.3 * delta_score_z)

            p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
            p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                continue

            df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'pre_close'])
            df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
            df_t1 = df_t1.rename(columns={'open': 'open_t1', 'pre_close': 'pre_close_t1'})
            df_t2 = df_t2.rename(columns={'open': 'open_t2', 'high': 'high_t2', 'low': 'low_t2',
                                           'close': 'close_t2', 'pre_close': 'pre_close_t2'})

            merged = pd.merge(df[['ts_code', 'prob', 'composite_score', 'delta_score',
                                   'delta_cost_50pct_1d', 'delta_turnover_rate_1d',
                                   'delta_chip_concentration_1d', 'chip_price_diverge',
                                   'ret_1d', 'delta_hot_1d']],
                              df_t1, on='ts_code', how='inner')
            merged = pd.merge(merged, df_t2, on='ts_code', how='inner')

            merged['is_gem'] = merged['ts_code'].str.contains('300|301|688|689', regex=True)
            merged['up_limit'] = np.where(merged['is_gem'],
                                           (merged['pre_close_t1'] * 1.2).round(2),
                                           (merged['pre_close_t1'] * 1.1).round(2))
            valid = merged[~merged['open_t1'].isna() & (merged['open_t1'] < merged['up_limit'])].copy()

            batch = valid.copy()
            batch = batch.rename(columns={'open_t1': 'buy_price', 'open_t2': 'sell_open',
                                           'high_t2': 'sell_high', 'close_t2': 'sell_close',
                                           'pre_close_t2': 'sell_pre_close'})
            batch['date_t'] = d_t
            batch['date_t1'] = d_t1
            batch['date_t2'] = d_t2
            all_picks.append(batch)

            count += 1
            if count % 20 == 0:
                n_trades = sum(len(b) for b in all_picks)
                print(f"  {count}/{total} days, {n_trades} trades", flush=True)

        trades_df = pd.concat(all_picks, ignore_index=True)
        trades_df.to_csv(trades_path, index=False)
        print(f"Delta v2 trade pool saved: {len(trades_df)} trades", flush=True)

    print(f"Trade pool: {len(trades_df)} trades", flush=True)
    print(f"Prob stats: mean={trades_df['prob'].mean():.4f}, std={trades_df['prob'].std():.4f}", flush=True)
    print(f"Composite score stats: mean={trades_df['composite_score'].mean():.4f}, std={trades_df['composite_score'].std():.4f}", flush=True)

    # Backtest with different selection methods
    print(f"\nBacktesting...", flush=True)

    def select_top_n(trades, score_col, prob_thresh, top_n):
        if prob_thresh > 0:
            filtered = trades[trades['prob'] >= prob_thresh].copy()
        else:
            filtered = trades.copy()
        selected = []
        for date_t, group in filtered.groupby('date_t', sort=True):
            top = group.nlargest(top_n, score_col)
            selected.append(top)
        if not selected:
            return pd.DataFrame()
        return pd.concat(selected)

    def filter_delta(trades, prob_thresh, top_n, delta_filter=None):
        filtered = trades[trades['prob'] >= prob_thresh].copy()
        if delta_filter == 'cost_down':
            filtered = filtered[filtered['delta_cost_50pct_1d'] < 0]
        elif delta_filter == 'chip_diverge_neg':
            filtered = filtered[filtered['chip_price_diverge'] < 0]
        elif delta_filter == 'turnover_up':
            filtered = filtered[filtered['delta_turnover_rate_1d'] > 0]
        elif delta_filter == 'cost_down_or_diverge':
            filtered = filtered[(filtered['delta_cost_50pct_1d'] < 0) | (filtered['chip_price_diverge'] < -0.01)]
        selected = []
        for date_t, group in filtered.groupby('date_t', sort=True):
            top = group.nlargest(top_n, 'prob')
            selected.append(top)
        if not selected:
            return pd.DataFrame()
        return pd.concat(selected)

    schemes = [
        ('Base_Top1_P04',            'prob',           0.4, 1, None),
        ('Composite_Top1_P04',       'composite_score', 0.4, 1, None),
        ('Composite_Top1_P03',       'composite_score', 0.3, 1, None),
        ('Composite_Top1_P05',       'composite_score', 0.5, 1, None),
        ('Composite_Top1_P04_TP18',  'composite_score', 0.4, 1, None),
        ('Composite_Top2_P04',       'composite_score', 0.4, 2, None),
        ('DeltaFilter_cost_Top1_P04','prob',           0.4, 1, 'cost_down'),
        ('DeltaFilter_diverge_Top1', 'prob',           0.4, 1, 'chip_diverge_neg'),
        ('DeltaFilter_turnover_Top1','prob',           0.4, 1, 'turnover_up'),
        ('DeltaFilter_combo_Top1',   'prob',           0.4, 1, 'cost_down_or_diverge'),
    ]

    results = {}
    for sname, score_col, p_thresh, top_n, delta_filter in schemes:
        if delta_filter:
            selected = filter_delta(trades_df, p_thresh, top_n, delta_filter)
        else:
            tp = 0.18 if 'TP18' in sname else None
            selected = select_top_n(trades_df, score_col, p_thresh, top_n)
        eq, stats = backtest(selected, all_dates_set, take_profit=0.18 if 'TP18' in sname else None)
        if stats:
            results[sname] = (eq, stats, selected)
            print(f"  {sname:<35} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  "
                  f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}  Trades={stats['trades']:>5d}", flush=True)

    # Ranking
    sorted_results = sorted(results.items(), key=lambda x: x[1][1]['sharpe'], reverse=True)
    print(f"\nFINAL RANKING (by Sharpe):", flush=True)
    for rank, (sname, (eq, stats, _)) in enumerate(sorted_results, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  {rank:>2}. {sname:<35} Sharpe={stats['sharpe']:>6.2f}  Total={stats['total']:>9.2%}  "
              f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}{marker}", flush=True)

    # Save best
    best_name = sorted_results[0][0]
    best_eq, best_stats, best_trades = sorted_results[0][1]
    best_eq.to_csv(os.path.join(THIS_DIR, 'best_equity_v2.csv'), index=False)
    best_trades.to_csv(os.path.join(THIS_DIR, 'best_trades_v2.csv'), index=False)

    # Plot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(16, 8))
    for sname, (eq, stats, _) in sorted_results[:8]:
        eq_norm = eq['nav'] / eq['nav'].iloc[0]
        label = f"{sname}: S={stats['sharpe']:.2f} R={stats['total']:.0%}"
        ax.plot(eq['date'], eq_norm, label=label, linewidth=1.5)

    doubao_eq_path = os.path.join(FINAL_DIR, 'doubao', 'equity.csv')
    if os.path.exists(doubao_eq_path):
        doubao_eq = pd.read_csv(doubao_eq_path)
        doubao_eq['date'] = pd.to_datetime(doubao_eq['date'])
        doubao_norm = doubao_eq['nav'] / doubao_eq['nav'].iloc[0]
        ax.plot(doubao_eq['date'], doubao_norm, label='doubao_result (ref)', linewidth=2.5, color='black', linestyle='--')

    ax.set_title('Delta v2: Secondary Filtering with Base Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('NAV (normalized)')
    ax.set_xlabel('Date')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(THIS_DIR, 'delta_v2_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\nChart saved", flush=True)

    for sname, (eq, stats, trades_df) in sorted_results:
        safe_name = sname.replace(' ', '_')
        eq.to_csv(os.path.join(THIS_DIR, f'equity_{safe_name}.csv'), index=False)
        trades_df.to_csv(os.path.join(THIS_DIR, f'trades_{safe_name}.csv'), index=False)

    print(f"All results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
