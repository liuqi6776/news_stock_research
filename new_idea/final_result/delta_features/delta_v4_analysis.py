"""
Delta Features v4 - Deep Analysis: Can delta features distinguish winners
from losers WITHIN the base model's top candidates?

If delta features can separate winners from losers among the top-N picks,
they're useful for re-ranking even if they can't replace the base model.
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
import seaborn as sns
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
TEST_START = '20230101'
TEST_END = '20260324'

def is_gem_or_star(ts_code):
    return ('300' in ts_code) or ('301' in ts_code) or ('688' in ts_code) or ('689' in ts_code)

def int_to_date(date_int):
    s = str(date_int)
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

def main():
    print("DELTA FEATURES v4 - Discriminative Power Analysis", flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
    loaded = joblib.load(doubao_model_path)
    base_model = loaded[0] if isinstance(loaded, tuple) else loaded
    print("Base model loaded", flush=True)

    # Collect: for each day, get base model top-10 candidates and their actual returns
    analysis_path = os.path.join(THIS_DIR, 'delta_discrim_analysis.csv')

    if os.path.exists(analysis_path):
        print("Loading existing analysis...", flush=True)
        analysis_df = pd.read_csv(analysis_path)
    else:
        records = []
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
            except:
                continue
            if df is None:
                continue

            X_base = df[BASE_FEATS].fillna(0)
            df['prob'] = base_model.predict_proba(X_base)[:, 1]

            pt1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
            pt2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
            if not os.path.exists(pt1) or not os.path.exists(pt2):
                continue

            df_t1 = pd.read_parquet(pt1, columns=['ts_code', 'open', 'pre_close'])
            df_t2 = pd.read_parquet(pt2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
            df_t1 = df_t1.rename(columns={'open': 'open_t1', 'pre_close': 'pre_close_t1'})
            df_t2 = df_t2.rename(columns={'open': 'open_t2', 'high': 'high_t2', 'low': 'low_t2',
                                           'close': 'close_t2', 'pre_close': 'pre_close_t2'})

            merged = pd.merge(df, df_t1, on='ts_code', how='inner')
            merged = pd.merge(merged, df_t2, on='ts_code', how='inner')

            merged['is_gem'] = merged['ts_code'].str.contains('300|301|688|689', regex=True)
            merged['up_limit'] = np.where(merged['is_gem'],
                                           (merged['pre_close_t1'] * 1.2).round(2),
                                           (merged['pre_close_t1'] * 1.1).round(2))
            valid = merged[~merged['open_t1'].isna() & (merged['open_t1'] < merged['up_limit'])].copy()

            # Get actual return
            valid['actual_ret'] = valid['close_t2'] / valid['open_t1'] - 1
            valid['is_winner'] = (valid['actual_ret'] > 0.04).astype(int)

            # Focus on top-10 by prob
            top10 = valid.nlargest(10, 'prob')

            for _, row in top10.iterrows():
                rec = {'date': d_t, 'ts_code': row['ts_code'], 'prob': row['prob'],
                       'actual_ret': row['actual_ret'], 'is_winner': row['is_winner'],
                       'rank_in_top10': 0}
                for f in DELTA_FEATS_ONLY:
                    rec[f'delta_{f}'] = row[f] if f in row.index else 0
                records.append(rec)

            count += 1
            if count % 50 == 0:
                print(f"  {count}/{total} days", flush=True)

        analysis_df = pd.DataFrame(records)
        analysis_df.to_csv(analysis_path, index=False)
        print(f"Analysis saved: {len(analysis_df)} records", flush=True)

    print(f"\nAnalysis: {len(analysis_df)} top-10 candidates", flush=True)
    print(f"Winners: {analysis_df['is_winner'].sum()} ({analysis_df['is_winner'].mean():.2%})", flush=True)

    # Analysis 1: Delta feature means for winners vs losers
    print("\n--- Delta Feature Discrimination (Winners vs Losers in Top-10) ---", flush=True)
    delta_cols = [c for c in analysis_df.columns if c.startswith('delta_')]
    print(f"{'Feature':<35} {'Winner_Mean':>12} {'Loser_Mean':>12} {'Diff':>12} {'t-stat':>10} {'p-value':>10}", flush=True)
    print('-' * 95)

    from scipy import stats as scipy_stats
    discrim_results = []
    for col in delta_cols:
        w = analysis_df[analysis_df['is_winner'] == 1][col].dropna()
        l = analysis_df[analysis_df['is_winner'] == 0][col].dropna()
        if len(w) < 10 or len(l) < 10:
            continue
        t_stat, p_val = scipy_stats.ttest_ind(w, l)
        diff = w.mean() - l.mean()
        print(f"  {col:<33} {w.mean():>12.6f} {l.mean():>12.6f} {diff:>12.6f} {t_stat:>10.3f} {p_val:>10.4f}", flush=True)
        discrim_results.append({'feature': col, 'winner_mean': w.mean(), 'loser_mean': l.mean(),
                                'diff': diff, 't_stat': t_stat, 'p_value': p_val,
                                'abs_t': abs(t_stat)})

    # Sort by discriminative power
    discrim_df = pd.DataFrame(discrim_results).sort_values('abs_t', ascending=False)
    print(f"\n--- Top Discriminative Delta Features ---", flush=True)
    for _, row in discrim_df.head(10).iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"  {row['feature']:<33} t={row['t_stat']:>7.3f}  p={row['p_value']:.4f}  diff={row['diff']:.6f} {sig}", flush=True)

    # Analysis 2: Conditional probability - does delta feature improve prediction?
    print("\n--- Conditional Analysis ---", flush=True)
    for col in discrim_df.head(5)['feature']:
        med = analysis_df[col].median()
        high_delta = analysis_df[analysis_df[col] >= med]
        low_delta = analysis_df[analysis_df[col] < med]
        if len(high_delta) > 20 and len(low_delta) > 20:
            print(f"  {col}:", flush=True)
            print(f"    High (>{med:.4f}): win_rate={high_delta['is_winner'].mean():.2%}, avg_ret={high_delta['actual_ret'].mean():.4f}, n={len(high_delta)}", flush=True)
            print(f"    Low (<{med:.4f}):  win_rate={low_delta['is_winner'].mean():.2%}, avg_ret={low_delta['actual_ret'].mean():.4f}, n={len(low_delta)}", flush=True)

    # Analysis 3: Rank correlation - does re-ranking by delta feature improve selection?
    print("\n--- Re-ranking Analysis ---", flush=True)
    for col in discrim_df.head(5)['feature']:
        # For each day, check if the top-1 by delta feature outperforms top-1 by prob
        daily_results = []
        for date, group in analysis_df.groupby('date'):
            if len(group) < 2:
                continue
            top_prob = group.nlargest(1, 'prob').iloc[0]
            top_delta = group.nlargest(1, col).iloc[0]
            daily_results.append({
                'date': date,
                'ret_prob_top1': top_prob['actual_ret'],
                'ret_delta_top1': top_delta['actual_ret'],
                'prob_top1_is_winner': top_prob['is_winner'],
                'delta_top1_is_winner': top_delta['is_winner'],
            })
        if daily_results:
            dr = pd.DataFrame(daily_results)
            prob_wr = dr['prob_top1_is_winner'].mean()
            delta_wr = dr['delta_top1_is_winner'].mean()
            prob_avg = dr['ret_prob_top1'].mean()
            delta_avg = dr['ret_delta_top1'].mean()
            print(f"  {col}:", flush=True)
            print(f"    Prob Top1:  avg_ret={prob_avg:.4f}, win_rate={prob_wr:.2%}", flush=True)
            print(f"    Delta Top1: avg_ret={delta_avg:.4f}, win_rate={delta_wr:.2%}", flush=True)

    # Save discrim results
    discrim_df.to_csv(os.path.join(THIS_DIR, 'delta_discrimination.csv'), index=False)

    # Plot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    top6 = discrim_df.head(6)['feature'].tolist()
    for ax, col in zip(axes.flatten(), top6):
        w = analysis_df[analysis_df['is_winner'] == 1][col].dropna()
        l = analysis_df[analysis_df['is_winner'] == 0][col].dropna()
        ax.hist(l, bins=50, alpha=0.5, label='Loser', density=True, color='red')
        ax.hist(w, bins=50, alpha=0.5, label='Winner', density=True, color='green')
        ax.set_title(f'{col}\nt={discrim_df[discrim_df["feature"]==col]["t_stat"].values[0]:.2f}')
        ax.legend()

    plt.suptitle('Delta Feature Distributions: Winners vs Losers (in Top-10)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(THIS_DIR, 'delta_discrimination.png'), dpi=150, bbox_inches='tight')
    print(f"\nChart saved", flush=True)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
