"""
TS Enhanced Backtest v10 - Faithful doubao reproduction + TS re-ranking
Key insights from analysis:
1. doubao uses rank_df as base (no fallback when empty) - trades start from 20230821
2. doubao selects prob>0.8 Top3, fallback Top1
3. doubao does NOT filter 300/301/BJ - only filters 688
4. Most trades are fallback Top1 (prob < 0.8)
5. TS features only computed for candidate stocks (not all 3000+)
"""
import os, sys, gc, traceback, time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
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

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']

CIRC_MV_LIMIT = 1000000

def is_gem_or_star(ts_code):
    return ('300' in ts_code) or ('301' in ts_code) or ('688' in ts_code) or ('689' in ts_code)

def int_to_date(date_int):
    s = str(int(date_int))
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

def load_features_exact_doubao(d_curr, news_mkt, news_stk):
    """Exact same as doubao's load_features - uses rank_df as base, no fallback"""
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
    df = add_news_features(df, d_curr, news_mkt, news_stk)
    for f in BASE_FEATS:
        if f not in df.columns:
            df[f] = 0.0
    return df

def load_prev_data_for_codes(ts_codes_set, prev_dates):
    """Load previous days data only for given ts_codes."""
    result_list = []
    for prev_d in prev_dates:
        p_chip = os.path.join(CHIP_DIR, f"{prev_d}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{prev_d}.parquet")
        p_rank = os.path.join(RANK_DIR, f"{prev_d}.parquet")

        if not os.path.exists(p_chip) or not os.path.exists(p_price):
            result_list.append(None)
            continue

        price = pd.read_parquet(p_price, columns=['ts_code', 'close', 'vol'])
        price = price[price['ts_code'].isin(ts_codes_set)]
        chip = pd.read_parquet(p_chip, columns=['ts_code', 'winner_rate', 'cost_85pct', 'cost_15pct', 'cost_50pct'])
        chip = chip[chip['ts_code'].isin(ts_codes_set)]
        chip['chip_concentration'] = (chip['cost_85pct'] - chip['cost_15pct']) / (chip['cost_50pct'] + 1e-8)

        df = pd.merge(price, chip[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code', how='left')

        day_dict = {}
        for rec in df.to_dict('records'):
            day_dict[rec['ts_code']] = rec

        if os.path.exists(p_rank):
            rank = pd.read_parquet(p_rank, columns=['ts_code', 'hot'])
            rank = rank[rank['ts_code'].isin(ts_codes_set)]
            if len(rank) > 0:
                rank['hot_rank_pct'] = rank['hot'].rank(pct=True)
                for rec in rank.to_dict('records'):
                    if rec['ts_code'] in day_dict:
                        day_dict[rec['ts_code']]['hot_rank_pct'] = rec['hot_rank_pct']

        result_list.append(day_dict)

    return result_list

def compute_ts_score(row, prev_data_list):
    """Compute TS score for a single stock from prev data."""
    tc = row['ts_code']
    curr_close = row['close']
    curr_chip_conc = row['chip_concentration']
    curr_winner_rate = row['winner_rate']
    curr_hot_rank = row['hot_rank_pct']

    ret_1d = 0.0
    delta_chip_conc = 0.0
    delta_winner_rate = 0.0
    delta_hot_rank = 0.0
    ret_5d = 0.0
    ma5 = 0.0

    if prev_data_list and prev_data_list[0] is not None and tc in prev_data_list[0]:
        p = prev_data_list[0][tc]
        c1 = p.get('close', 0)
        if c1 > 0:
            ret_1d = curr_close / c1 - 1
        delta_chip_conc = curr_chip_conc - p.get('chip_concentration', 0)
        delta_winner_rate = curr_winner_rate - p.get('winner_rate', 0)
        delta_hot_rank = curr_hot_rank - p.get('hot_rank_pct', 0.5)

    if len(prev_data_list) >= 5 and prev_data_list[4] is not None and tc in prev_data_list[4]:
        c5 = prev_data_list[4][tc].get('close', 0)
        if c5 > 0:
            ret_5d = curr_close / c5 - 1

    if len(prev_data_list) >= 5:
        cs, cnt = 0.0, 0
        for j in range(5):
            if prev_data_list[j] is not None and tc in prev_data_list[j]:
                cs += prev_data_list[j][tc].get('close', 0)
                cnt += 1
        if cnt >= 3:
            ma5 = cs / cnt

    ma5_dist = (curr_close / (ma5 + 1e-8) - 1) if ma5 > 0 else 0.0

    ts_score = (
        -abs(ret_1d) * 0.3 +
        delta_winner_rate * 2.0 +
        -abs(delta_chip_conc) * 1.0 +
        delta_hot_rank * 0.5 +
        -abs(ret_5d) * 0.1 +
        -abs(ma5_dist) * 0.2
    )

    return {
        'ts_score': ts_score,
        'ret_1d': ret_1d,
        'delta_chip_conc': delta_chip_conc,
        'delta_winner_rate': delta_winner_rate,
        'delta_hot_rank': delta_hot_rank,
        'ret_5d': ret_5d,
        'ma5_dist': ma5_dist,
    }

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
    print("=" * 90, flush=True)
    print("  TS Enhanced Backtest v10 - Faithful doubao + TS re-ranking", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    print("\n[Step 1] Loading base model...", flush=True)
    doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
    loaded = joblib.load(doubao_model_path)
    base_model = loaded[0] if isinstance(loaded, tuple) else loaded
    print("  Base model loaded", flush=True)

    schemes = {
        'Doubao_Base': {'method': 'doubao', 'top_n': 3, 'tp': None},
        'Doubao_TP18': {'method': 'doubao', 'top_n': 3, 'tp': 0.18},
        'Doubao_TP20': {'method': 'doubao', 'top_n': 3, 'tp': 0.20},
        'TS_Rerank_Top1': {'method': 'ts_rerank', 'top_n': 1, 'tp': None},
        'TS_Rerank_Top2': {'method': 'ts_rerank', 'top_n': 2, 'tp': None},
        'TS_Rerank_Top3': {'method': 'ts_rerank', 'top_n': 3, 'tp': None},
        'TS_Rerank_Top1_TP18': {'method': 'ts_rerank', 'top_n': 1, 'tp': 0.18},
        'TS_Rerank_Top1_TP20': {'method': 'ts_rerank', 'top_n': 1, 'tp': 0.20},
        'Comb_w30_Top1': {'method': 'combined', 'top_n': 1, 'ts_weight': 0.3, 'tp': None},
        'Comb_w30_Top2': {'method': 'combined', 'top_n': 2, 'ts_weight': 0.3, 'tp': None},
        'Comb_w30_Top3': {'method': 'combined', 'top_n': 3, 'ts_weight': 0.3, 'tp': None},
        'Comb_w50_Top1': {'method': 'combined', 'top_n': 1, 'ts_weight': 0.5, 'tp': None},
        'Comb_w30_Top1_TP18': {'method': 'combined', 'top_n': 1, 'ts_weight': 0.3, 'tp': 0.18},
        'Comb_w30_Top1_TP20': {'method': 'combined', 'top_n': 1, 'ts_weight': 0.3, 'tp': 0.20},
    }

    scheme_trades = {sname: [] for sname in schemes}

    print("\n[Step 2] Generating trades...", flush=True)
    test_dates = []
    for idx in range(1, len(all_dates) - 2):
        d = all_dates[idx]
        if d >= '20230101' and d <= '20260324':
            test_dates.append((idx, d))

    total = len(test_dates)
    print(f"  Test dates: {total}", flush=True)

    for i, (idx, d_t) in enumerate(test_dates):
        try:
            d_t1 = all_dates[idx + 1]
            d_t2 = all_dates[idx + 2]

            df = load_features_exact_doubao(d_t, news_mkt, news_stk)
            if df is None or len(df) == 0:
                continue

            X = df[BASE_FEATS].fillna(0)
            if len(X) == 0:
                continue
            df['prob'] = base_model.predict_proba(X)[:, 1]

            # Get candidates for all schemes: prob > 0.8 or top N
            candidates_08 = df[df['prob'] > 0.8].sort_values('prob', ascending=False)
            fallback = df.sort_values('prob', ascending=False).head(1)

            # Determine which stocks need TS features
            ts_codes_needed = set()
            if not candidates_08.empty:
                ts_codes_needed.update(candidates_08['ts_code'].values[:10])
            ts_codes_needed.update(fallback['ts_code'].values[:5])

            # Load prev 5 days for TS features (only for needed codes)
            prev_dates = []
            for j in range(1, 6):
                prev_idx = idx - j
                if prev_idx >= 0:
                    prev_dates.append(all_dates[prev_idx])

            prev_data_list = load_prev_data_for_codes(ts_codes_needed, prev_dates)

            # Compute TS features for candidates
            def add_ts_to_df(sub_df):
                if sub_df.empty:
                    return sub_df
                sub_df = sub_df.copy()
                ts_scores = []
                ts_details = []
                for _, row in sub_df.iterrows():
                    ts_info = compute_ts_score(row, prev_data_list)
                    ts_scores.append(ts_info['ts_score'])
                    ts_details.append(ts_info)
                sub_df['ts_score'] = ts_scores
                for key in ['ret_1d', 'delta_chip_conc', 'delta_winner_rate', 'delta_hot_rank', 'ret_5d', 'ma5_dist']:
                    sub_df[key] = [d[key] for d in ts_details]
                return sub_df

            candidates_08_ts = add_ts_to_df(candidates_08.head(10))
            fallback_ts = add_ts_to_df(fallback)

            del prev_data_list
            gc.collect()

            # Get T+1/T+2 prices
            p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
            p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                continue

            df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'pre_close'])
            df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])

            def make_trade(pick, sname):
                ts_code = pick['ts_code']
                t1r = df_t1[df_t1['ts_code'] == ts_code]
                if t1r.empty:
                    return None
                t1 = t1r.iloc[0]
                pre_close_t1 = t1['pre_close']
                up_limit = round(pre_close_t1 * 1.2, 2) if is_gem_or_star(ts_code) else round(pre_close_t1 * 1.1, 2)
                if pd.isna(t1['open']) or t1['open'] >= up_limit:
                    return None
                t2r = df_t2[df_t2['ts_code'] == ts_code]
                if t2r.empty:
                    return None
                t2 = t2r.iloc[0]
                return {
                    'date_t': d_t, 'date_t1': d_t1, 'date_t2': int(d_t2),
                    'ts_code': ts_code, 'buy_price': t1['open'],
                    'sell_open': t2['open'], 'sell_high': t2['high'],
                    'sell_close': t2['close'], 'sell_pre_close': t2['pre_close'],
                    'prob': pick['prob'],
                    'ts_score': pick.get('ts_score', 0),
                    'ret_1d': pick.get('ret_1d', 0),
                    'delta_winner_rate': pick.get('delta_winner_rate', 0),
                    'delta_chip_conc': pick.get('delta_chip_conc', 0),
                }

            for sname, scheme in schemes.items():
                method = scheme['method']
                top_n = scheme['top_n']

                if method == 'doubao':
                    if not candidates_08_ts.empty:
                        picks = candidates_08_ts.head(min(3, top_n))
                    else:
                        picks = fallback_ts
                elif method == 'ts_rerank':
                    if not candidates_08_ts.empty:
                        picks = candidates_08_ts.nlargest(top_n, 'ts_score')
                    else:
                        picks = fallback_ts.head(top_n)
                elif method == 'combined':
                    ts_weight = scheme.get('ts_weight', 0.3)
                    if not candidates_08_ts.empty:
                        cands = candidates_08_ts.copy()
                        ts_min = cands['ts_score'].min()
                        ts_max = cands['ts_score'].max()
                        if ts_max > ts_min:
                            cands['ts_score_norm'] = (cands['ts_score'] - ts_min) / (ts_max - ts_min)
                        else:
                            cands['ts_score_norm'] = 0.5
                        cands['combined_score'] = (1 - ts_weight) * cands['prob'] + ts_weight * cands['ts_score_norm']
                        picks = cands.nlargest(top_n, 'combined_score')
                    else:
                        picks = fallback_ts.head(top_n)
                else:
                    continue

                if picks.empty:
                    continue

                for _, pick in picks.iterrows():
                    trade = make_trade(pick, sname)
                    if trade is not None:
                        scheme_trades[sname].append(trade)

            del df, candidates_08_ts, fallback_ts
            gc.collect()

            if (i + 1) % 50 == 0:
                counts = {s: len(t) for s, t in scheme_trades.items()}
                print(f"  {i+1}/{total} days, trades: {counts}", flush=True)

        except Exception as e:
            print(f"  ERROR on {d_t}: {e}", flush=True)
            traceback.print_exc()
            gc.collect()
            continue

    print("\n[Step 3] Backtesting...", flush=True)
    results = {}

    for sname, scheme in schemes.items():
        trades_list = scheme_trades[sname]
        if not trades_list:
            continue
        trades_df = pd.DataFrame(trades_list)
        tp = scheme.get('tp', None)
        eq, stats = backtest(trades_df, all_dates_set, take_profit=tp)
        if stats:
            results[sname] = (eq, stats, trades_df)
            print(f"  {sname:<30} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  "
                  f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}  Trades={stats['trades']:>5d}", flush=True)

    sorted_r = sorted(results.items(), key=lambda x: x[1][1]['sharpe'], reverse=True)

    print("\n[Step 4] Plotting...", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(18, 14))

    ax = axes[0]
    for sname, (eq, stats, _) in sorted_r[:8]:
        if not eq.empty:
            ax.plot(eq['date'], eq['nav'], label=f"{sname} (S={stats['sharpe']:.2f})")
    ax.set_title('Equity Curves - Top 8 Schemes')
    ax.set_xlabel('Date')
    ax.set_ylabel('NAV')
    ax.legend(fontsize=7)
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
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Sharpe Ratio')
    ax2.set_ylabel('Total Return')
    ax.set_title('Scheme Comparison')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(THIS_DIR, 'ts_backtest_v10_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Chart saved", flush=True)

    print(f"\n{'Rank':>4} {'Scheme':<30} {'Total':>10} {'Ann':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'WinRate':>8} {'Trades':>7}")
    print('-' * 105)
    for rank, (sname, (eq, stats, tdf)) in enumerate(sorted_r, 1):
        print(f"{rank:>4} {sname:<30} {stats['total']:>9.2%} {stats['ann']:>9.2%} {stats['sharpe']:>7.2f} "
              f"{stats['mdd']:>9.2%} {stats['calmar']:>7.2f} {stats['win_rate']:>7.2%} {stats['trades']:>7}")

    for sname, (eq, stats, tdf) in sorted_r[:5]:
        if not eq.empty:
            eq.to_csv(os.path.join(THIS_DIR, f'equity_ts_v10_{sname}.csv'), index=False)
            tdf.to_csv(os.path.join(THIS_DIR, f'trades_ts_v10_{sname}.csv'), index=False)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
