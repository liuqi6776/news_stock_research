"""
Time-Series Enhanced Model - Training & Backtest

Uses the 28 selected TS features to train a new model and compare
with the base model (5 features from doubao_result).

Schemes:
1. Baseline: doubao_result base model (5 features)
2. TS-Only: New model with 28 TS features only
3. Combined: Base model probability + TS features (33 features total)
4. Base+ReRank: Base model top candidates, re-ranked by TS model
"""
import os, sys
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
FINAL_DIR = os.path.join(os.path.dirname(THIS_DIR))
DOUBAO_DIR = os.path.join(FINAL_DIR, 'doubao')

CIRC_MV_LIMIT = 1000000
TEST_START = '20230101'
TEST_END = '20260324'
TRAIN_START = '20200801'

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

def load_features_for_day(d, all_dates, all_dates_set, idx, news_mkt, news_stk):
    """Load base features for a single day (same as doubao_result)."""
    d_int = int(d)
    p_price = os.path.join(PRICE_DIR, f"{d}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d}.parquet")
    p_rank = os.path.join(RANK_DIR, f"{d}.parquet")

    if not os.path.exists(p_price) or not os.path.exists(p_chip) or not os.path.exists(p_other):
        return None

    price = pd.read_parquet(p_price, columns=['ts_code', 'open', 'close', 'pct_chg', 'pre_close'])
    price = price[price['ts_code'].apply(is_main_board)]

    chip = pd.read_parquet(p_chip)
    chip = chip[chip['ts_code'].apply(is_main_board)]
    chip['chip_concentration'] = (chip['cost_85pct'] - chip['cost_15pct']) / (chip['cost_50pct'] + 1e-8)

    other = pd.read_parquet(p_other, columns=['ts_code', 'turnover_rate', 'volume_ratio', 'circ_mv'])
    other = other[other['ts_code'].apply(is_main_board)]

    df = pd.merge(price, chip[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code', how='left')
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

    # News
    if news_stk is not None and not news_stk.empty:
        ns = news_stk.copy()
        if pd.api.types.is_datetime64_any_dtype(ns['trade_date']):
            ns['date'] = ns['trade_date'].dt.strftime('%Y%m%d').astype(int)
        else:
            ns['date'] = ns['trade_date']
        stock_news = ns[ns['date'] == d_int]
        if not stock_news.empty:
            sn = stock_news.groupby('ts_code')['news_stock_impact'].max().reset_index()
            df = pd.merge(df, sn, on='ts_code', how='left')
            df['news_stock_impact'] = df['news_stock_impact'].fillna(0)
        else:
            df['news_stock_impact'] = 0
    else:
        df['news_stock_impact'] = 0

    if news_mkt is not None and not news_mkt.empty:
        nm = news_mkt.copy()
        if pd.api.types.is_datetime64_any_dtype(nm['trade_date']):
            nm['date'] = nm['trade_date'].dt.strftime('%Y%m%d').astype(int)
        else:
            nm['date'] = nm['trade_date']
        mkt_news = nm[nm['date'] == d_int]
        if not mkt_news.empty:
            df['news_market_impact'] = mkt_news['news_market_impact'].max()
        else:
            df['news_market_impact'] = 0
    else:
        df['news_market_impact'] = 0

    df['date'] = d_int
    return df

def backtest(trades_df, initial_capital=1000000, max_positions=3, take_profit=0.15):
    """Simple backtest engine."""
    if trades_df.empty:
        return pd.DataFrame(), {'total': 0, 'sharpe': 0, 'mdd': 0, 'win_rate': 0, 'calmar': 0, 'n_trades': 0}

    trades = trades_df.sort_values('entry_date').copy()
    capital = initial_capital
    equity_records = []
    active_trades = []

    for _, trade in trades.iterrows():
        entry_date = trade['entry_date']
        entry_price = trade['entry_price']
        exit_date = trade['exit_date']
        exit_price = trade['exit_price']
        ret = trade.get('return', (exit_price / entry_price) - 1)

        if len(active_trades) >= max_positions:
            continue

        position_size = capital / max_positions
        pnl = position_size * ret
        capital += pnl
        active_trades.append(ret)

        equity_records.append({
            'date': exit_date,
            'capital': capital,
            'return': ret,
            'pnl': pnl
        })

    if not equity_records:
        return pd.DataFrame(), {'total': 0, 'sharpe': 0, 'mdd': 0, 'win_rate': 0, 'calmar': 0, 'n_trades': 0}

    eq = pd.DataFrame(equity_records)
    eq = eq.sort_values('date').reset_index(drop=True)
    eq['equity'] = eq['capital']

    total_ret = (capital / initial_capital) - 1
    daily_rets = eq.groupby('date')['return'].mean()
    if len(daily_rets) > 1:
        sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * np.sqrt(252)
    else:
        sharpe = 0

    eq['peak'] = eq['equity'].cummax()
    eq['drawdown'] = (eq['equity'] - eq['peak']) / eq['peak']
    mdd = eq['drawdown'].min()

    win_rate = (eq['return'] > 0).mean()
    calmar = total_ret / abs(mdd) if mdd != 0 else 0

    stats = {
        'total': total_ret,
        'sharpe': sharpe,
        'mdd': mdd,
        'win_rate': win_rate,
        'calmar': calmar,
        'n_trades': len(eq)
    }
    return eq, stats

def main():
    print("=" * 90, flush=True)
    print("  TIME-SERIES ENHANCED MODEL - Training & Backtest", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    # Load selected TS features
    ts_selected = pd.read_csv(os.path.join(THIS_DIR, 'ts_selected_features.csv'))['feature'].tolist()
    print(f"  Selected TS features: {len(ts_selected)}", flush=True)

    # Load labeled panel
    print("\n[Step 1] Loading labeled panel...", flush=True)
    labeled_path = os.path.join(THIS_DIR, 'ts_panel_labeled.parquet')
    labeled_df = pd.read_parquet(labeled_path)
    print(f"  Loaded: {len(labeled_df)} rows", flush=True)

    # Split train/test
    train_df = labeled_df[(labeled_df['date'] >= int(TRAIN_START)) & (labeled_df['date'] < int(TEST_START))].copy()
    test_df = labeled_df[labeled_df['date'] >= int(TEST_START)].copy()
    print(f"  Train: {len(train_df)} rows, pos_rate={train_df['label'].mean():.3f}", flush=True)
    print(f"  Test: {len(test_df)} rows, pos_rate={test_df['label'].mean():.3f}", flush=True)

    # Load base model
    print("\n[Step 2] Loading base model...", flush=True)
    doubao_model_path = os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib')
    loaded = joblib.load(doubao_model_path)
    base_model = loaded[0] if isinstance(loaded, tuple) else loaded
    print("  Base model loaded", flush=True)

    # Train TS model
    print("\n[Step 3] Training TS-enhanced model...", flush=True)
    ts_model_path = os.path.join(THIS_DIR, 'models', 'ts_model.joblib')
    os.makedirs(os.path.join(THIS_DIR, 'models'), exist_ok=True)

    if os.path.exists(ts_model_path):
        ts_model = joblib.load(ts_model_path)
        print("  Loaded existing TS model", flush=True)
    else:
        # Use only rows with valid TS features
        ts_train = train_df.dropna(subset=ts_selected)
        print(f"  TS training samples: {len(ts_train)} (after dropping NaN)", flush=True)

        X_ts = ts_train[ts_selected].fillna(0)
        y_ts = ts_train['label'].astype(int)

        pos_rate = y_ts.mean()
        scale_pos = max(1, (1 - pos_rate) / max(pos_rate, 0.01))

        ts_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            scale_pos_weight=scale_pos,
            eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1
        )
        ts_model.fit(X_ts, y_ts)
        joblib.dump(ts_model, ts_model_path)
        print("  TS model trained and saved!", flush=True)

    # Train Combined model (base features + TS features)
    print("\n[Step 4] Training Combined model (base + TS)...", flush=True)
    combined_model_path = os.path.join(THIS_DIR, 'models', 'combined_model.joblib')

    combined_feats = BASE_FEATS + ts_selected
    if os.path.exists(combined_model_path):
        combined_model = joblib.load(combined_model_path)
        print("  Loaded existing Combined model", flush=True)
    else:
        comb_train = train_df.dropna(subset=combined_feats)
        print(f"  Combined training samples: {len(comb_train)}", flush=True)

        X_comb = comb_train[combined_feats].fillna(0)
        y_comb = comb_train['label'].astype(int)

        pos_rate = y_comb.mean()
        scale_pos = max(1, (1 - pos_rate) / max(pos_rate, 0.01))

        combined_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            scale_pos_weight=scale_pos,
            eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1
        )
        combined_model.fit(X_comb, y_comb)
        joblib.dump(combined_model, combined_model_path)
        print("  Combined model trained and saved!", flush=True)

    # Generate trades for backtest
    print("\n[Step 5] Generating trades for backtest...", flush=True)
    trades_path = os.path.join(THIS_DIR, 'ts_trades_all.parquet')

    if os.path.exists(trades_path):
        print("  Loading existing trades...", flush=True)
        all_trades = pd.read_parquet(trades_path)
    else:
        test_dates = [(idx, all_dates[idx]) for idx in range(5, len(all_dates) - 2)
                       if all_dates[idx] >= TEST_START and all_dates[idx] <= TEST_END]
        print(f"  Test dates: {len(test_dates)}", flush=True)

        all_trades_records = []

        for i, (idx, d) in enumerate(test_dates):
            d_int = int(d)
            dt = int_to_date(d_int)
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

            # Get features for this day
            day_data = load_features_for_day(d, all_dates, all_dates_set, idx, news_mkt, news_stk)
            if day_data is None:
                continue

            # Get TS features from panel
            day_ts = labeled_df[labeled_df['date'] == d_int].copy()
            if day_ts.empty:
                continue

            # Merge base features with TS features
            day_merged = pd.merge(day_data, day_ts[['ts_code'] + ts_selected], on='ts_code', how='left')

            # Get T+1 and T+2 prices
            p1_path = os.path.join(PRICE_DIR, f"{t1}.parquet")
            p2_path = os.path.join(PRICE_DIR, f"{t2}.parquet")
            if not os.path.exists(p1_path) or not os.path.exists(p2_path):
                continue

            price_t1 = pd.read_parquet(p1_path, columns=['ts_code', 'open']).rename(columns={'open': 'open_t1'})
            price_t2 = pd.read_parquet(p2_path, columns=['ts_code', 'close', 'high', 'low']).rename(
                columns={'close': 'close_t2', 'high': 'high_t2', 'low': 'low_t2'})

            day_merged = pd.merge(day_merged, price_t1, on='ts_code', how='left')
            day_merged = pd.merge(day_merged, price_t2, on='ts_code', how='left')

            day_merged = day_merged.dropna(subset=['open_t1', 'close_t2'])

            if day_merged.empty:
                continue

            # Base model predictions
            X_base = day_merged[BASE_FEATS].fillna(0)
            base_prob = base_model.predict_proba(X_base)[:, 1]

            # TS model predictions
            X_ts = day_merged[ts_selected].fillna(0)
            ts_prob = ts_model.predict_proba(X_ts)[:, 1]

            # Combined model predictions
            X_comb = day_merged[combined_feats].fillna(0)
            comb_prob = combined_model.predict_proba(X_comb)[:, 1]

            day_merged['base_prob'] = base_prob
            day_merged['ts_prob'] = ts_prob
            day_merged['comb_prob'] = comb_prob

            # Compute actual returns
            day_merged['actual_ret'] = day_merged['close_t2'] / day_merged['open_t1'] - 1
            day_merged['actual_high_ret'] = day_merged['high_t2'] / day_merged['open_t1'] - 1

            for _, row in day_merged.iterrows():
                all_trades_records.append({
                    'date': d_int,
                    'ts_code': row['ts_code'],
                    'entry_date': t1,
                    'exit_date': t2,
                    'entry_price': row['open_t1'],
                    'exit_price': row['close_t2'],
                    'high_price': row.get('high_t2', row['close_t2']),
                    'return': row['actual_ret'],
                    'high_return': row.get('actual_high_ret', row['actual_ret']),
                    'base_prob': row['base_prob'],
                    'ts_prob': row['ts_prob'],
                    'comb_prob': row['comb_prob'],
                })

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(test_dates)} dates processed", flush=True)

        all_trades = pd.DataFrame(all_trades_records)
        all_trades.to_parquet(trades_path, index=False)
        print(f"  Total trades: {len(all_trades)}", flush=True)

    print(f"  Total trades: {len(all_trades)}", flush=True)

    # Backtest different schemes
    print("\n[Step 6] Backtesting schemes...", flush=True)

    schemes = {
        'base_top3_prob04': {'prob_col': 'base_prob', 'prob_thresh': 0.4, 'top_n': 3, 'take_profit': 0.15},
        'ts_top3_prob04': {'prob_col': 'ts_prob', 'prob_thresh': 0.4, 'top_n': 3, 'take_profit': 0.15},
        'comb_top3_prob04': {'prob_col': 'comb_prob', 'prob_thresh': 0.4, 'top_n': 3, 'take_profit': 0.15},
        'base_top3_prob05': {'prob_col': 'base_prob', 'prob_thresh': 0.5, 'top_n': 3, 'take_profit': 0.15},
        'ts_top3_prob05': {'prob_col': 'ts_prob', 'prob_thresh': 0.5, 'top_n': 3, 'take_profit': 0.15},
        'comb_top3_prob05': {'prob_col': 'comb_prob', 'prob_thresh': 0.5, 'top_n': 3, 'take_profit': 0.15},
        'base_top2_prob04': {'prob_col': 'base_prob', 'prob_thresh': 0.4, 'top_n': 2, 'take_profit': 0.15},
        'comb_top2_prob04': {'prob_col': 'comb_prob', 'prob_thresh': 0.4, 'top_n': 2, 'take_profit': 0.15},
        'base_top3_prob04_tp18': {'prob_col': 'base_prob', 'prob_thresh': 0.4, 'top_n': 3, 'take_profit': 0.18},
        'comb_top3_prob04_tp18': {'prob_col': 'comb_prob', 'prob_thresh': 0.4, 'top_n': 3, 'take_profit': 0.18},
        'rerank_base_top10_ts': {'prob_col': 'base_prob', 'prob_thresh': 0.3, 'top_n': 3, 'take_profit': 0.15, 'rerank_col': 'ts_prob', 'rerank_top': 10},
        'rerank_base_top10_comb': {'prob_col': 'base_prob', 'prob_thresh': 0.3, 'top_n': 3, 'take_profit': 0.15, 'rerank_col': 'comb_prob', 'rerank_top': 10},
    }

    results = {}
    for sname, params in schemes.items():
        prob_col = params['prob_col']
        prob_thresh = params['prob_thresh']
        top_n = params['top_n']
        take_profit = params['take_profit']
        rerank_col = params.get('rerank_col', None)
        rerank_top = params.get('rerank_top', None)

        # Filter trades by date
        daily_picks = []
        for date, group in all_trades.groupby('date'):
            if rerank_col and rerank_top:
                candidates = group[group[prob_col] >= prob_thresh].nlargest(rerank_top, prob_col)
                picks = candidates.nlargest(top_n, rerank_col)
            else:
                candidates = group[group[prob_col] >= prob_thresh]
                picks = candidates.nlargest(top_n, prob_col)

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

        trades_df = pd.DataFrame(daily_picks)
        eq, stats = backtest(trades_df, take_profit=take_profit)
        results[sname] = (eq, stats, trades_df)
        print(f"  {sname:<30} Total={stats['total']:>8.2%}  Sharpe={stats['sharpe']:>6.2f}  "
              f"MDD={stats['mdd']:>8.2%}  WinRate={stats['win_rate']:>6.2%}  Trades={stats['n_trades']}", flush=True)

    # Sort by Sharpe
    sorted_r = sorted(results.items(), key=lambda x: x[1][1]['sharpe'], reverse=True)

    # Plot
    print("\n[Step 7] Plotting...", flush=True)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Equity curves
    ax = axes[0]
    for sname, (eq, stats, _) in sorted_r[:6]:
        if not eq.empty:
            ax.plot(eq['date'], eq['equity'], label=f"{sname} (Sharpe={stats['sharpe']:.2f})")
    ax.set_title('Equity Curves - Top 6 Schemes')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Stats comparison
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
    plt.savefig(os.path.join(THIS_DIR, 'ts_backtest_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Chart saved", flush=True)

    # Save results
    print(f"\n{'Rank':>4} {'Scheme':<35} {'Total':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'WinRate':>8} {'Trades':>7}")
    print('-' * 100)
    for rank, (sname, (eq, stats, tdf)) in enumerate(sorted_r, 1):
        print(f"{rank:>4} {sname:<35} {stats['total']:>9.2%} {stats['sharpe']:>7.2f} "
              f"{stats['mdd']:>9.2%} {stats['calmar']:>7.2f} {stats['win_rate']:>7.2%} {stats['n_trades']:>7}")

    # Save equity curves and trades
    for sname, (eq, stats, tdf) in sorted_r:
        if not eq.empty:
            eq.to_csv(os.path.join(THIS_DIR, f'equity_ts_{sname}.csv'), index=False)
        tdf.to_csv(os.path.join(THIS_DIR, f'trades_ts_{sname}.csv'), index=False)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
