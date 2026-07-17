"""
Run backtest on pool_ts_v8.csv - separate from pool generation.
"""
import os, gc, time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def is_gem_or_star(ts_code):
    return any(x in ts_code for x in ['300', '301', '688', '689'])

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

def apply_selection(all_trades, prob_col='prob', prob_thresh=0.0, top_n=1):
    if prob_thresh > 0:
        filtered = all_trades[all_trades[prob_col] >= prob_thresh].copy()
    else:
        filtered = all_trades.copy()
    daily_groups = filtered.groupby('date_t', sort=True)
    selected = []
    for date_t, group in daily_groups:
        top = group.nlargest(top_n, prob_col)
        selected.append(top)
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected)

def apply_selection_ts_rerank(all_trades, base_thresh=0.4, top_n=1):
    filtered = all_trades[all_trades['base_prob'] >= base_thresh].copy()
    daily_groups = filtered.groupby('date_t', sort=True)
    selected = []
    for date_t, group in daily_groups:
        top = group.nlargest(top_n, 'ts_score')
        selected.append(top)
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected)

def apply_selection_combined_score(all_trades, base_thresh=0.4, top_n=1, ts_weight=0.3):
    filtered = all_trades[all_trades['base_prob'] >= base_thresh].copy()
    ts_min = filtered['ts_score'].min()
    ts_max = filtered['ts_score'].max()
    if ts_max > ts_min:
        filtered['ts_score_norm'] = (filtered['ts_score'] - ts_min) / (ts_max - ts_min)
    else:
        filtered['ts_score_norm'] = 0.5
    filtered['combined_score'] = (1 - ts_weight) * filtered['base_prob'] + ts_weight * filtered['ts_score_norm']
    daily_groups = filtered.groupby('date_t', sort=True)
    selected = []
    for date_t, group in daily_groups:
        top = group.nlargest(top_n, 'combined_score')
        selected.append(top)
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected)

def main():
    print("=" * 90, flush=True)
    print("  TS Backtest v8 - Backtesting from saved pool", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)

    print("\n[Step 1] Loading trade pool...", flush=True)
    trades_path = os.path.join(THIS_DIR, 'pool_ts_v8.csv')
    trades_df = pd.read_csv(trades_path)
    print(f"  Loaded: {len(trades_df)} trades", flush=True)
    print(f"  base_prob: mean={trades_df['base_prob'].mean():.4f}, std={trades_df['base_prob'].std():.4f}", flush=True)
    print(f"  ts_score:  mean={trades_df['ts_score'].mean():.4f}, std={trades_df['ts_score'].std():.4f}", flush=True)

    print("\n[Step 2] Backtesting...", flush=True)

    schemes = [
        ('Base_Top1_P04',          'base_prob', 0.4, 1, None),
        ('Base_Top1_P05',          'base_prob', 0.5, 1, None),
        ('Base_Top2_P04',          'base_prob', 0.4, 2, None),
        ('Base_Top3_P04',          'base_prob', 0.4, 3, None),
        ('Base_Top1_P04_TP18',     'base_prob', 0.4, 1, 0.18),
        ('Base_Top1_P04_TP20',     'base_prob', 0.4, 1, 0.20),
    ]

    results = {}

    for sname, prob_col, p_thresh, top_n, tp in schemes:
        print(f"  Running {sname}...", flush=True)
        selected = apply_selection(trades_df, prob_col=prob_col, prob_thresh=p_thresh, top_n=top_n)
        eq, stats = backtest(selected, all_dates_set, take_profit=tp)
        if stats:
            results[sname] = (eq, stats, selected)
            print(f"    {sname:<35} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  "
                  f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}  Trades={stats['trades']:>5d}", flush=True)
        del selected
        gc.collect()

    # TS re-ranking schemes
    ts_schemes = [
        ('TS_Rerank_P04_Top1',     0.4, 1, None),
        ('TS_Rerank_P04_Top2',     0.4, 2, None),
        ('TS_Rerank_P04_Top3',     0.4, 3, None),
        ('TS_Rerank_P05_Top1',     0.5, 1, None),
        ('TS_Rerank_P04_Top1_TP18', 0.4, 1, 0.18),
        ('TS_Rerank_P04_Top1_TP20', 0.4, 1, 0.20),
    ]

    for sname, base_thresh, top_n, tp in ts_schemes:
        print(f"  Running {sname}...", flush=True)
        selected = apply_selection_ts_rerank(trades_df, base_thresh=base_thresh, top_n=top_n)
        eq, stats = backtest(selected, all_dates_set, take_profit=tp)
        if stats:
            results[sname] = (eq, stats, selected)
            print(f"    {sname:<35} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  "
                  f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}  Trades={stats['trades']:>5d}", flush=True)
        del selected
        gc.collect()

    # Combined score schemes
    comb_schemes = [
        ('CombScore_P04_Top1_w30', 0.4, 1, 0.3, None),
        ('CombScore_P04_Top2_w30', 0.4, 2, 0.3, None),
        ('CombScore_P04_Top1_w50', 0.4, 1, 0.5, None),
        ('CombScore_P04_Top1_w30_TP18', 0.4, 1, 0.3, 0.18),
        ('CombScore_P04_Top1_w30_TP20', 0.4, 1, 0.3, 0.20),
    ]

    for sname, base_thresh, top_n, ts_weight, tp in comb_schemes:
        print(f"  Running {sname}...", flush=True)
        selected = apply_selection_combined_score(trades_df, base_thresh=base_thresh, top_n=top_n, ts_weight=ts_weight)
        eq, stats = backtest(selected, all_dates_set, take_profit=tp)
        if stats:
            results[sname] = (eq, stats, selected)
            print(f"    {sname:<35} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  "
                  f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}  Trades={stats['trades']:>5d}", flush=True)
        del selected
        gc.collect()

    sorted_r = sorted(results.items(), key=lambda x: x[1][1]['sharpe'], reverse=True)

    print("\n[Step 3] Plotting...", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(18, 14))

    ax = axes[0]
    for sname, (eq, stats, _) in sorted_r[:8]:
        if not eq.empty:
            ax.plot(eq['date'], eq['nav'], label=f"{sname} (Sharpe={stats['sharpe']:.2f})")
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
    plt.savefig(os.path.join(THIS_DIR, 'ts_backtest_v8_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Chart saved", flush=True)

    print(f"\n{'Rank':>4} {'Scheme':<40} {'Total':>10} {'Ann':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'WinRate':>8} {'Trades':>7}")
    print('-' * 115)
    for rank, (sname, (eq, stats, tdf)) in enumerate(sorted_r, 1):
        print(f"{rank:>4} {sname:<40} {stats['total']:>9.2%} {stats['ann']:>9.2%} {stats['sharpe']:>7.2f} "
              f"{stats['mdd']:>9.2%} {stats['calmar']:>7.2f} {stats['win_rate']:>7.2%} {stats['trades']:>7}")

    for sname, (eq, stats, tdf) in sorted_r[:5]:
        if not eq.empty:
            eq.to_csv(os.path.join(THIS_DIR, f'equity_ts_v8_{sname}.csv'), index=False)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
