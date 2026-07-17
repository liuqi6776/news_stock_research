"""
TS Enhanced Backtest v14 - Deep optimization based on v13 results.
Key findings from v13:
  - TS_Filter_PosWR (delta_winner_rate > 0): Sharpe 5.49 -> 11.95
  - TS_Filter_NegRet1d (ret_1d < 0.05): Sharpe 5.49 -> 9.78, MDD -68.6% -> -29.7%
Now: fine-tune thresholds, optimize ts_score weights, test more combinations.
"""
import os, gc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.dirname(THIS_DIR)
DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')

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

def backtest(trades_df, all_dates_set, take_profit=None, stop_loss=None):
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
            elif stop_loss and sell_low <= buy_price * (1 - stop_loss):
                sell_price = buy_price * (1 - stop_loss)
            elif take_profit and sell_high >= buy_price * (1 + take_profit):
                sell_price = buy_price * (1 + take_profit)
            else:
                sell_price = sell_close

            ret = (sell_price / buy_price) - 1 - 0.0015
            day_pnl += alloc * ret

        capital += day_pnl
        equity.append({'date': int_to_date(date_t2), 'nav': capital})

    eq_df = pd.DataFrame(equity)
    if len(eq_df) < 2:
        return eq_df, {}
    total_ret = capital / initial_cap - 1
    years = len(eq_df) / 252.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    df_ret = eq_df['nav'].pct_change().dropna()
    if len(df_ret) == 0 or df_ret.std() == 0:
        return eq_df, {'total': total_ret, 'ann': ann_ret, 'sharpe': 0, 'mdd': 0,
                       'calmar': 0, 'win_rate': 0, 'trades': total_trades,
                       'cannot_sell': cannot_sell_trades, 'final_nav': capital}
    mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    win_rate = (df_ret > 0).mean()
    return eq_df, {'total': total_ret, 'ann': ann_ret, 'sharpe': sharpe, 'mdd': mdd,
                   'calmar': calmar, 'win_rate': win_rate, 'trades': total_trades,
                   'cannot_sell': cannot_sell_trades, 'final_nav': capital}

def main():
    print("=" * 100, flush=True)
    print("  TS Enhanced Backtest v14 - Deep Optimization", flush=True)
    print("=" * 100, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)

    trades_ts = pd.read_csv(os.path.join(THIS_DIR, 'trades_with_ts_features.csv'))
    print(f"Loaded {len(trades_ts)} trades with TS features", flush=True)

    print("\n[Step 1] Fine-tuning filter thresholds...", flush=True)

    wr_thresholds = [0, -2, -5, -10, 5, 10]
    ret1d_thresholds = [0.03, 0.05, 0.08, 0.10, 0.15]
    ret5d_thresholds = [0.3, 0.5, 0.8]
    ma5dist_thresholds = [0.1, 0.2, 0.3]

    results = {}

    for wr_t in wr_thresholds:
        for ret_t in ret1d_thresholds:
            mask = (trades_ts['delta_winner_rate'] > wr_t) & (trades_ts['ret_1d'] < ret_t)
            t = trades_ts[mask]
            if len(t) < 20:
                continue
            eq, stats = backtest(t, all_dates_set)
            if stats and stats['sharpe'] > 0:
                sname = f"WR>{wr_t}_R1d<{ret_t}"
                results[sname] = (eq, stats, t)

    for ret_t in ret1d_thresholds:
        mask = trades_ts['ret_1d'] < ret_t
        t = trades_ts[mask]
        if len(t) < 20:
            continue
        eq, stats = backtest(t, all_dates_set)
        if stats and stats['sharpe'] > 0:
            sname = f"R1d<{ret_t}"
            results[sname] = (eq, stats, t)

    for wr_t in wr_thresholds:
        mask = trades_ts['delta_winner_rate'] > wr_t
        t = trades_ts[mask]
        if len(t) < 20:
            continue
        eq, stats = backtest(t, all_dates_set)
        if stats and stats['sharpe'] > 0:
            sname = f"WR>{wr_t}"
            results[sname] = (eq, stats, t)

    for ret5_t in ret5d_thresholds:
        mask = trades_ts['ret_5d'] < ret5_t
        t = trades_ts[mask]
        if len(t) < 20:
            continue
        eq, stats = backtest(t, all_dates_set)
        if stats and stats['sharpe'] > 0:
            sname = f"R5d<{ret5_t}"
            results[sname] = (eq, stats, t)

    for ma_t in ma5dist_thresholds:
        mask = trades_ts['ma5_dist'] < ma_t
        t = trades_ts[mask]
        if len(t) < 20:
            continue
        eq, stats = backtest(t, all_dates_set)
        if stats and stats['sharpe'] > 0:
            sname = f"MA5d<{ma_t}"
            results[sname] = (eq, stats, t)

    print("\n[Step 2] Testing take-profit combinations on top filters...", flush=True)

    sorted_by_sharpe = sorted(results.items(), key=lambda x: x[1][1]['sharpe'], reverse=True)
    top_filters = [(n, d) for n, d in sorted_by_sharpe[:10] if d[1]['trades'] >= 30]

    for fname, (eq, stats, tdf) in top_filters:
        for tp in [0.15, 0.18, 0.20, 0.25]:
            eq2, stats2 = backtest(tdf, all_dates_set, take_profit=tp)
            if stats2 and stats2['sharpe'] > 0:
                sname = f"{fname}_TP{int(tp*100)}"
                results[sname] = (eq2, stats2, tdf)

    print("\n[Step 3] Optimizing ts_score weights...", flush=True)

    weight_combos = [
        {'w_ret1d': -0.3, 'w_dwr': 2.0, 'w_dcc': -1.0, 'w_ret5d': -0.1, 'w_ma5': -0.2},
        {'w_ret1d': -0.5, 'w_dwr': 3.0, 'w_dcc': -1.5, 'w_ret5d': -0.2, 'w_ma5': -0.3},
        {'w_ret1d': -0.2, 'w_dwr': 1.5, 'w_dcc': -0.5, 'w_ret5d': -0.05, 'w_ma5': -0.1},
        {'w_ret1d': -0.1, 'w_dwr': 1.0, 'w_dcc': -0.3, 'w_ret5d': 0.0, 'w_ma5': -0.1},
        {'w_ret1d': -0.4, 'w_dwr': 2.5, 'w_dcc': -1.0, 'w_ret5d': -0.15, 'w_ma5': -0.2},
        {'w_ret1d': -0.6, 'w_dwr': 4.0, 'w_dcc': -2.0, 'w_ret5d': -0.3, 'w_ma5': -0.4},
    ]

    for wi, wc in enumerate(weight_combos):
        trades_w = trades_ts.copy()
        trades_w['ts_score_v2'] = (
            trades_w['ret_1d'].abs() * wc['w_ret1d'] +
            trades_w['delta_winner_rate'] * wc['w_dwr'] +
            trades_w['delta_chip_conc'].abs() * wc['w_dcc'] +
            trades_w['ret_5d'].abs() * wc['w_ret5d'] +
            trades_w['ma5_dist'].abs() * wc['w_ma5']
        )

        for ts_thresh in [0, -5, -10]:
            mask = trades_w['ts_score_v2'] > ts_thresh
            t = trades_w[mask]
            if len(t) < 20:
                continue
            eq, stats = backtest(t, all_dates_set)
            if stats and stats['sharpe'] > 0:
                sname = f"W{wi}_TS>{ts_thresh}"
                results[sname] = (eq, stats, t)

    print("\n[Step 4] Top-N selection with TS score ranking...", flush=True)

    for top_n in [1, 2, 3]:
        for rank_col in ['ts_score', 'prob', 'delta_winner_rate']:
            if rank_col not in trades_ts.columns:
                continue
            t = trades_ts.groupby('date_t2', group_keys=False).apply(
                lambda g: g.nlargest(top_n, rank_col)
            )
            if len(t) < 20:
                continue
            eq, stats = backtest(t, all_dates_set)
            if stats and stats['sharpe'] > 0:
                sname = f"Top{top_n}_by_{rank_col}"
                results[sname] = (eq, stats, t)

    print("\n[Step 5] Combined: filter + Top-N + take-profit...", flush=True)

    best_filters = [
        ('delta_winner_rate > 0', trades_ts[trades_ts['delta_winner_rate'] > 0]),
        ('ret_1d < 0.05', trades_ts[trades_ts['ret_1d'] < 0.05]),
        ('delta_winner_rate > 0 & ret_1d < 0.05', trades_ts[(trades_ts['delta_winner_rate'] > 0) & (trades_ts['ret_1d'] < 0.05)]),
        ('delta_winner_rate > -2 & ret_1d < 0.08', trades_ts[(trades_ts['delta_winner_rate'] > -2) & (trades_ts['ret_1d'] < 0.08)]),
    ]

    for fname, filtered in best_filters:
        if len(filtered) < 20:
            continue
        for top_n in [1, 2, 3]:
            t = filtered.groupby('date_t2', group_keys=False).apply(
                lambda g: g.nlargest(top_n, 'prob')
            )
            if len(t) < 20:
                continue
            for tp in [None, 0.18, 0.20]:
                eq, stats = backtest(t, all_dates_set, take_profit=tp)
                if stats and stats['sharpe'] > 0:
                    tp_str = f"_TP{int(tp*100)}" if tp else ""
                    sname = f"{fname}_Top{top_n}{tp_str}"
                    sname = sname.replace(' & ', '_').replace(' > ', '>').replace(' < ', '<').replace(' ', '')
                    results[sname] = (eq, stats, t)

    sorted_r = sorted(results.items(), key=lambda x: x[1][1]['sharpe'], reverse=True)

    valid_results = [(n, d) for n, d in sorted_r if d[1]['trades'] >= 20 and d[1]['sharpe'] < 100]

    print(f"\n{'Rank':>4} {'Scheme':<50} {'Total':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'WinRate':>8} {'Trades':>7}")
    print('-' * 120)
    for rank, (sname, (eq, stats, tdf)) in enumerate(valid_results[:40], 1):
        print(f"{rank:>4} {sname:<50} {stats['total']:>9.2%} {stats['sharpe']:>7.2f} "
              f"{stats['mdd']:>9.2%} {stats['calmar']:>7.2f} {stats['win_rate']:>7.2%} {stats['trades']:>7}")

    print("\n[Step 6] Plotting top schemes...", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    top_for_plot = valid_results[:10]

    fig, axes = plt.subplots(3, 1, figsize=(20, 20))

    ax = axes[0]
    for sname, (eq, stats, _) in top_for_plot:
        if not eq.empty:
            ax.plot(eq['date'], eq['nav'], label=f"{sname[:40]} (S={stats['sharpe']:.1f})")
    ax.set_title('Equity Curves - Top 10 by Sharpe')
    ax.set_ylabel('NAV')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    names = [s[0][:35] for s in top_for_plot]
    sharpes = [s[1][1]['sharpe'] for s in top_for_plot]
    mdds = [abs(s[1][1]['mdd']) for s in top_for_plot]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, sharpes, width, label='Sharpe', color='steelblue')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, mdds, width, label='|MDD|', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Sharpe Ratio')
    ax2.set_ylabel('|Max Drawdown|')
    ax.set_title('Risk-Return Comparison')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    ax = axes[2]
    calmar_vals = [s[1][1]['calmar'] for s in top_for_plot]
    totals = [s[1][1]['total'] for s in top_for_plot]
    ax.bar(x - width/2, calmar_vals, width, label='Calmar', color='green')
    ax3 = ax.twinx()
    ax3.bar(x + width/2, totals, width, label='Total Return', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Calmar Ratio')
    ax3.set_ylabel('Total Return')
    ax.set_title('Calmar & Return Comparison')
    ax.legend(loc='upper left', fontsize=8)
    ax3.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(THIS_DIR, 'ts_backtest_v14_deep_optimization.png'), dpi=150, bbox_inches='tight')

    for sname, (eq, stats, tdf) in valid_results[:5]:
        if not eq.empty:
            eq.to_csv(os.path.join(THIS_DIR, f'equity_ts_v14_{sname[:40].replace(">","gt").replace("<","lt")}.csv'), index=False)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
