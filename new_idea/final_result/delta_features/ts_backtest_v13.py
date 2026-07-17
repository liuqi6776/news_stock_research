"""
TS Enhanced Backtest v13 - Simplified, process one trade at a time.
"""
import os, gc, time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.dirname(THIS_DIR)

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
    print("  TS Enhanced Backtest v13 - Simplified single-trade processing", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    all_dates_idx = {d: i for i, d in enumerate(all_dates)}

    trades = pd.read_csv(os.path.join(FINAL_DIR, 'doubao', 'trades.csv'))
    print(f"Base trades: {len(trades)}", flush=True)

    eq_base, stats_base = backtest(trades, all_dates_set)
    print(f"Base: Total={stats_base['total']:.2%}, Sharpe={stats_base['sharpe']:.2f}, MDD={stats_base['mdd']:.2%}", flush=True)

    print("\n[Step 1] Computing TS features (one trade at a time)...", flush=True)

    ts_features = []
    for i, (_, trade) in enumerate(trades.iterrows()):
        ts_code = trade['ts_code']
        date_t = str(trade['date_t'])
        idx = all_dates_idx.get(date_t, -1)

        current_close = 0
        current_chip_conc = 0
        current_winner_rate = 0

        p_chip = os.path.join(CHIP_DIR, f"{date_t}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{date_t}.parquet")

        if os.path.exists(p_price):
            price = pd.read_parquet(p_price, columns=['ts_code', 'close'])
            pr = price[price['ts_code'] == ts_code]
            if not pr.empty:
                current_close = pr.iloc[0]['close']

        if os.path.exists(p_chip):
            chip = pd.read_parquet(p_chip, columns=['ts_code', 'winner_rate', 'cost_85pct', 'cost_15pct', 'cost_50pct'])
            cr = chip[chip['ts_code'] == ts_code]
            if not cr.empty:
                current_chip_conc = (cr.iloc[0]['cost_85pct'] - cr.iloc[0]['cost_15pct']) / (cr.iloc[0]['cost_50pct'] + 1e-8)
                current_winner_rate = cr.iloc[0]['winner_rate']

        ret_1d = 0.0
        delta_chip_conc = 0.0
        delta_winner_rate = 0.0
        ret_5d = 0.0
        ma5 = 0.0

        if idx >= 5:
            prev_dates = [all_dates[idx - j] for j in range(1, 6)]

            # Day -1
            p1_price = os.path.join(PRICE_DIR, f"{prev_dates[0]}.parquet")
            p1_chip = os.path.join(CHIP_DIR, f"{prev_dates[0]}.parquet")
            if os.path.exists(p1_price) and os.path.exists(p1_chip):
                pr1 = pd.read_parquet(p1_price, columns=['ts_code', 'close', 'vol'])
                pr1 = pr1[pr1['ts_code'] == ts_code]
                ch1 = pd.read_parquet(p1_chip, columns=['ts_code', 'winner_rate', 'cost_85pct', 'cost_15pct', 'cost_50pct'])
                ch1 = ch1[ch1['ts_code'] == ts_code]
                if not pr1.empty:
                    c1 = pr1.iloc[0]['close']
                    if c1 > 0:
                        ret_1d = current_close / c1 - 1
                if not ch1.empty:
                    cc1 = (ch1.iloc[0]['cost_85pct'] - ch1.iloc[0]['cost_15pct']) / (ch1.iloc[0]['cost_50pct'] + 1e-8)
                    delta_chip_conc = current_chip_conc - cc1
                    delta_winner_rate = current_winner_rate - ch1.iloc[0]['winner_rate']

            # Day -5
            p5_price = os.path.join(PRICE_DIR, f"{prev_dates[4]}.parquet")
            if os.path.exists(p5_price):
                pr5 = pd.read_parquet(p5_price, columns=['ts_code', 'close'])
                pr5 = pr5[pr5['ts_code'] == ts_code]
                if not pr5.empty:
                    c5 = pr5.iloc[0]['close']
                    if c5 > 0:
                        ret_5d = current_close / c5 - 1

            # MA5
            cs, cnt = 0.0, 0
            for j in range(5):
                pj_price = os.path.join(PRICE_DIR, f"{prev_dates[j]}.parquet")
                if os.path.exists(pj_price):
                    prj = pd.read_parquet(pj_price, columns=['ts_code', 'close'])
                    prj = prj[prj['ts_code'] == ts_code]
                    if not prj.empty:
                        cs += prj.iloc[0]['close']
                        cnt += 1
            if cnt >= 3:
                ma5 = cs / cnt

        ma5_dist = (current_close / (ma5 + 1e-8) - 1) if ma5 > 0 else 0.0

        ts_score = (
            -abs(ret_1d) * 0.3 +
            delta_winner_rate * 2.0 +
            -abs(delta_chip_conc) * 1.0 +
            -abs(ret_5d) * 0.1 +
            -abs(ma5_dist) * 0.2
        )

        ts_features.append({
            'ret_1d': ret_1d,
            'delta_chip_conc': delta_chip_conc,
            'delta_winner_rate': delta_winner_rate,
            'ret_5d': ret_5d,
            'ma5_dist': ma5_dist,
            'ts_score': ts_score,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(trades)} processed", flush=True)

        gc.collect()

    ts_df = pd.DataFrame(ts_features)
    trades_ts = pd.concat([trades.reset_index(drop=True), ts_df], axis=1)

    print(f"\nTS features stats:", flush=True)
    for col in ts_df.columns:
        print(f"  {col}: mean={ts_df[col].mean():.4f}, std={ts_df[col].std():.4f}, min={ts_df[col].min():.4f}, max={ts_df[col].max():.4f}", flush=True)

    trades_ts.to_csv(os.path.join(THIS_DIR, 'trades_with_ts_features.csv'), index=False)

    print("\n[Step 2] Testing schemes...", flush=True)

    schemes = {
        'Base_NoTP': {'filter': None, 'top_n': None, 'tp': None},
        'Base_TP15': {'filter': None, 'top_n': None, 'tp': 0.15},
        'Base_TP18': {'filter': None, 'top_n': None, 'tp': 0.18},
        'Base_TP20': {'filter': None, 'top_n': None, 'tp': 0.20},
        'TS_Filter_PosWR': {'filter': 'delta_winner_rate > 0', 'top_n': None, 'tp': None},
        'TS_Filter_PosWR_TP18': {'filter': 'delta_winner_rate > 0', 'top_n': None, 'tp': 0.18},
        'TS_Filter_NegRet1d': {'filter': 'ret_1d < 0.05', 'top_n': None, 'tp': None},
        'TS_Filter_LowVol': {'filter': 'ret_1d < 0.05 and delta_winner_rate > -5', 'top_n': None, 'tp': None},
        'TS_Filter_LowVol_TP18': {'filter': 'ret_1d < 0.05 and delta_winner_rate > -5', 'top_n': None, 'tp': 0.18},
        'TS_Filter_Strong': {'filter': 'delta_winner_rate > 0 and ret_1d < 0.03', 'top_n': None, 'tp': None},
        'TS_Filter_Strong_TP18': {'filter': 'delta_winner_rate > 0 and ret_1d < 0.03', 'top_n': None, 'tp': 0.18},
        'TS_Top1_by_ts_score': {'filter': None, 'top_n': 1, 'rank_by': 'ts_score', 'tp': None},
        'TS_Top1_by_prob': {'filter': None, 'top_n': 1, 'rank_by': 'prob', 'tp': None},
        'TS_Top1_TP18': {'filter': None, 'top_n': 1, 'rank_by': 'prob', 'tp': 0.18},
    }

    results = {}

    for sname, scheme in schemes.items():
        t = trades_ts.copy()

        filt = scheme.get('filter')
        if filt:
            try:
                t = t.query(filt)
                if t.empty:
                    continue
            except:
                continue

        top_n = scheme.get('top_n')
        rank_by = scheme.get('rank_by', 'prob')
        if top_n:
            t = t.groupby('date_t2', group_keys=False).apply(
                lambda g: g.nlargest(top_n, rank_by) if rank_by in g.columns else g.head(top_n)
            )

        if t.empty:
            continue

        tp = scheme.get('tp')
        eq, stats = backtest(t, all_dates_set, take_profit=tp)
        if stats:
            results[sname] = (eq, stats, t)
            print(f"  {sname:<35} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  "
                  f"MDD={stats['mdd']:>8.2%}  Calmar={stats['calmar']:>6.2f}  Trades={stats['trades']:>5d}", flush=True)

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
    plt.savefig(os.path.join(THIS_DIR, 'ts_backtest_v13_comparison.png'), dpi=150, bbox_inches='tight')

    print(f"\n{'Rank':>4} {'Scheme':<35} {'Total':>10} {'Ann':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'WinRate':>8} {'Trades':>7}")
    print('-' * 110)
    for rank, (sname, (eq, stats, tdf)) in enumerate(sorted_r, 1):
        print(f"{rank:>4} {sname:<35} {stats['total']:>9.2%} {stats['ann']:>9.2%} {stats['sharpe']:>7.2f} "
              f"{stats['mdd']:>9.2%} {stats['calmar']:>7.2f} {stats['win_rate']:>7.2%} {stats['trades']:>7}")

    for sname, (eq, stats, tdf) in sorted_r[:5]:
        if not eq.empty:
            eq.to_csv(os.path.join(THIS_DIR, f'equity_ts_v13_{sname}.csv'), index=False)

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
