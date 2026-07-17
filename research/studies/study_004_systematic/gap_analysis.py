import os, sys, time, json, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_FILE = os.path.join(STUDY_DIR, 'data', 'all_features_v2.parquet')
PRED_FILE = os.path.join(STUDY_DIR, 'predictions', 'predictions_1d_open_wf_monthly.parquet')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

BUY_COST = 0.001
SELL_COST = 0.001
LIMIT_UP_THRESHOLD = 0.095

GAP_BINS = [
    ('gap<-5%', -np.inf, -0.05),
    ('-5%<=gap<-2%', -0.05, -0.02),
    ('-2%<=gap<0%', -0.02, 0.0),
    ('0%<=gap<2%', 0.0, 0.02),
    ('2%<=gap<5%', 0.02, 0.05),
    ('5%<=gap<6%', 0.05, 0.06),
    ('2%<=gap<6%(最优)', 0.02, 0.06),
    ('6%<=gap<7%', 0.06, 0.07),
    ('7%<=gap<8%', 0.07, 0.08),
    ('8%<=gap<9%', 0.08, 0.09),
    ('9%<=gap<9.5%', 0.09, 0.095),
    ('gap>=9.5%(涨停区)', 0.095, np.inf),
]

PERIODS = [
    ('validation_2022_2024', '20220101', '20241231'),
    ('test_2025_2026', '20250101', '20261231'),
    ('full_2022_2026', '20220101', '20261231'),
]


def load_data():
    feat = pd.read_parquet(FEATURES_FILE)
    feat['trade_date'] = feat['trade_date'].astype(str)
    feat = feat.sort_values(['ts_code', 'trade_date'])
    feat = feat.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')

    ohlc_lookup = {}
    pctchg_lookup = {}
    close_lookup = {}

    feat_sorted = feat.sort_values(['ts_code', 'trade_date'])
    for ts_code, group in feat_sorted.groupby('ts_code'):
        dates = group['trade_date'].values
        opens = group['open'].values
        closes = group['close'].values
        pct_chgs = group['pct_chg'].values if 'pct_chg' in group.columns else [np.nan] * len(group)
        for i in range(len(dates)):
            d = str(dates[i])
            key = (ts_code, d)
            ohlc_lookup[key] = (float(opens[i]), float(group['high'].values[i]),
                                float(group['low'].values[i]), float(closes[i]))
            close_lookup[key] = float(closes[i])
            pctchg_lookup[key] = float(pct_chgs[i]) if not np.isnan(pct_chgs[i]) else None

    print(f"OHLC: {len(ohlc_lookup)} entries", flush=True)
    return ohlc_lookup, pctchg_lookup, close_lookup


def is_limit_up(ts_code, pct_chg):
    if pct_chg is None or (isinstance(pct_chg, float) and np.isnan(pct_chg)):
        return False
    if ts_code.startswith(('30', '68')):
        return pct_chg >= 0.195
    return pct_chg >= LIMIT_UP_THRESHOLD


def is_limit_down(ts_code, pct_chg):
    if pct_chg is None or (isinstance(pct_chg, float) and np.isnan(pct_chg)):
        return False
    if ts_code.startswith(('30', '68')):
        return pct_chg <= -0.195
    return pct_chg <= -0.095


def backtest_by_gap(pred_df, ohlc_lookup, pctchg_lookup, close_lookup,
                    threshold, max_pos, gap_min, gap_max, hold_days=2):
    above = pred_df[pred_df['prob'] >= threshold].copy()
    above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
    selected = above[above['rank'] <= max_pos].copy()

    trading_dates = sorted(pred_df['ds'].unique())
    date_idx_map = {d: i for i, d in enumerate(trading_dates)}
    n_dates = len(trading_dates)

    pos_size = 1.0 / (hold_days * max_pos)

    filtered = []
    gap_values = []
    for _, row in selected.iterrows():
        ts_code = row['ts_code']
        ds = row['ds']
        t_close = close_lookup.get((ts_code, ds))
        if t_close is None or t_close <= 0:
            continue

        ds_idx = date_idx_map.get(ds)
        if ds_idx is None or ds_idx + 1 >= len(trading_dates):
            continue
        next_d = trading_dates[ds_idx + 1]

        t1_ohlc = ohlc_lookup.get((ts_code, next_d))
        if t1_ohlc is None:
            continue

        t1_open_price = t1_ohlc[0]
        gap = (t1_open_price - t_close) / t_close

        if gap_min <= gap < gap_max:
            filtered.append(row)
            gap_values.append(gap)

    if not filtered:
        return None, 0, []

    filtered_df = pd.DataFrame(filtered)
    n_pos = len(filtered_df)

    entry_date_idx = np.array([date_idx_map[r['ds']] for _, r in filtered_df.iterrows()], dtype=np.int32)
    ts_codes_arr = [r['ts_code'] for _, r in filtered_df.iterrows()]
    buy_price = np.full(n_pos, np.nan, dtype=np.float64)
    sell_price = np.full(n_pos, np.nan, dtype=np.float64)
    last_price = np.full(n_pos, np.nan, dtype=np.float64)
    status = np.ones(n_pos, dtype=np.int8)
    daily_pnl = np.zeros(n_dates, dtype=np.float64)
    trade_returns = [np.nan] * n_pos
    trade_skipped = [False] * n_pos

    for day_i, d in enumerate(trading_dates):
        open_mask = status == 1
        if not open_mask.any():
            continue
        open_idx = np.where(open_mask)[0]
        hold_days_all = day_i - entry_date_idx[open_idx]

        buy_mask = hold_days_all == 1
        for pos_i in open_idx[buy_mask]:
            ohlc = ohlc_lookup.get((ts_codes_arr[pos_i], d))
            if ohlc is None:
                status[pos_i] = 0
                trade_skipped[pos_i] = True
                continue
            o, h, l, c = ohlc

            pct_t1 = pctchg_lookup.get((ts_codes_arr[pos_i], d))
            if is_limit_up(ts_codes_arr[pos_i], pct_t1):
                status[pos_i] = 0
                trade_skipped[pos_i] = True
                continue

            pct_t0 = pctchg_lookup.get((ts_codes_arr[pos_i], trading_dates[entry_date_idx[pos_i]]))
            if is_limit_up(ts_codes_arr[pos_i], pct_t0):
                status[pos_i] = 0
                trade_skipped[pos_i] = True
                continue

            bp = o
            buy_price[pos_i] = bp
            last_price[pos_i] = bp
            daily_pnl[day_i] -= pos_size * BUY_COST
            daily_pnl[day_i] += pos_size * (c - bp) / bp
            last_price[pos_i] = c

        active_sub = (hold_days_all >= 2) & (hold_days_all <= hold_days)
        if not active_sub.any():
            continue
        active_positions = open_idx[active_sub]
        active_hold = hold_days_all[active_sub]

        for j in range(len(active_positions)):
            pos_i = active_positions[j]
            hd = active_hold[j]
            ohlc = ohlc_lookup.get((ts_codes_arr[pos_i], d))
            if ohlc is None:
                daily_pnl[day_i] -= pos_size * SELL_COST
                status[pos_i] = 0
                continue
            o, h, l, c = ohlc
            prev = last_price[pos_i]

            pct_d = pctchg_lookup.get((ts_codes_arr[pos_i], d))
            at_limit_down = is_limit_down(ts_codes_arr[pos_i], pct_d)

            if hd == hold_days:
                if at_limit_down:
                    daily_pnl[day_i] += pos_size * (c - prev) / prev
                    last_price[pos_i] = c
                else:
                    daily_pnl[day_i] += pos_size * (c - prev) / prev - pos_size * SELL_COST
                    status[pos_i] = 0
                    sell_price[pos_i] = c
                    last_price[pos_i] = c
                    if not np.isnan(buy_price[pos_i]) and buy_price[pos_i] > 0:
                        ret = (c - buy_price[pos_i]) / buy_price[pos_i] - BUY_COST - SELL_COST
                        trade_returns[pos_i] = ret
            else:
                daily_pnl[day_i] += pos_size * (c - prev) / prev
                last_price[pos_i] = c

    valid_trades = []
    for i in range(n_pos):
        if not trade_skipped[i] and not np.isnan(trade_returns[i]):
            valid_trades.append({
                'ts_code': ts_codes_arr[i],
                'ds': trading_dates[entry_date_idx[i]],
                'gap': gap_values[i],
                'buy_price': float(buy_price[i]),
                'sell_price': float(sell_price[i]),
                'return': float(trade_returns[i]),
            })

    n_executed = len(valid_trades)
    n_skipped = sum(trade_skipped)

    return {d: float(daily_pnl[i]) for i, d in enumerate(trading_dates)}, n_pos, valid_trades, n_executed, n_skipped


def calc_stats(daily_pnl, trading_dates):
    dates = pd.to_datetime(trading_dates, format='%Y%m%d')
    pnl_s = pd.Series([daily_pnl.get(d, 0.0) for d in trading_dates], index=dates)
    equity = (1 + pnl_s).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    n_days = len(pnl_s)
    n_years = n_days / 252
    if n_years <= 0 or equity.iloc[-1] <= 0:
        return {'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'total_return': 0,
                'win_rate_days': 0, 'n_days': n_days}
    total_return = equity.iloc[-1] - 1
    cagr = (equity.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    max_dd = drawdown.min()
    std = pnl_s.std()
    sharpe = (pnl_s.mean() / std * np.sqrt(252)) if std > 1e-10 else 0
    win_rate = (pnl_s > 0).mean()
    return {
        'cagr': float(cagr), 'sharpe': float(sharpe), 'max_dd': float(max_dd),
        'total_return': float(total_return), 'win_rate_days': float(win_rate),
        'n_days': int(n_days),
    }


def run():
    t0 = time.time()
    print("Loading data...", flush=True)
    ohlc_lookup, pctchg_lookup, close_lookup = load_data()

    pred = pd.read_parquet(PRED_FILE)
    pred['ds'] = pred['trade_date'].astype(str)
    print(f"Predictions: {len(pred)} rows", flush=True)

    threshold = 0.55
    max_pos = 3

    print(f"\nStrategy: threshold={threshold}, max_pos={max_pos}", flush=True)
    print(f"Analyzing gap = (T+1 open - T close) / T close impact on returns\n", flush=True)

    all_results = {}
    all_trades_by_gap = {}

    for gap_name, gap_min, gap_max in GAP_BINS:
        print(f"\n{'='*60}", flush=True)
        print(f"Gap range: {gap_name} ({gap_min:.2%} to {gap_max:.2%})", flush=True)
        print(f"{'='*60}", flush=True)

        daily_pnl, n_selected, trades, n_executed, n_skipped = backtest_by_gap(
            pred, ohlc_lookup, pctchg_lookup, close_lookup,
            threshold, max_pos, gap_min, gap_max)

        if daily_pnl is None or n_selected == 0:
            print(f"  No trades in this gap range!", flush=True)
            for period_name, start, end in PERIODS:
                all_results[f"{gap_name}|{period_name}"] = {
                    'gap_range': gap_name, 'period': period_name,
                    'cagr': 0, 'sharpe': 0, 'max_dd': 0,
                    'n_selected': 0, 'n_executed': 0, 'n_skipped': 0,
                    'n_win': 0, 'n_loss': 0, 'win_rate_trades': 0,
                    'avg_return': 0, 'median_return': 0,
                }
            continue

        print(f"  Selected: {n_selected}, Executed: {n_executed}, Skipped(limit): {n_skipped}", flush=True)

        all_trades_by_gap[gap_name] = trades

        for period_name, start, end in PERIODS:
            mask_dates = [d for d in sorted(daily_pnl.keys()) if start <= d <= end]
            if not mask_dates:
                continue
            period_pnl = {d: daily_pnl.get(d, 0.0) for d in mask_dates}
            stats = calc_stats(period_pnl, mask_dates)
            stats['gap_range'] = gap_name
            stats['period'] = period_name
            stats['n_selected'] = n_selected
            stats['n_executed'] = n_executed
            stats['n_skipped'] = n_skipped

            period_trades = [t for t in trades if start <= t['ds'] <= end]
            if period_trades:
                rets = [t['return'] for t in period_trades]
                n_win = sum(1 for r in rets if r > 0)
                n_loss = sum(1 for r in rets if r <= 0)
                stats['n_win'] = n_win
                stats['n_loss'] = n_loss
                stats['win_rate_trades'] = float(n_win / len(rets)) if len(rets) > 0 else 0
                stats['avg_return'] = float(np.mean(rets))
                stats['median_return'] = float(np.median(rets))
                stats['avg_win'] = float(np.mean([r for r in rets if r > 0])) if n_win > 0 else 0
                stats['avg_loss'] = float(np.mean([r for r in rets if r <= 0])) if n_loss > 0 else 0
                stats['max_win'] = float(max(rets))
                stats['max_loss'] = float(min(rets))
            else:
                stats['n_win'] = 0
                stats['n_loss'] = 0
                stats['win_rate_trades'] = 0
                stats['avg_return'] = 0
                stats['median_return'] = 0
                stats['avg_win'] = 0
                stats['avg_loss'] = 0
                stats['max_win'] = 0
                stats['max_loss'] = 0

            all_results[f"{gap_name}|{period_name}"] = stats
            print(f"  {period_name}: CAGR={stats['cagr']:.1%}, Sharpe={stats['sharpe']:.2f}, "
                  f"Win={stats['n_win']}, Loss={stats['n_loss']}, "
                  f"WinRate(trade)={stats['win_rate_trades']:.1%}, "
                  f"AvgRet={stats['avg_return']:.2%}", flush=True)

    print(f"\n\n{'='*100}", flush=True)
    print("SUMMARY TABLE (per-trade stats)", flush=True)
    print(f"{'='*100}", flush=True)

    for period_name, _, _ in PERIODS:
        print(f"\n--- {period_name} ---", flush=True)
        header = (f"{'Gap Range':<22} {'Sel':>4} {'Exe':>4} {'Skip':>4} "
                  f"{'Win':>4} {'Loss':>4} {'WinR%':>6} "
                  f"{'AvgRet%':>8} {'MedRet%':>8} {'AvgWin%':>8} {'AvgLoss%':>8} "
                  f"{'CAGR%':>7} {'Sharpe':>7}")
        print(header, flush=True)
        print('-' * 110, flush=True)
        for gap_name, _, _ in GAP_BINS:
            key = f"{gap_name}|{period_name}"
            if key in all_results:
                r = all_results[key]
                print(f"{gap_name:<22} {r['n_selected']:>4} {r['n_executed']:>4} {r['n_skipped']:>4} "
                      f"{r['n_win']:>4} {r['n_loss']:>4} {r['win_rate_trades']*100:>5.1f}% "
                      f"{r['avg_return']*100:>7.2f}% {r['median_return']*100:>7.2f}% "
                      f"{r['avg_win']*100:>7.2f}% {r['avg_loss']*100:>7.2f}% "
                      f"{r['cagr']*100:>6.1f}% {r['sharpe']:>6.2f}", flush=True)

    fig, axes = plt.subplots(3, 1, figsize=(18, 20), gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1 = axes[0]
    for gap_name, gap_min, gap_max in GAP_BINS:
        key_full = f"{gap_name}|full_2022_2026"
        if key_full not in all_results:
            continue
        r = all_results[key_full]
        if r['n_executed'] == 0:
            continue

        daily_pnl, _, _, _, _ = backtest_by_gap(
            pred, ohlc_lookup, pctchg_lookup, close_lookup,
            threshold, max_pos, gap_min, gap_max)
        if daily_pnl is None:
            continue

        trading_dates = sorted(daily_pnl.keys())
        dates = pd.to_datetime(trading_dates, format='%Y%m%d')
        pnl_s = pd.Series([daily_pnl.get(d, 0.0) for d in trading_dates], index=dates)
        equity = (1 + pnl_s).cumprod()

        ax1.plot(equity.index, equity.values,
                 label=f"{gap_name} (CAGR={r['cagr']:.1%}, Sharpe={r['sharpe']:.2f})",
                 linewidth=1.3, alpha=0.8)

    ax1.set_title(f'1D Strategy Equity by Gap Range (T+1 Open vs T Close)\n'
                  f'threshold={threshold}, max_pos={max_pos}, limit-up/down filtered', fontsize=13)
    ax1.set_ylabel('Equity')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    ax2 = axes[1]
    gap_labels = [g[0] for g in GAP_BINS]
    val_cagrs = []
    test_cagrs = []
    for gap_name, _, _ in GAP_BINS:
        val_key = f"{gap_name}|validation_2022_2024"
        test_key = f"{gap_name}|test_2025_2026"
        val_cagrs.append(all_results.get(val_key, {}).get('cagr', 0) * 100)
        test_cagrs.append(all_results.get(test_key, {}).get('cagr', 0) * 100)

    x = np.arange(len(gap_labels))
    width = 0.35
    ax2.bar(x - width/2, val_cagrs, width, label='Validation (2022-2024)', alpha=0.8)
    ax2.bar(x + width/2, test_cagrs, width, label='Test (2025-2026)', alpha=0.8)
    ax2.set_xlabel('Gap Range')
    ax2.set_ylabel('CAGR (%)')
    ax2.set_title('CAGR by Gap Range: Validation vs Test')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gap_labels, rotation=30, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)

    ax3 = axes[2]
    n_wins = []
    n_losses = []
    n_skips = []
    for gap_name, _, _ in GAP_BINS:
        full_key = f"{gap_name}|full_2022_2026"
        r = all_results.get(full_key, {})
        n_wins.append(r.get('n_win', 0))
        n_losses.append(r.get('n_loss', 0))
        n_skips.append(r.get('n_skipped', 0))

    ax3.bar(x - width, n_wins, width, label='Win', color='green', alpha=0.7)
    ax3.bar(x, n_losses, width, label='Loss', color='red', alpha=0.7)
    ax3.bar(x + width, n_skips, width, label='Skipped(limit)', color='gray', alpha=0.7)
    ax3.set_xlabel('Gap Range')
    ax3.set_ylabel('Trade Count')
    ax3.set_title('Trade Counts by Gap Range (Win / Loss / Skipped)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(gap_labels, rotation=30, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'gap_analysis.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fname}", flush=True)

    results_file = os.path.join(RESULTS_DIR, 'gap_analysis_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {results_file}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    run()
