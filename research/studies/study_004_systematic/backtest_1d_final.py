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
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

BUY_COST = 0.001
SELL_COST = 0.001
LIMIT_UP_THRESHOLD = 0.095

COMBOS = [
    {'threshold': 0.50, 'max_pos': 3},
    {'threshold': 0.55, 'max_pos': 3},
    {'threshold': 0.60, 'max_pos': 3},
    {'threshold': 0.50, 'max_pos': 5},
    {'threshold': 0.55, 'max_pos': 5},
    {'threshold': 0.60, 'max_pos': 5},
    {'threshold': 0.55, 'max_pos': 3, 'stop_loss': -0.05},
    {'threshold': 0.55, 'max_pos': 5, 'stop_loss': -0.05},
]

PERIODS = [
    ('train_2022_2024', '20220101', '20241231'),
    ('test_2025_2026', '20250101', '20261231'),
    ('full_2022_2026', '20220101', '20261231'),
]


def load_ohlc_and_pctchg():
    feat = pd.read_parquet(FEATURES_FILE)
    avail = [c for c in ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'pct_chg', 'pre_close'] if c in feat.columns]
    feat = feat[avail].copy()
    feat['trade_date'] = feat['trade_date'].astype(str)
    feat = feat.sort_values(['ts_code', 'trade_date'])
    feat = feat.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')

    ohlc_lookup = {}
    pctchg_lookup = {}
    for _, row in feat.iterrows():
        key = (row['ts_code'], row['trade_date'])
        ohlc_lookup[key] = (row['open'], row['high'], row['low'], row['close'])
        if 'pct_chg' in row.index and pd.notna(row['pct_chg']):
            pctchg_lookup[key] = row['pct_chg']
        elif 'pre_close' in row.index and pd.notna(row['pre_close']) and row['pre_close'] > 0:
            pctchg_lookup[key] = (row['close'] - row['pre_close']) / row['pre_close']

    print(f"OHLC: {len(ohlc_lookup)} entries, pct_chg: {len(pctchg_lookup)} entries", flush=True)
    return ohlc_lookup, pctchg_lookup


def is_limit_up(ts_code, pct_chg):
    if pd.isna(pct_chg):
        return False
    if ts_code.startswith(('30', '68')):
        return pct_chg >= 0.195
    return pct_chg >= LIMIT_UP_THRESHOLD


def is_limit_down(ts_code, pct_chg):
    if pd.isna(pct_chg):
        return False
    if ts_code.startswith(('30', '68')):
        return pct_chg <= -0.195
    return pct_chg <= -0.095


def backtest_1d(pred_df, ohlc_lookup, pctchg_lookup, threshold, max_pos, stop_loss=0.0):
    hold_days = 2
    above = pred_df[pred_df['prob'] >= threshold].copy()
    above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
    selected = above[above['rank'] <= max_pos].copy()

    n_skip_t_limit = 0
    n_skip_t1_limit = 0
    n_skip_sell_limit = 0
    n_total = len(selected)

    trading_dates = sorted(pred_df['ds'].unique())
    date_idx_map = {d: i for i, d in enumerate(trading_dates)}
    n_dates = len(trading_dates)

    pos_size = 1.0 / (hold_days * max_pos)
    n_pos = n_total
    if n_pos == 0:
        return {d: 0.0 for d in trading_dates}, trading_dates, {}

    entry_date_idx = np.array([date_idx_map[r['ds']] for _, r in selected.iterrows()], dtype=np.int32)
    ts_codes_arr = [r['ts_code'] for _, r in selected.iterrows()]
    buy_price = np.full(n_pos, np.nan, dtype=np.float64)
    last_price = np.full(n_pos, np.nan, dtype=np.float64)
    sl_price = np.full(n_pos, 0.0, dtype=np.float64)
    status = np.ones(n_pos, dtype=np.int8)
    daily_pnl = np.zeros(n_dates, dtype=np.float64)

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
                continue
            o, h, l, c = ohlc

            pct_t1 = pctchg_lookup.get((ts_codes_arr[pos_i], d))
            if pct_t1 is not None and is_limit_up(ts_codes_arr[pos_i], pct_t1):
                n_skip_t1_limit += 1
                status[pos_i] = 0
                continue

            pct_t0 = pctchg_lookup.get((ts_codes_arr[pos_i], trading_dates[entry_date_idx[pos_i]]))
            if pct_t0 is not None and is_limit_up(ts_codes_arr[pos_i], pct_t0):
                n_skip_t_limit += 1
                status[pos_i] = 0
                continue

            bp = o
            buy_price[pos_i] = bp
            last_price[pos_i] = bp
            if stop_loss < 0:
                sl_price[pos_i] = bp * (1 + stop_loss)
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
            triggered = False

            pct_d = pctchg_lookup.get((ts_codes_arr[pos_i], d))
            at_limit_down = pct_d is not None and is_limit_down(ts_codes_arr[pos_i], pct_d)

            if sl_price[pos_i] > 0 and o <= sl_price[pos_i]:
                if at_limit_down:
                    daily_pnl[day_i] += pos_size * (c - prev) / prev
                    last_price[pos_i] = c
                    n_skip_sell_limit += 1
                else:
                    daily_pnl[day_i] += pos_size * (o - prev) / prev - pos_size * SELL_COST
                    status[pos_i] = 0
                    last_price[pos_i] = o
                    triggered = True
            elif sl_price[pos_i] > 0 and l <= sl_price[pos_i] and not at_limit_down:
                daily_pnl[day_i] += pos_size * (sl_price[pos_i] - prev) / prev - pos_size * SELL_COST
                status[pos_i] = 0
                last_price[pos_i] = sl_price[pos_i]
                triggered = True

            if not triggered:
                if hd == hold_days:
                    if at_limit_down:
                        daily_pnl[day_i] += pos_size * (c - prev) / prev
                        last_price[pos_i] = c
                        n_skip_sell_limit += 1
                    else:
                        daily_pnl[day_i] += pos_size * (c - prev) / prev - pos_size * SELL_COST
                        status[pos_i] = 0
                        last_price[pos_i] = c
                else:
                    daily_pnl[day_i] += pos_size * (c - prev) / prev
                    last_price[pos_i] = c

    skip_stats = {
        'total_selected': n_total,
        'skipped_T_limit_up': n_skip_t_limit,
        'skipped_T1_limit_up': n_skip_t1_limit,
        'skipped_sell_limit_down': n_skip_sell_limit,
    }
    return {d: float(daily_pnl[i]) for i, d in enumerate(trading_dates)}, trading_dates, skip_stats


def calc_stats(daily_pnl, trading_dates):
    dates = pd.to_datetime(trading_dates, format='%Y%m%d')
    pnl_s = pd.Series([daily_pnl.get(d, 0.0) for d in trading_dates], index=dates)
    equity = (1 + pnl_s).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    n_days = len(pnl_s)
    n_years = n_days / 252
    total_return = equity.iloc[-1] - 1
    cagr = (equity.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    max_dd = drawdown.min()
    std = pnl_s.std()
    sharpe = (pnl_s.mean() / std * np.sqrt(252)) if std > 1e-10 else 0
    win_rate = (pnl_s > 0).mean()
    monthly_rets = []
    for period, group in pnl_s.groupby(pnl_s.index.to_period('M')):
        monthly_rets.append((1 + group).prod() - 1)
    monthly_win = np.mean([1 if r > 0 else 0 for r in monthly_rets]) if monthly_rets else 0
    return {
        'cagr': float(cagr), 'sharpe': float(sharpe), 'max_dd': float(max_dd),
        'total_return': float(total_return), 'win_rate_days': float(win_rate),
        'monthly_win_rate': float(monthly_win), 'n_days': int(n_days), 'n_months': len(monthly_rets),
    }, equity, drawdown


def run():
    t0 = time.time()
    print("Loading OHLC + pct_chg...", flush=True)
    ohlc_lookup, pctchg_lookup = load_ohlc_and_pctchg()

    pred_file = os.path.join(STUDY_DIR, 'predictions', 'predictions_1d_open_wf_monthly.parquet')
    pred = pd.read_parquet(pred_file)
    pred['ds'] = pred['trade_date'].astype(str)
    print(f"Predictions: {len(pred)} rows, {pred['ds'].min()}-{pred['ds'].max()}", flush=True)

    all_results = {}
    all_equities = {}

    for ci, combo in enumerate(COMBOS):
        th = combo['threshold']
        mp = combo['max_pos']
        sl = combo.get('stop_loss', 0.0)
        sl_str = f"sl={sl:.0%}" if sl < 0 else "no-sl"
        label = f"1d: th={th} pos={mp} {sl_str}"
        print(f"\n[{ci+1}/{len(COMBOS)}] {label}", flush=True)

        bt0 = time.time()
        daily_pnl, trading_dates, skip_stats = backtest_1d(
            pred, ohlc_lookup, pctchg_lookup, th, mp, sl)
        print(f"  backtest done in {time.time()-bt0:.0f}s", flush=True)
        print(f"  skip: T涨停={skip_stats['skipped_T_limit_up']}, "
              f"T+1涨停={skip_stats['skipped_T1_limit_up']}, "
              f"跌停无法卖={skip_stats['skipped_sell_limit_down']}, "
              f"总选中={skip_stats['total_selected']}", flush=True)

        for period_name, start, end in PERIODS:
            mask_dates = [d for d in trading_dates if start <= d <= end]
            if not mask_dates:
                continue
            period_pnl = {d: daily_pnl.get(d, 0.0) for d in mask_dates}
            stats, equity, dd = calc_stats(period_pnl, mask_dates)
            key = f"{label} | {period_name}"
            all_results[key] = {**stats, 'label': label, 'period': period_name,
                                 'threshold': th, 'max_pos': mp, 'stop_loss': sl}
            if period_name == 'full_2022_2026':
                all_equities[label] = equity
            print(f"  {period_name}: CAGR={stats['cagr']:.1%}, Sharpe={stats['sharpe']:.2f}, "
                  f"MaxDD={stats['max_dd']:.1%}, WinRate={stats['win_rate_days']:.1%}", flush=True)

    best_label = None
    best_sharpe = -999
    for key, s in all_results.items():
        if s['period'] == 'test_2025_2026' and s['cagr'] > 0 and s['sharpe'] > best_sharpe:
            best_sharpe = s['sharpe']
            best_label = s['label']

    print(f"\nBest (test period): {best_label} (Sharpe={best_sharpe:.2f})", flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    ax1 = axes[0]
    for label, equity in all_equities.items():
        s = all_results[f"{label} | full_2022_2026"]
        if s['sharpe'] > 0.8:
            ax1.plot(equity.index, equity.values,
                     label=f"{label} (CAGR={s['cagr']:.1%}, Sharpe={s['sharpe']:.2f})",
                     linewidth=1.2, alpha=0.8)
    ax1.set_title('1D Strategy Equity Curves (Limit Up/Down Filtered)\nTarget: (T+2 close - T+1 open)/T+1 open, Entry: T+1 open, T+1 constraint', fontsize=13)
    ax1.set_ylabel('Equity')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    ax2 = axes[1]
    for label, equity in all_equities.items():
        s = all_results[f"{label} | full_2022_2026"]
        if s['sharpe'] > 0.8:
            running_max = equity.cummax()
            dd = (equity - running_max) / running_max
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, label=label)
    ax2.set_title('Drawdown', fontsize=12)
    ax2.set_ylabel('Drawdown')
    ax2.legend(fontsize=7, loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, '1d_strategy_limit_filter.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fname}", flush=True)

    results_file = os.path.join(RESULTS_DIR, '1d_strategy_limit_filter_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {results_file}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    run()
