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

COMBOS = [
    {'threshold': 0.50, 'max_pos': 3},
    {'threshold': 0.55, 'max_pos': 3},
    {'threshold': 0.60, 'max_pos': 3},
    {'threshold': 0.55, 'max_pos': 5},
    {'threshold': 0.55, 'max_pos': 10},
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

    for ts_code, group in feat.groupby('ts_code'):
        dates = group['trade_date'].values
        opens = group['open'].values
        closes = group['close'].values
        pre_closes = group['pre_close'].values if 'pre_close' in group.columns else [np.nan] * len(group)
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


def is_limit_up_by_open(ts_code, t1_open, t0_close):
    if t0_close <= 0:
        return False
    gap = (t1_open - t0_close) / t0_close
    if ts_code.startswith(('30', '68')):
        return gap >= 0.195
    return gap >= LIMIT_UP_THRESHOLD


def is_limit_down(ts_code, pct_chg):
    if pct_chg is None or (isinstance(pct_chg, float) and np.isnan(pct_chg)):
        return False
    if ts_code.startswith(('30', '68')):
        return pct_chg <= -0.195
    return pct_chg <= -0.095


def backtest_v2(pred_df, ohlc_lookup, pctchg_lookup, close_lookup,
                threshold, max_pos, stop_loss=0.0, hold_days=2):
    above = pred_df[pred_df['prob'] >= threshold].copy()
    above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
    selected = above[above['rank'] <= max_pos].copy()

    trading_dates = sorted(pred_df['ds'].unique())
    date_idx_map = {d: i for i, d in enumerate(trading_dates)}
    n_dates = len(trading_dates)

    pos_size = 1.0 / (hold_days * max_pos)
    n_pos = len(selected)
    if n_pos == 0:
        return {d: 0.0 for d in trading_dates}, trading_dates, {}, np.zeros(n_dates), np.zeros(n_dates, dtype=np.int32)

    entry_date_idx = np.array([date_idx_map[r['ds']] for _, r in selected.iterrows()], dtype=np.int32)
    ts_codes_arr = [r['ts_code'] for _, r in selected.iterrows()]
    buy_price = np.full(n_pos, np.nan, dtype=np.float64)
    last_price = np.full(n_pos, np.nan, dtype=np.float64)
    sl_price = np.full(n_pos, 0.0, dtype=np.float64)
    status = np.ones(n_pos, dtype=np.int8)
    daily_pnl = np.zeros(n_dates, dtype=np.float64)
    daily_capital_used = np.zeros(n_dates, dtype=np.float64)
    daily_n_positions = np.zeros(n_dates, dtype=np.int32)

    n_skip_t_limit = 0
    n_skip_t1_open_limit = 0
    n_skip_sell_limit = 0

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

            t0_close = close_lookup.get((ts_codes_arr[pos_i], trading_dates[entry_date_idx[pos_i]]))
            if t0_close is not None and t0_close > 0:
                if is_limit_up_by_open(ts_codes_arr[pos_i], o, t0_close):
                    n_skip_t1_open_limit += 1
                    status[pos_i] = 0
                    continue

            pct_t0 = pctchg_lookup.get((ts_codes_arr[pos_i], trading_dates[entry_date_idx[pos_i]]))
            if pct_t0 is not None:
                if ts_codes_arr[pos_i].startswith(('30', '68')):
                    if pct_t0 >= 0.195:
                        n_skip_t_limit += 1
                        status[pos_i] = 0
                        continue
                else:
                    if pct_t0 >= LIMIT_UP_THRESHOLD:
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
            for pos_i in open_idx[hold_days_all == 1]:
                if status[pos_i] == 1 and not np.isnan(buy_price[pos_i]):
                    daily_capital_used[day_i] += pos_size
                    daily_n_positions[day_i] += 1
            continue

        active_positions = open_idx[active_sub]
        active_hold = hold_days_all[active_sub]

        for pos_i in open_idx[hold_days_all == 1]:
            if status[pos_i] == 1 and not np.isnan(buy_price[pos_i]):
                daily_capital_used[day_i] += pos_size
                daily_n_positions[day_i] += 1

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
            at_limit_down = is_limit_down(ts_codes_arr[pos_i], pct_d)

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

            if status[pos_i] == 1 or not triggered:
                daily_capital_used[day_i] += pos_size
                daily_n_positions[day_i] += 1

    skip_stats = {
        'total_selected': n_pos,
        'skipped_T_limit_up': n_skip_t_limit,
        'skipped_T1_open_limit_up': n_skip_t1_open_limit,
        'skipped_sell_limit_down': n_skip_sell_limit,
    }
    return {d: float(daily_pnl[i]) for i, d in enumerate(trading_dates)}, trading_dates, skip_stats, daily_capital_used, daily_n_positions


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
                'win_rate_days': 0, 'n_days': n_days}, equity, drawdown
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
    print("Loading data...", flush=True)
    ohlc_lookup, pctchg_lookup, close_lookup = load_data()

    pred = pd.read_parquet(PRED_FILE)
    pred['ds'] = pred['trade_date'].astype(str)
    print(f"Predictions: {len(pred)} rows, {pred['ds'].min()}-{pred['ds'].max()}", flush=True)

    all_results = {}
    all_equities = {}
    all_capital = {}

    for ci, combo in enumerate(COMBOS):
        th = combo['threshold']
        mp = combo['max_pos']
        sl = combo.get('stop_loss', 0.0)
        sl_str = f"sl={sl:.0%}" if sl < 0 else "no-sl"
        label = f"1d_v2: th={th} pos={mp} {sl_str}"
        print(f"\n[{ci+1}/{len(COMBOS)}] {label}", flush=True)

        bt0 = time.time()
        daily_pnl, trading_dates, skip_stats, capital_used, n_positions = backtest_v2(
            pred, ohlc_lookup, pctchg_lookup, close_lookup, th, mp, sl)
        print(f"  backtest done in {time.time()-bt0:.0f}s", flush=True)
        print(f"  skip: T涨停={skip_stats['skipped_T_limit_up']}, "
              f"T+1开盘涨停={skip_stats['skipped_T1_open_limit_up']}, "
              f"跌停无法卖={skip_stats['skipped_sell_limit_down']}, "
              f"总选中={skip_stats['total_selected']}", flush=True)

        dates = pd.to_datetime(trading_dates, format='%Y%m%d')
        cap_series = pd.Series(capital_used, index=dates)
        npos_series = pd.Series(n_positions, index=dates)

        print(f"  日均持仓: {npos_series.mean():.2f} 只", flush=True)
        print(f"  平均资金占用: {cap_series.mean()*100:.1f}%", flush=True)
        print(f"  全空仓天数: {(npos_series==0).sum()} / {len(npos_series)} ({(npos_series==0).mean()*100:.1f}%)", flush=True)

        for period_name, start, end in PERIODS:
            mask_dates = [d for d in trading_dates if start <= d <= end]
            if not mask_dates:
                continue
            period_pnl = {d: daily_pnl.get(d, 0.0) for d in mask_dates}
            stats, equity, dd = calc_stats(period_pnl, mask_dates)

            mask_idx = [i for i, d in enumerate(trading_dates) if start <= d <= end]
            period_cap = capital_used[mask_idx] if len(mask_idx) > 0 else np.array([])
            period_npos = n_positions[mask_idx] if len(mask_idx) > 0 else np.array([])

            stats['avg_capital_utilization'] = float(np.mean(period_cap)) if len(period_cap) > 0 else 0
            stats['avg_daily_positions'] = float(np.mean(period_npos)) if len(period_npos) > 0 else 0
            stats['days_zero_positions'] = int(np.sum(period_npos == 0)) if len(period_npos) > 0 else 0
            stats['total_trades'] = skip_stats['total_selected']
            stats['trades_per_day'] = float(skip_stats['total_selected'] / len(mask_dates)) if len(mask_dates) > 0 else 0

            key = f"{label} | {period_name}"
            all_results[key] = {**stats, 'label': label, 'period': period_name,
                                 'threshold': th, 'max_pos': mp, 'stop_loss': sl}
            if period_name == 'full_2022_2026':
                all_equities[label] = equity
                all_capital[label] = cap_series

            print(f"  {period_name}: CAGR={stats['cagr']:.1%}, Sharpe={stats['sharpe']:.2f}, "
                  f"MaxDD={stats['max_dd']:.1%}, 资金占用={stats['avg_capital_utilization']*100:.1f}%, "
                  f"日均持仓={stats['avg_daily_positions']:.1f}", flush=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1 = axes[0]
    for label, equity in all_equities.items():
        s = all_results[f"{label} | full_2022_2026"]
        ax1.plot(equity.index, equity.values,
                 label=f"{label} (CAGR={s['cagr']:.1%}, Sharpe={s['sharpe']:.2f}, "
                       f"资金占用={s['avg_capital_utilization']*100:.0f}%)",
                 linewidth=1.2, alpha=0.8)
    ax1.set_title('1D Strategy V2 (Fixed: T+1 Open Limit-Up Check)\n'
                  'Target: (T+2 close - T+1 open)/T+1 open, Entry: T+1 open', fontsize=13)
    ax1.set_ylabel('Equity')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    ax2 = axes[1]
    for label, cap in all_capital.items():
        ax2.plot(cap.index, cap.values * 100, label=label, linewidth=0.8, alpha=0.7)
    ax2.set_title('Capital Utilization (%)', fontsize=12)
    ax2.set_ylabel('% of Total Capital')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    for label, equity in all_equities.items():
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max
        ax3.fill_between(dd.index, dd.values, 0, alpha=0.3, label=label)
    ax3.set_title('Drawdown', fontsize=12)
    ax3.set_ylabel('Drawdown')
    ax3.legend(fontsize=7, loc='lower left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'backtest_v2_fixed.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fname}", flush=True)

    results_file = os.path.join(RESULTS_DIR, 'backtest_v2_fixed_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {results_file}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("资金利用率总结", flush=True)
    print("=" * 80, flush=True)
    for label in all_equities:
        s = all_results[f"{label} | full_2022_2026"]
        cap_util = s['avg_capital_utilization']
        cagr = s['cagr']
        implied = cagr / cap_util if cap_util > 0 else 0
        print(f"\n{label}:", flush=True)
        print(f"  总账户CAGR: {cagr*100:.1f}%", flush=True)
        print(f"  平均资金占用: {cap_util*100:.1f}%", flush=True)
        print(f"  投入资金隐含CAGR: {implied*100:.1f}%", flush=True)
        print(f"  日均持仓: {s['avg_daily_positions']:.1f} 只", flush=True)
        print(f"  全空仓天数: {s['days_zero_positions']}", flush=True)
        print(f"  日均交易: {s['trades_per_day']:.2f} 笔", flush=True)

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    run()
