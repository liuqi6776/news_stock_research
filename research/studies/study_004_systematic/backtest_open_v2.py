"""
5d & 28d 回测 (正确target: return_Xd_open, d1_open入场, 逐日盯市)

使用新训练的模型预测, target = (T+N_close - T+1_open) / T+1_open
入场价 = T+1日开盘 (next_open), 与target一致

运行:
  cd study_004_systematic
  python -u backtest_open_v2.py
"""
import os
import sys
import pandas as pd
import numpy as np
import json
import time
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

STRATEGIES = [
    {
        'name': '5d_open',
        'pred_file': os.path.join(STUDY_DIR, 'predictions', 'predictions_5d_open_wf_monthly.parquet'),
        'hold_days': 5,
        'combos': [
            {'threshold': 0.50, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.05, 'label': '5d: th=0.50 pos=3 tp=5%'},
            {'threshold': 0.55, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.05, 'label': '5d: th=0.55 pos=3 tp=5%'},
            {'threshold': 0.60, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.05, 'label': '5d: th=0.60 pos=3 tp=5%'},
            {'threshold': 0.50, 'max_pos': 3, 'stop_loss': -0.10, 'take_profit': 0.05, 'label': '5d: th=0.50 pos=3 sl=-10% tp=5%'},
            {'threshold': 0.50, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.0,  'label': '5d: th=0.50 pos=3 no-sltp'},
            {'threshold': 0.60, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.0,  'label': '5d: th=0.60 pos=3 no-sltp'},
        ]
    },
    {
        'name': '28d_open',
        'pred_file': os.path.join(STUDY_DIR, 'predictions', 'predictions_28d_open_wf_monthly.parquet'),
        'hold_days': 28,
        'combos': [
            {'threshold': 0.40, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.15, 'label': '28d: th=0.40 pos=3 tp=15%'},
            {'threshold': 0.50, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.15, 'label': '28d: th=0.50 pos=3 tp=15%'},
            {'threshold': 0.60, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.15, 'label': '28d: th=0.60 pos=3 tp=15%'},
            {'threshold': 0.40, 'max_pos': 3, 'stop_loss': -0.10, 'take_profit': 0.15, 'label': '28d: th=0.40 pos=3 sl=-10% tp=15%'},
            {'threshold': 0.40, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.0,  'label': '28d: th=0.40 pos=3 no-sltp'},
            {'threshold': 0.50, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.0,  'label': '28d: th=0.50 pos=3 no-sltp'},
        ]
    },
]


def load_ohlc_index(pred_ts_codes=None):
    print("Loading OHLC data...", flush=True)
    feat = pd.read_parquet(FEATURES_FILE)
    cols = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close']
    feat = feat[cols].copy()
    feat['trade_date'] = feat['trade_date'].astype(str)
    if pred_ts_codes is not None:
        feat = feat[feat['ts_code'].isin(pred_ts_codes)].copy()
    feat = feat.sort_values(['ts_code', 'trade_date'])
    feat = feat.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    print(f"  OHLC: {len(feat)} rows", flush=True)
    return feat


def backtest(pred_df, ohlc_df, threshold, max_pos, stop_loss, take_profit, hold_days):
    above = pred_df[pred_df['prob'] >= threshold].copy()
    above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
    selected = above[above['rank'] <= max_pos].copy()
    print(f"    Selected: {len(selected)} positions", flush=True)

    trading_dates = sorted(pred_df['ds'].unique())
    date_idx_map = {d: i for i, d in enumerate(trading_dates)}
    n_dates = len(trading_dates)

    ohlc_lookup = {}
    for _, row in ohlc_df.iterrows():
        ohlc_lookup[(row['ts_code'], row['trade_date'])] = (row['open'], row['high'], row['low'], row['close'])

    pos_size = 1.0 / (hold_days * max_pos)

    n_pos = len(selected)
    if n_pos == 0:
        return {d: 0.0 for d in trading_dates}, trading_dates

    entry_date_idx = np.array([date_idx_map[r['ds']] for _, r in selected.iterrows()], dtype=np.int32)
    ts_codes = [r['ts_code'] for _, r in selected.iterrows()]
    buy_price = np.full(n_pos, np.nan, dtype=np.float64)
    last_price = np.full(n_pos, np.nan, dtype=np.float64)
    sl_price = np.full(n_pos, 0.0, dtype=np.float64)
    tp_price = np.full(n_pos, np.inf, dtype=np.float64)
    status = np.ones(n_pos, dtype=np.int8)

    daily_pnl = np.zeros(n_dates, dtype=np.float64)

    for day_i, d in enumerate(trading_dates):
        if day_i % 200 == 0:
            active = (status == 1).sum()
            print(f"    Day {day_i}/{n_dates}, active={active}", flush=True)

        open_mask = status == 1
        if not open_mask.any():
            continue

        open_idx = np.where(open_mask)[0]
        hold_days_all = day_i - entry_date_idx[open_idx]

        # hold_day=1: BUY DAY - buy at T+1 open, can't sell (T+1)
        buy_mask = hold_days_all == 1
        buy_positions = open_idx[buy_mask]
        for pos_i in buy_positions:
            tc = ts_codes[pos_i]
            ohlc = ohlc_lookup.get((tc, d))
            if ohlc is None:
                status[pos_i] = 0
                continue
            o, h, l, c = ohlc
            bp = o
            buy_price[pos_i] = bp
            last_price[pos_i] = bp
            if stop_loss < 0:
                sl_price[pos_i] = bp * (1 + stop_loss)
            if take_profit > 0:
                tp_price[pos_i] = bp * (1 + take_profit)
            daily_pnl[day_i] -= pos_size * BUY_COST
            daily_pnl[day_i] += pos_size * (c - bp) / bp
            last_price[pos_i] = c

        # hold_day=2~N: can sell
        active_sub = (hold_days_all >= 2) & (hold_days_all <= hold_days)
        if not active_sub.any():
            continue

        active_positions = open_idx[active_sub]
        active_hold = hold_days_all[active_sub]

        for j in range(len(active_positions)):
            pos_i = active_positions[j]
            hd = active_hold[j]
            tc = ts_codes[pos_i]
            ohlc = ohlc_lookup.get((tc, d))
            if ohlc is None:
                daily_pnl[day_i] -= pos_size * SELL_COST
                status[pos_i] = 0
                continue
            o, h, l, c = ohlc
            prev = last_price[pos_i]
            triggered = False

            if sl_price[pos_i] > 0 and o <= sl_price[pos_i]:
                daily_pnl[day_i] += pos_size * (o - prev) / prev
                daily_pnl[day_i] -= pos_size * SELL_COST
                status[pos_i] = 0
                last_price[pos_i] = o
                triggered = True
            elif tp_price[pos_i] < np.inf and o >= tp_price[pos_i]:
                daily_pnl[day_i] += pos_size * (o - prev) / prev
                daily_pnl[day_i] -= pos_size * SELL_COST
                status[pos_i] = 0
                last_price[pos_i] = o
                triggered = True

            if not triggered:
                sl_trig = sl_price[pos_i] > 0 and l <= sl_price[pos_i]
                tp_trig = tp_price[pos_i] < np.inf and h >= tp_price[pos_i]
                if sl_trig and tp_trig:
                    daily_pnl[day_i] += pos_size * (sl_price[pos_i] - prev) / prev
                    daily_pnl[day_i] -= pos_size * SELL_COST
                    status[pos_i] = 0
                    last_price[pos_i] = sl_price[pos_i]
                    triggered = True
                elif sl_trig:
                    daily_pnl[day_i] += pos_size * (sl_price[pos_i] - prev) / prev
                    daily_pnl[day_i] -= pos_size * SELL_COST
                    status[pos_i] = 0
                    last_price[pos_i] = sl_price[pos_i]
                    triggered = True
                elif tp_trig:
                    daily_pnl[day_i] += pos_size * (tp_price[pos_i] - prev) / prev
                    daily_pnl[day_i] -= pos_size * SELL_COST
                    status[pos_i] = 0
                    last_price[pos_i] = tp_price[pos_i]
                    triggered = True

            if not triggered:
                daily_pnl[day_i] += pos_size * (c - prev) / prev
                last_price[pos_i] = c
                if hd == hold_days:
                    daily_pnl[day_i] -= pos_size * SELL_COST
                    status[pos_i] = 0

    pnl_dict = {d: float(daily_pnl[i]) for i, d in enumerate(trading_dates)}
    return pnl_dict, trading_dates


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


def plot_equity(equity_dict, dd_dict, stats_dict, period_name, filename):
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    ax1 = axes[0]
    for label, equity in equity_dict.items():
        s = stats_dict[label]
        ax1.plot(equity.index, equity.values,
                 label=f"{label} (CAGR={s['cagr']:.1%}, Sharpe={s['sharpe']:.2f})",
                 linewidth=1.5)
    ax1.set_title(f'Equity Curves (d1_open entry) - {period_name}', fontsize=14)
    ax1.set_ylabel('Equity')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2 = axes[1]
    for label, dd in dd_dict.items():
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, label=label)
    ax2.set_title('Drawdown', fontsize=12)
    ax2.set_ylabel('Drawdown')
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}", flush=True)


def run():
    t0 = time.time()
    print("=" * 90, flush=True)
    print("Backtest with CORRECT target (return from T+1 open)", flush=True)
    print("=" * 90, flush=True)

    ohlc_df = load_ohlc_index()

    periods = [
        ('opt_2022_2024', '20220101', '20241231'),
        ('test_2025_2026', '20250101', '20261231'),
        ('full_2022_2026', '20220101', '20261231'),
    ]

    all_stats = {}
    equities_by_period = {p[0]: {} for p in periods}
    dd_by_period = {p[0]: {} for p in periods}
    stats_by_period = {p[0]: {} for p in periods}

    for strat in STRATEGIES:
        print(f"\n{'='*90}", flush=True)
        print(f"Strategy: {strat['name']}, hold_days={strat['hold_days']}", flush=True)
        print(f"{'='*90}", flush=True)

        pred = pd.read_parquet(strat['pred_file'])
        pred['ds'] = pred['trade_date'].astype(str)
        print(f"Predictions: {len(pred)} rows", flush=True)

        for ci, combo in enumerate(strat['combos']):
            label = combo['label']
            print(f"\n[{ci+1}/{len(strat['combos'])}] {label}", flush=True)
            print(f"  th={combo['threshold']}, pos={combo['max_pos']}, sl={combo['stop_loss']}, tp={combo['take_profit']}", flush=True)
            print(f"  pos_size = 1/{strat['hold_days']*combo['max_pos']}", flush=True)

            bt_start = time.time()
            daily_pnl, trading_dates = backtest(
                pred, ohlc_df,
                combo['threshold'], combo['max_pos'],
                combo['stop_loss'], combo['take_profit'],
                strat['hold_days']
            )
            print(f"  Backtest done in {time.time()-bt_start:.0f}s", flush=True)

            for period_name, start, end in periods:
                mask_dates = [d for d in trading_dates if start <= d <= end]
                if not mask_dates:
                    continue
                period_pnl = {d: daily_pnl.get(d, 0.0) for d in mask_dates}
                stats, equity, dd = calc_stats(period_pnl, mask_dates)
                key = f"{label} | {period_name}"
                all_stats[key] = {**stats, 'label': label, 'period': period_name}
                equities_by_period[period_name][label] = equity
                dd_by_period[period_name][label] = dd
                stats_by_period[period_name][label] = stats
                print(f"  {period_name}: CAGR={stats['cagr']:.2%}, Sharpe={stats['sharpe']:.2f}, "
                      f"MaxDD={stats['max_dd']:.2%}, DayWR={stats['win_rate_days']:.1%}, "
                      f"MonWR={stats['monthly_win_rate']:.1%}", flush=True)

    print(f"\n{'='*90}", flush=True)
    print("Plotting...", flush=True)
    for period_name, start, end in periods:
        if equities_by_period[period_name]:
            fname = os.path.join(RESULTS_DIR, f'open_v2_equity_{period_name}.png')
            plot_equity(equities_by_period[period_name], dd_by_period[period_name],
                        stats_by_period[period_name], period_name, fname)

    print(f"\n{'='*90}", flush=True)
    print("SUMMARY TABLE", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'Label':<40} {'Period':<18} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'DayWR':>8} {'MonWR':>8}", flush=True)
    print(f"{'-'*100}", flush=True)
    for strat in STRATEGIES:
        for combo in strat['combos']:
            label = combo['label']
            for period_name, start, end in periods:
                key = f"{label} | {period_name}"
                if key in all_stats:
                    s = all_stats[key]
                    print(f"{label:<40} {period_name:<18} {s['cagr']:>7.1%} {s['sharpe']:>7.2f} "
                          f"{s['max_dd']:>7.1%} {s['win_rate_days']:>7.1%} {s['monthly_win_rate']:>7.1%}", flush=True)
            print(f"{'-'*100}", flush=True)

    results_file = os.path.join(RESULTS_DIR, 'open_v2_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {results_file}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    run()
