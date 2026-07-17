"""
30日持有期 T+1约束 回测 (d1_open入场, 逐日盯市)

逻辑:
  - 使用5d预测的prob选股, 但持有30个交易日
  - 入场: 次日开盘(d1_open), 非T日收盘
  - T+1: 买入日(hold_day=1)不可卖出
  - 逐日盯市, 每日计算持仓市值变化
  - pos_size = 1/(30*max_pos), 最多30批仓位重叠
  - SL/TP基于实际买入价(d1_open)
  - 买入/卖出均扣手续费

运行:
  cd study_004_systematic
  python -u backtest_30d_t1.py
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
PRED_FILE = os.path.join(STUDY_DIR, 'predictions', 'predictions_5d_wf_monthly.parquet')
FEATURES_FILE = os.path.join(STUDY_DIR, 'data', 'all_features_v2.parquet')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

BUY_COST = 0.001
SELL_COST = 0.001
HOLD_DAYS = 30

COMBOS = [
    {'threshold': 0.50, 'max_pos': 2, 'stop_loss': 0.0,   'take_profit': 0.10, 'label': '30d: th=0.50 pos=2 tp=10%'},
    {'threshold': 0.58, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.10, 'label': '30d: th=0.58 pos=3 tp=10%'},
    {'threshold': 0.58, 'max_pos': 3, 'stop_loss': -0.10, 'take_profit': 0.10, 'label': '30d: th=0.58 pos=3 sl=-10% tp=10%'},
    {'threshold': 0.62, 'max_pos': 5, 'stop_loss': 0.0,   'take_profit': 0.10, 'label': '30d: th=0.62 pos=5 tp=10%'},
    {'threshold': 0.64, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.10, 'label': '30d: th=0.64 pos=3 tp=10%'},
    {'threshold': 0.58, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.0,  'label': '30d: th=0.58 pos=3 no-sltp'},
    {'threshold': 0.64, 'max_pos': 3, 'stop_loss': 0.0,   'take_profit': 0.0,  'label': '30d: th=0.64 pos=3 no-sltp'},
]


def load_ohlc_index(pred_ts_codes=None):
    print("Loading OHLC data for daily lookup...", flush=True)
    feat = pd.read_parquet(FEATURES_FILE)
    cols = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close']
    feat = feat[cols].copy()
    feat['trade_date'] = feat['trade_date'].astype(str)

    if pred_ts_codes is not None:
        before = len(feat)
        feat = feat[feat['ts_code'].isin(pred_ts_codes)].copy()
        print(f"  Filtered to prediction stocks: {before} -> {len(feat)} rows", flush=True)

    feat = feat.sort_values(['ts_code', 'trade_date'])
    feat = feat.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    print(f"  OHLC data: {len(feat)} rows", flush=True)
    return feat


def backtest_30d(pred_df, ohlc_df, threshold, max_pos, stop_loss, take_profit):
    above = pred_df[pred_df['prob'] >= threshold].copy()
    above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
    selected = above[above['rank'] <= max_pos].copy()
    print(f"    Selected positions: {len(selected)}", flush=True)

    trading_dates = sorted(pred_df['ds'].unique())
    date_idx_map = {d: i for i, d in enumerate(trading_dates)}
    n_dates = len(trading_dates)

    ohlc_df_idx = ohlc_df.set_index(['ts_code', 'trade_date'])
    ohlc_dict = {}
    for (tc, td), row in ohlc_df_idx.iterrows():
        ohlc_dict[(tc, td)] = (row['open'], row['high'], row['low'], row['close'])

    pos_size = 1.0 / (HOLD_DAYS * max_pos)

    positions = []
    for _, row in selected.iterrows():
        positions.append({
            'ts_code': row['ts_code'],
            'entry_date': row['ds'],
            'entry_date_idx': date_idx_map[row['ds']],
            'buy_price': np.nan,
            'last_price': np.nan,
            'sl_price': 0.0,
            'tp_price': np.inf,
            'status': 1,
        })

    n_pos = len(positions)
    if n_pos == 0:
        pnl_dict = {d: 0.0 for d in trading_dates}
        return pnl_dict, trading_dates

    entry_date_idx = np.array([p['entry_date_idx'] for p in positions], dtype=np.int32)
    ts_codes = [p['ts_code'] for p in positions]
    buy_price = np.full(n_pos, np.nan, dtype=np.float64)
    last_price = np.full(n_pos, np.nan, dtype=np.float64)
    sl_price = np.full(n_pos, 0.0, dtype=np.float64)
    tp_price = np.full(n_pos, np.inf, dtype=np.float64)
    status = np.ones(n_pos, dtype=np.int8)

    daily_pnl = np.zeros(n_dates, dtype=np.float64)

    print(f"    Starting daily loop ({n_dates} days, {n_pos} positions)...", flush=True)
    for day_i, d in enumerate(trading_dates):
        if day_i % 100 == 0:
            active = (status == 1).sum()
            print(f"    Day {day_i}/{n_dates}, active={active}", flush=True)

        open_mask = status == 1
        if not open_mask.any():
            continue

        open_idx = np.where(open_mask)[0]
        hold_days_all = day_i - entry_date_idx[open_idx]

        # hold_day=1: BUY DAY - buy at d1_open, can't sell (T+1)
        buy_mask = hold_days_all == 1
        buy_positions = open_idx[buy_mask]
        for pos_i in buy_positions:
            tc = ts_codes[pos_i]
            key = (tc, d)
            ohlc = ohlc_dict.get(key)
            if ohlc is None:
                status[pos_i] = 0
                continue

            o, h, l, c = ohlc
            bp = o  # buy at open
            buy_price[pos_i] = bp
            last_price[pos_i] = bp

            if stop_loss < 0:
                sl_price[pos_i] = bp * (1 + stop_loss)
            if take_profit > 0:
                tp_price[pos_i] = bp * (1 + take_profit)

            daily_pnl[day_i] -= pos_size * BUY_COST
            daily_pnl[day_i] += pos_size * (c - bp) / bp
            last_price[pos_i] = c

        # hold_day=2~30: can sell, check SL/TP
        active_sub = (hold_days_all >= 2) & (hold_days_all <= HOLD_DAYS)
        if not active_sub.any():
            continue

        active_positions = open_idx[active_sub]
        active_hold = hold_days_all[active_sub]

        for j in range(len(active_positions)):
            pos_i = active_positions[j]
            hd = active_hold[j]
            tc = ts_codes[pos_i]
            key = (tc, d)
            ohlc = ohlc_dict.get(key)

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

                if hd == HOLD_DAYS:
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

    stats = {
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'total_return': float(total_return),
        'win_rate_days': float(win_rate),
        'monthly_win_rate': float(monthly_win),
        'n_days': int(n_days),
        'n_months': len(monthly_rets),
    }

    return stats, equity, drawdown


def plot_equity_curves(equity_dict, drawdown_dict, stats_dict, period_name, filename):
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})

    ax1 = axes[0]
    for label, equity in equity_dict.items():
        s = stats_dict[label]
        ax1.plot(equity.index, equity.values,
                 label=f"{label} (CAGR={s['cagr']:.1%}, Sharpe={s['sharpe']:.2f})",
                 linewidth=1.5)

    ax1.set_title(f'30d T+1 Equity Curves (d1_open entry) - {period_name}\n'
                  f'(Mark-to-Market, pos_size=1/(30*max_pos))',
                  fontsize=14)
    ax1.set_ylabel('Equity')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    ax2 = axes[1]
    for label, dd in drawdown_dict.items():
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, label=label)
    ax2.set_title('Drawdown', fontsize=12)
    ax2.set_ylabel('Drawdown')
    ax2.legend(fontsize=9, loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}", flush=True)


def run():
    t0 = time.time()

    print("=" * 80, flush=True)
    print("30d T+1 Backtest: Entry at d1_open (next-day open)", flush=True)
    print("  - Buy at d1_open (T+1 day open)", flush=True)
    print("  - T+1: Can't sell on buy day (hold_day=1)", flush=True)
    print("  - Hold up to 30 trading days", flush=True)
    print("  - pos_size = 1/(30*max_pos)", flush=True)
    print("  - Both buy and sell transaction costs", flush=True)
    print("=" * 80, flush=True)

    print("\nLoading predictions...", flush=True)
    pred = pd.read_parquet(PRED_FILE)
    pred['ds'] = pred['trade_date'].astype(str)
    print(f"Predictions: {len(pred)} rows", flush=True)

    pred_ts_codes = set(pred['ts_code'].unique())
    print(f"Unique stocks: {len(pred_ts_codes)}", flush=True)

    ohlc_df = load_ohlc_index(pred_ts_codes)

    periods = [
        ('opt_2022_2024', '20220101', '20241231'),
        ('test_2025_2026', '20250101', '20261231'),
        ('full_2022_2026', '20220101', '20261231'),
    ]

    all_stats = {}
    equities_by_period = {p[0]: {} for p in periods}
    drawdowns_by_period = {p[0]: {} for p in periods}
    stats_by_period = {p[0]: {} for p in periods}

    for ci, combo in enumerate(COMBOS):
        label = combo['label']
        threshold = combo['threshold']
        max_pos = combo['max_pos']
        stop_loss = combo['stop_loss']
        take_profit = combo['take_profit']

        print(f"\n{'='*80}", flush=True)
        print(f"[{ci+1}/{len(COMBOS)}] {label}", flush=True)
        print(f"  threshold={threshold}, max_pos={max_pos}, sl={stop_loss}, tp={take_profit}", flush=True)
        print(f"  pos_size = 1/{HOLD_DAYS*max_pos} = {1.0/(HOLD_DAYS*max_pos):.4f}", flush=True)

        bt_start = time.time()
        daily_pnl, trading_dates = backtest_30d(
            pred, ohlc_df, threshold, max_pos, stop_loss, take_profit
        )
        print(f"  Backtest done in {time.time()-bt_start:.0f}s", flush=True)

        for period_name, start, end in periods:
            mask_dates = [d for d in trading_dates if start <= d <= end]
            if not mask_dates:
                continue

            period_pnl = {d: daily_pnl.get(d, 0.0) for d in mask_dates}
            stats, equity, drawdown = calc_stats(period_pnl, mask_dates)

            key = f"{label} | {period_name}"
            all_stats[key] = {**stats, 'label': label, 'period': period_name}

            equities_by_period[period_name][label] = equity
            drawdowns_by_period[period_name][label] = drawdown
            stats_by_period[period_name][label] = stats

            print(f"  {period_name}: CAGR={stats['cagr']:.2%}, Sharpe={stats['sharpe']:.2f}, "
                  f"MaxDD={stats['max_dd']:.2%}, DayWR={stats['win_rate_days']:.1%}, "
                  f"MonWR={stats['monthly_win_rate']:.1%}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("Plotting equity curves...", flush=True)

    for period_name, start, end in periods:
        if equities_by_period[period_name]:
            fname = os.path.join(RESULTS_DIR, f'30d_t1_equity_{period_name}.png')
            plot_equity_curves(
                equities_by_period[period_name],
                drawdowns_by_period[period_name],
                stats_by_period[period_name],
                period_name, fname
            )

    print(f"\n{'='*80}", flush=True)
    print("Summary Table (30d, d1_open entry, buy+sell costs)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Label':<40} {'Period':<18} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'DayWR':>8} {'MonWR':>8}", flush=True)
    print(f"{'-'*100}", flush=True)

    for combo in COMBOS:
        label = combo['label']
        for period_name, start, end in periods:
            key = f"{label} | {period_name}"
            if key in all_stats:
                s = all_stats[key]
                print(f"{label:<40} {period_name:<18} {s['cagr']:>7.1%} {s['sharpe']:>7.2f} "
                      f"{s['max_dd']:>7.1%} {s['win_rate_days']:>7.1%} {s['monthly_win_rate']:>7.1%}", flush=True)
        print(f"{'-'*100}", flush=True)

    results_file = os.path.join(RESULTS_DIR, '30d_t1_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {results_file}", flush=True)

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    run()
