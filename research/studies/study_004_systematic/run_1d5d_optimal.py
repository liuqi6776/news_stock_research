"""
1d + 5d 完整流程: 训练 -> 网格搜索 -> 绘制最优收益曲线

所有模型使用正确target: 从T+1开盘计算收益
  1d: return_1d_open = (T+2_close - T+1_open) / T+1_open  (T+1买, T+2卖)
  5d: return_5d_open = (T+5_close - T+1_open) / T+1_open  (T+1买, T+5卖)

回测中 hold_days 含义:
  hold_day=1: T+1 买入日 (不能卖)
  hold_day=2: T+2 可卖出日
  所以 1d策略 hold_days=2, 5d策略 hold_days=5

运行:
  cd study_004_systematic
  python -u run_1d5d_optimal.py
"""
import os
import sys
import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(STUDY_DIR, 'data')
FEATURES_FILE = os.path.join(DATA_DIR, 'all_features_v2.parquet')
PREDICTIONS_DIR = os.path.join(STUDY_DIR, 'predictions')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BUY_COST = 0.001
SELL_COST = 0.001
TRAIN_START = '20200101'
MIN_TRAIN_SAMPLES = 50000

MODELS = [
    {
        'name': '1d_open',
        'return_col': 'return_1d_open',
        'threshold': 0.01,
        'hold_days': 2,
        'output': os.path.join(PREDICTIONS_DIR, 'predictions_1d_open_wf_monthly.parquet'),
    },
    {
        'name': '5d_open',
        'return_col': 'return_5d_open',
        'threshold': 0.03,
        'hold_days': 5,
        'output': os.path.join(PREDICTIONS_DIR, 'predictions_5d_open_wf_monthly.parquet'),
    },
]

GRID_1D = [
    {'threshold': 0.50, 'max_pos': 3, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.55, 'max_pos': 3, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.60, 'max_pos': 3, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.50, 'max_pos': 5, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.55, 'max_pos': 5, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.60, 'max_pos': 5, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.50, 'max_pos': 3, 'stop_loss': -0.05, 'take_profit': 0.0},
    {'threshold': 0.55, 'max_pos': 3, 'stop_loss': -0.05, 'take_profit': 0.0},
    {'threshold': 0.60, 'max_pos': 3, 'stop_loss': -0.05, 'take_profit': 0.0},
]

GRID_5D = [
    {'threshold': 0.50, 'max_pos': 3, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.55, 'max_pos': 3, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.60, 'max_pos': 3, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.50, 'max_pos': 5, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.55, 'max_pos': 5, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.60, 'max_pos': 5, 'stop_loss': 0.0, 'take_profit': 0.0},
    {'threshold': 0.50, 'max_pos': 3, 'stop_loss': -0.10, 'take_profit': 0.0},
    {'threshold': 0.55, 'max_pos': 3, 'stop_loss': -0.10, 'take_profit': 0.0},
    {'threshold': 0.60, 'max_pos': 3, 'stop_loss': -0.10, 'take_profit': 0.0},
]


def get_feature_cols(df):
    exclude_cols = {'ts_code', 'trade_date', 'ds',
                    'open', 'high', 'low', 'close', 'pre_close',
                    'entry_price', 'next_open',
                    'exit_price_1d', 'return_1d', 'return_1d_open',
                    'exit_price_5d', 'return_5d', 'return_5d_open',
                    'exit_price_28d', 'return_28d', 'return_28d_open',
                    'exit_28d_close',
                    'calc_ret5d', 'return_5d_from_open', 'return_28d_from_open',
                    'entry_vs_close'}
    return [c for c in df.columns
            if c not in exclude_cols
            and not c.startswith('hist_')
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


def train_models(features_df, feature_cols):
    from xgboost import XGBClassifier

    months = sorted(features_df['ds'].str[:6].unique())
    pred_months = [m for m in months if m >= '202201']

    for model_info in MODELS:
        return_col = model_info['return_col']
        if return_col not in features_df.columns:
            print(f"ERROR: {return_col} not in features", flush=True)
            continue

        if os.path.exists(model_info['output']):
            print(f"Skipping {model_info['name']}: predictions already exist", flush=True)
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"Training: {model_info['name']} ({return_col})", flush=True)
        print(f"{'='*80}", flush=True)

        all_predictions = []
        total_start = time.time()

        for i, month in enumerate(pred_months):
            month_start = time.time()
            train_end_month = str(int(month) - 1)
            if train_end_month.endswith('00'):
                year = int(train_end_month[:4]) - 1
                train_end_month = f"{year}12"

            train_mask = (features_df['ds'] >= TRAIN_START) & (features_df['ds'].str[:6] <= train_end_month)
            pred_mask = features_df['ds'].str[:6] == month

            train_df = features_df[train_mask & features_df[return_col].notna()].copy()
            pred_df = features_df[pred_mask].copy()

            if len(train_df) < MIN_TRAIN_SAMPLES or len(pred_df) == 0:
                continue

            train_df['label'] = (train_df[return_col] > model_info['threshold']).astype(int)
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df['label']

            model = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, eval_metric='logloss'
            )
            model.fit(X_train, y_train)

            X_pred = pred_df[feature_cols].fillna(0)
            proba = model.predict_proba(X_pred)[:, 1]

            month_pred = pred_df[['trade_date', 'ts_code']].copy()
            month_pred['prob'] = proba
            month_pred['target'] = model_info['name']
            if return_col in pred_df.columns:
                month_pred['actual_return'] = pred_df[return_col].values
            if 'next_open' in pred_df.columns:
                month_pred['entry_price'] = pred_df['next_open'].values

            all_predictions.append(month_pred)

            elapsed = time.time() - month_start
            total_elapsed = time.time() - total_start
            remaining = (total_elapsed / (i + 1)) * (len(pred_months) - i - 1)
            print(f"  [{i+1}/{len(pred_months)}] {month}: train={len(train_df)}, "
                  f"pos={train_df['label'].mean():.1%}, "
                  f"prob>=0.5={(proba>=0.5).sum()}, "
                  f"elapsed={elapsed:.0f}s, remaining~{remaining/60:.0f}min", flush=True)

        if all_predictions:
            combined = pd.concat(all_predictions, ignore_index=True)
            combined.to_parquet(model_info['output'])
            print(f"Saved: {model_info['output']} ({len(combined)} rows)", flush=True)


def load_ohlc_index(pred_ts_codes=None):
    feat = pd.read_parquet(FEATURES_FILE)
    cols = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close']
    feat = feat[cols].copy()
    feat['trade_date'] = feat['trade_date'].astype(str)
    if pred_ts_codes is not None:
        feat = feat[feat['ts_code'].isin(pred_ts_codes)].copy()
    feat = feat.sort_values(['ts_code', 'trade_date'])
    feat = feat.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    ohlc_lookup = {}
    for _, row in feat.iterrows():
        ohlc_lookup[(row['ts_code'], row['trade_date'])] = (row['open'], row['high'], row['low'], row['close'])
    return ohlc_lookup


def backtest(pred_df, ohlc_lookup, threshold, max_pos, stop_loss, take_profit, hold_days):
    above = pred_df[pred_df['prob'] >= threshold].copy()
    above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
    selected = above[above['rank'] <= max_pos].copy()

    trading_dates = sorted(pred_df['ds'].unique())
    date_idx_map = {d: i for i, d in enumerate(trading_dates)}
    n_dates = len(trading_dates)

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
        open_mask = status == 1
        if not open_mask.any():
            continue

        open_idx = np.where(open_mask)[0]
        hold_days_all = day_i - entry_date_idx[open_idx]

        buy_mask = hold_days_all == 1
        for pos_i in open_idx[buy_mask]:
            ohlc = ohlc_lookup.get((ts_codes[pos_i], d))
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

        active_sub = (hold_days_all >= 2) & (hold_days_all <= hold_days)
        if not active_sub.any():
            continue

        active_positions = open_idx[active_sub]
        active_hold = hold_days_all[active_sub]

        for j in range(len(active_positions)):
            pos_i = active_positions[j]
            hd = active_hold[j]
            ohlc = ohlc_lookup.get((ts_codes[pos_i], d))
            if ohlc is None:
                daily_pnl[day_i] -= pos_size * SELL_COST
                status[pos_i] = 0
                continue
            o, h, l, c = ohlc
            prev = last_price[pos_i]
            triggered = False

            if sl_price[pos_i] > 0 and o <= sl_price[pos_i]:
                daily_pnl[day_i] += pos_size * (o - prev) / prev - pos_size * SELL_COST
                status[pos_i] = 0
                last_price[pos_i] = o
                triggered = True
            elif tp_price[pos_i] < np.inf and o >= tp_price[pos_i]:
                daily_pnl[day_i] += pos_size * (o - prev) / prev - pos_size * SELL_COST
                status[pos_i] = 0
                last_price[pos_i] = o
                triggered = True

            if not triggered:
                sl_trig = sl_price[pos_i] > 0 and l <= sl_price[pos_i]
                tp_trig = tp_price[pos_i] < np.inf and h >= tp_price[pos_i]
                if sl_trig and tp_trig:
                    daily_pnl[day_i] += pos_size * (sl_price[pos_i] - prev) / prev - pos_size * SELL_COST
                    status[pos_i] = 0
                    last_price[pos_i] = sl_price[pos_i]
                    triggered = True
                elif sl_trig:
                    daily_pnl[day_i] += pos_size * (sl_price[pos_i] - prev) / prev - pos_size * SELL_COST
                    status[pos_i] = 0
                    last_price[pos_i] = sl_price[pos_i]
                    triggered = True
                elif tp_trig:
                    daily_pnl[day_i] += pos_size * (tp_price[pos_i] - prev) / prev - pos_size * SELL_COST
                    status[pos_i] = 0
                    last_price[pos_i] = tp_price[pos_i]
                    triggered = True

            if not triggered:
                daily_pnl[day_i] += pos_size * (c - prev) / prev
                last_price[pos_i] = c
                if hd == hold_days:
                    daily_pnl[day_i] -= pos_size * SELL_COST
                    status[pos_i] = 0

    return {d: float(daily_pnl[i]) for i, d in enumerate(trading_dates)}, trading_dates


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

    print("Loading features...", flush=True)
    features_df = pd.read_parquet(FEATURES_FILE)
    features_df['ds'] = features_df['trade_date'].astype(str)
    feature_cols = get_feature_cols(features_df)
    print(f"Features: {len(features_df)} rows, {len(feature_cols)} cols", flush=True)

    for m in MODELS:
        rc = m['return_col']
        if rc in features_df.columns:
            n = features_df[rc].notna().sum()
            print(f"  {rc}: {n} valid", flush=True)

    print("\n=== STEP 1: Train Models ===", flush=True)
    train_models(features_df, feature_cols)

    print("\n=== STEP 2: Grid Search ===", flush=True)
    ohlc_lookup = load_ohlc_index()

    all_results = {}
    all_equities = {}

    for model_info in MODELS:
        pred_file = model_info['output']
        if not os.path.exists(pred_file):
            print(f"SKIP {model_info['name']}: no predictions", flush=True)
            continue

        pred = pd.read_parquet(pred_file)
        pred['ds'] = pred['trade_date'].astype(str)
        hold_days = model_info['hold_days']

        grid = GRID_1D if hold_days == 2 else GRID_5D

        print(f"\nGrid search: {model_info['name']} ({len(grid)} combos)", flush=True)

        for ci, combo in enumerate(grid):
            th = combo['threshold']
            mp = combo['max_pos']
            sl = combo['stop_loss']
            tp = combo['take_profit']
            sl_str = f"sl={sl:.0%}" if sl < 0 else "no-sl"
            tp_str = f"tp={tp:.0%}" if tp > 0 else "no-tp"
            label = f"{model_info['name']}: th={th} pos={mp} {sl_str} {tp_str}"

            print(f"  [{ci+1}/{len(grid)}] {label}", flush=True)

            daily_pnl, trading_dates = backtest(
                pred, ohlc_lookup, th, mp, sl, tp, hold_days
            )

            for period_name, start, end in [
                ('opt_2022_2024', '20220101', '20241231'),
                ('test_2025_2026', '20250101', '20261231'),
                ('full_2022_2026', '20220101', '20261231'),
            ]:
                mask_dates = [d for d in trading_dates if start <= d <= end]
                if not mask_dates:
                    continue
                period_pnl = {d: daily_pnl.get(d, 0.0) for d in mask_dates}
                stats, equity, dd = calc_stats(period_pnl, mask_dates)
                key = f"{label} | {period_name}"
                all_results[key] = {**stats, 'label': label, 'period': period_name,
                                     'threshold': th, 'max_pos': mp, 'stop_loss': sl,
                                     'take_profit': tp, 'hold_days': hold_days}
                if period_name == 'full_2022_2026':
                    all_equities[label] = equity

                if period_name in ('full_2022_2026',):
                    print(f"    {period_name}: CAGR={stats['cagr']:.1%}, Sharpe={stats['sharpe']:.2f}, "
                          f"MaxDD={stats['max_dd']:.1%}", flush=True)

    print("\n=== STEP 3: Find Best & Plot ===", flush=True)

    best_1d = None
    best_5d = None
    best_1d_sharpe = -999
    best_5d_sharpe = -999

    for key, s in all_results.items():
        if s['period'] != 'full_2022_2026':
            continue
        if s['cagr'] <= 0:
            continue
        if s['hold_days'] == 2 and s['sharpe'] > best_1d_sharpe:
            best_1d_sharpe = s['sharpe']
            best_1d = s['label']
        elif s['hold_days'] == 5 and s['sharpe'] > best_5d_sharpe:
            best_5d_sharpe = s['sharpe']
            best_5d = s['label']

    print(f"\nBest 1d: {best_1d} (Sharpe={best_1d_sharpe:.2f})", flush=True)
    print(f"Best 5d: {best_5d} (Sharpe={best_5d_sharpe:.2f})", flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 1]})

    ax1 = axes[0]
    for label in [best_1d, best_5d]:
        if label and label in all_equities:
            s = all_results[f"{label} | full_2022_2026"]
            equity = all_equities[label]
            ax1.plot(equity.index, equity.values,
                     label=f"{label}\n  CAGR={s['cagr']:.1%}, Sharpe={s['sharpe']:.2f}, MaxDD={s['max_dd']:.1%}",
                     linewidth=1.5)

    ax1.set_title('1d vs 5d Optimal Strategy Equity Curves\n(Target: return from T+1 open, Entry: d1_open)',
                  fontsize=14)
    ax1.set_ylabel('Equity')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    ax2 = axes[1]
    for label in [best_1d, best_5d]:
        if label and label in all_equities:
            equity = all_equities[label]
            running_max = equity.cummax()
            dd = (equity - running_max) / running_max
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, label=label)
    ax2.set_title('Drawdown', fontsize=12)
    ax2.set_ylabel('Drawdown')
    ax2.legend(fontsize=10, loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, '1d_vs_5d_optimal.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}", flush=True)

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"{'Label':<55} {'Period':<18} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'MonWR':>8}", flush=True)
    print('-' * 100, flush=True)
    for key in sorted(all_results.keys()):
        s = all_results[key]
        if s['period'] == 'full_2022_2026':
            print(f"{s['label']:<55} {s['period']:<18} {s['cagr']:>7.1%} {s['sharpe']:>7.2f} "
                  f"{s['max_dd']:>7.1%} {s['monthly_win_rate']:>7.1%}", flush=True)

    results_file = os.path.join(RESULTS_DIR, '1d5d_grid_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {results_file}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    run()
