"""
Step 3: T+1 Constrained Grid Search

Grid search over (threshold, max_positions, stop_loss, take_profit) with A-share T+1 constraint.

T+1 Constraint Logic:
  - Purchase Day (T): Buy at entry_price, CANNOT sell
  - Sell Day (T+1):
    - If next_open <= entry_price * (1 + stop_loss):
        Gap-down below SL, sell at next_open (actual loss > SL)
    - If next_open >= entry_price * (1 + take_profit):
        Gap-up above TP, sell at next_open
    - Else:
      - If next_low <= SL price: SL triggered, sell at SL price
      - Elif next_high >= TP price: TP triggered, sell at TP price
      - Else: No trigger, sell at next_close
    - If both SL and TP trigger on same day: SL takes priority (conservative)

Optimization period: 2022-2024
Test period: 2025-2026
Runtime: ~5 minutes

Output: results/wf_monthly_grid_t1_full.csv
"""
import os
import sys
import pandas as pd
import numpy as np
import time
from itertools import product

sys.stdout.reconfigure(line_buffering=True)

RELEASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEMATIC_DIR = os.path.join(os.path.dirname(RELEASE_DIR), 'studies', 'study_004_systematic')
PRED_FILE = os.path.join(SYSTEMATIC_DIR, 'predictions', 'predictions_1d_wf_monthly.parquet')
FEATURES_FILE = os.path.join(SYSTEMATIC_DIR, 'data', 'all_features_v2.parquet')
RESULTS_DIR = os.path.join(RELEASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Transaction costs
BUY_COMMISSION = 0.00015  # 万1.5
SELL_COMMISSION = 0.00015 # 万1.5
STAMP_DUTY = 0.0005       # 千分之0.5
SLIPPAGE = 0.002          # 假设单边/双边总滑点 0.2%
TRANSACTION_COST = BUY_COMMISSION + SELL_COMMISSION + STAMP_DUTY + SLIPPAGE # 约等于 0.0028

THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66]
MAX_POSITIONS = [1, 2, 3, 5, 10]
STOP_LOSSES = [0.0, -0.03, -0.05, -0.07, -0.10]
TAKE_PROFIT = [0.0, 0.05, 0.08, 0.10, 0.15]


def load_next_day_ohlc():
    print("Loading features for T+1 OHLC...", flush=True)
    feat = pd.read_parquet(FEATURES_FILE)
    feat = feat[['trade_date', 'ts_code', 'open', 'high', 'low', 'close',
                  'entry_price', 'exit_price_1d']].copy()
    feat['trade_date'] = feat['trade_date'].astype(str)

    feat = feat.sort_values(['ts_code', 'trade_date'])
    feat['next_open'] = feat.groupby('ts_code')['open'].shift(-1)
    feat['next_high'] = feat.groupby('ts_code')['high'].shift(-1)
    feat['next_low'] = feat.groupby('ts_code')['low'].shift(-1)
    feat['next_close'] = feat.groupby('ts_code')['close'].shift(-1)

    feat = feat.dropna(subset=['next_open', 'next_low', 'next_high'])
    print(f"  Features with T+1 OHLC: {len(feat)} rows", flush=True)
    return feat


def compute_realized_return(entry_price, next_open, next_high, next_low, next_close,
                             stop_loss, take_profit):
    if stop_loss == 0 and take_profit == 0:
        return (next_close - entry_price) / entry_price

    sl_price = entry_price * (1 + stop_loss) if stop_loss < 0 else 0
    tp_price = entry_price * (1 + take_profit) if take_profit > 0 else float('inf')

    if sl_price > 0 and next_open <= sl_price:
        return (next_open - entry_price) / entry_price

    if tp_price < float('inf') and next_open >= tp_price:
        return (next_open - entry_price) / entry_price

    sl_triggered = sl_price > 0 and next_low <= sl_price
    tp_triggered = tp_price < float('inf') and next_high >= tp_price

    if sl_triggered and tp_triggered:
        return stop_loss

    if sl_triggered:
        return stop_loss

    if tp_triggered:
        return take_profit

    return (next_close - entry_price) / entry_price


def precompute_selections_t1(df, start, end):
    print(f"  Precomputing T+1 selections for {start}-{end}...", flush=True)
    mask = (df['ds'] >= start) & (df['ds'] <= end)
    pdf = df[mask].copy()
    trading_dates = sorted(pdf['ds'].unique())

    selections = {}
    for threshold in THRESHOLDS:
        above = pdf[pdf['prob'] >= threshold].copy()
        above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
        for max_pos in MAX_POSITIONS:
            selected = above[above['rank'] <= max_pos].copy()
            key = (threshold, max_pos)
            day_groups = {}
            for d in trading_dates:
                day_trades = selected[selected['ds'] == d]
                if len(day_trades) == 0:
                    day_groups[d] = None
                else:
                    day_groups[d] = day_trades[['entry_price', 'next_open', 'next_high',
                                                 'next_low', 'next_close']].values
            selections[key] = (day_groups, trading_dates, len(selected))
    return selections


def backtest_t1(day_groups, trading_dates, n_trades, max_pos_val, stop_loss, take_profit):
    daily_pnl = np.zeros(len(trading_dates))
    all_trade_rets = []
    for i, d in enumerate(trading_dates):
        trades = day_groups[d]
        if trades is None or len(trades) == 0:
            daily_pnl[i] = 0.0
        else:
            trade_rets = np.array([
                compute_realized_return(row[0], row[1], row[2], row[3], row[4],
                                        stop_loss, take_profit)
                for row in trades
            ])
            trade_rets = trade_rets - TRANSACTION_COST
            all_trade_rets.extend(trade_rets.tolist())
            
            pos_size = 1.0 / max_pos_val
            daily_pnl[i] = pos_size * trade_rets.sum()

    n_days = len(daily_pnl)
    n_years = n_days / 252
    if n_years == 0:
        return None

    equity = np.cumprod(1 + daily_pnl)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max

    total_return = equity[-1] - 1
    cagr = (equity[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    max_dd = drawdown.min()
    std = daily_pnl.std()
    sharpe = (daily_pnl.mean() / std * np.sqrt(252)) if std > 1e-10 else 0
    win_rate_days = (daily_pnl > 0).mean()

    monthly_idx = pd.to_datetime(trading_dates, format='%Y%m%d')
    monthly_s = pd.Series(daily_pnl, index=monthly_idx)
    monthly_rets = []
    for period, group in monthly_s.groupby(monthly_s.index.to_period('M')):
        monthly_rets.append((1 + group).prod() - 1)
    monthly_win_rate = np.mean([1 if r > 0 else 0 for r in monthly_rets]) if monthly_rets else 0

    avg_trade_return = np.mean(all_trade_rets) if all_trade_rets else 0.0
    trade_win_rate = np.mean([1 if r > 0 else 0 for r in all_trade_rets]) if all_trade_rets else 0.0

    return {
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'total_return': float(total_return),
        'n_trades': int(n_trades),
        'win_rate_days': float(win_rate_days),
        'monthly_win_rate': float(monthly_win_rate),
        'n_months': len(monthly_rets),
        'avg_trade_return': float(avg_trade_return),
        'trade_win_rate': float(trade_win_rate),
    }


def run():
    t0 = time.time()

    if not os.path.exists(PRED_FILE):
        print(f"ERROR: Prediction file not found: {PRED_FILE}")
        print("Please run step2_walkforward_predict.py first.")
        return

    if not os.path.exists(FEATURES_FILE):
        print(f"ERROR: Feature file not found: {FEATURES_FILE}")
        print("Please run step1_build_features.py first.")
        return

    feat = load_next_day_ohlc()

    print("Loading predictions...", flush=True)
    pred = pd.read_parquet(PRED_FILE)
    pred['ds'] = pred['trade_date'].astype(str)
    print(f"Predictions: {len(pred)} rows", flush=True)

    df = pred.merge(feat[['trade_date', 'ts_code', 'entry_price',
                           'next_open', 'next_high', 'next_low', 'next_close']],
                     on=['trade_date', 'ts_code'], how='inner')
    df = df.dropna(subset=['actual_return', 'next_open', 'next_low', 'next_high', 'next_close']).copy()
    
    # 增加选股条件：买入当天的开盘价相比昨晚（买入前一天）的收盘价，高开幅度在 2% 到 6% 之间
    gap_up = (df['entry_price'] - df['next_close']) / df['next_close']
    df = df[(gap_up > 0.02) & (gap_up < 0.06)].copy()

    print(f"Merged with T+1 OHLC and filtered gap-up: {len(df)} rows, date range: {df['ds'].min()} - {df['ds'].max()}", flush=True)

    opt_sel = precompute_selections_t1(df, '20220101', '20241231')
    test_sel = precompute_selections_t1(df, '20250101', '20261231')
    print(f"  Precompute done in {time.time() - t0:.0f}s", flush=True)

    total_combos = len(THRESHOLDS) * len(MAX_POSITIONS) * len(STOP_LOSSES) * len(TAKE_PROFIT)
    print(f"\nGrid: {total_combos} combos (9 thresh x 5 pos x 5 SL x 5 TP)", flush=True)

    all_results = []
    start_time = time.time()

    for i, (threshold, max_pos_val, sl, tp) in enumerate(product(THRESHOLDS, MAX_POSITIONS, STOP_LOSSES, TAKE_PROFIT)):
        key = (threshold, max_pos_val)
        opt_dg, opt_td, opt_nt = opt_sel[key]
        test_dg, test_td, test_nt = test_sel[key]

        opt = backtest_t1(opt_dg, opt_td, opt_nt, max_pos_val, sl, tp)
        test = backtest_t1(test_dg, test_td, test_nt, max_pos_val, sl, tp)

        row = {
            'threshold': threshold,
            'max_pos': max_pos_val,
            'stop_loss': sl,
            'take_profit': tp,
        }
        if opt is not None:
            row.update({f'{k}_opt': v for k, v in opt.items()})
        if test is not None:
            row.update({f'{k}_test': v for k, v in test.items()})
        all_results.append(row)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1)
            remaining = avg * (total_combos - i - 1)
            print(f"  [{i + 1}/{total_combos}] elapsed={elapsed:.0f}s, remaining={remaining / 60:.1f}min", flush=True)

    total_time = time.time() - start_time
    print(f"\nGrid search done in {total_time:.0f}s", flush=True)

    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, 'wf_monthly_grid_t1_full.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Full results: {len(results_df)} rows -> {output_path}", flush=True)

    opt_df = results_df.dropna(subset=['cagr_opt']).copy()

    print(f"\n{'=' * 90}", flush=True)
    print("TOP 20 by Opt Sharpe (2022-2024) - T+1 Constraint", flush=True)
    print(f"{'=' * 90}", flush=True)
    top20 = opt_df.nlargest(20, 'sharpe_opt')
    cols = ['threshold', 'max_pos', 'stop_loss', 'take_profit',
            'cagr_opt', 'sharpe_opt', 'max_dd_opt', 'n_trades_opt',
            'avg_trade_return_opt', 'trade_win_rate_opt',
            'cagr_test', 'sharpe_test', 'max_dd_test', 'n_trades_test',
            'avg_trade_return_test', 'trade_win_rate_test']
    print(top20[[c for c in cols if c in top20.columns]].to_string(index=False), flush=True)

    print(f"\n{'=' * 90}", flush=True)
    print("TOP 20 by Test Sharpe (2025-2026) - T+1 Constraint", flush=True)
    print(f"{'=' * 90}", flush=True)
    top20_test = opt_df.nlargest(20, 'sharpe_test')
    print(top20_test[[c for c in cols if c in top20_test.columns]].to_string(index=False), flush=True)

    print(f"\n{'=' * 90}", flush=True)
    print("STOP LOSS impact (threshold=0.58, max_pos=3, tp=0) - T+1 Constraint", flush=True)
    print(f"{'=' * 90}", flush=True)
    sl_sub = opt_df[(opt_df['threshold'] == 0.58) & (opt_df['max_pos'] == 3) & (opt_df['take_profit'] == 0)]
    if len(sl_sub) > 0:
        sl_cols = ['stop_loss', 'sharpe_opt', 'cagr_opt', 'max_dd_opt',
                   'sharpe_test', 'cagr_test', 'max_dd_test']
        print(sl_sub[[c for c in sl_cols if c in sl_sub.columns]].to_string(index=False), flush=True)

    print(f"\n{'=' * 90}", flush=True)
    print("TAKE PROFIT impact (threshold=0.58, max_pos=3, sl=0) - T+1 Constraint", flush=True)
    print(f"{'=' * 90}", flush=True)
    tp_sub = opt_df[(opt_df['threshold'] == 0.58) & (opt_df['max_pos'] == 3) & (opt_df['stop_loss'] == 0)]
    if len(tp_sub) > 0:
        tp_cols = ['take_profit', 'sharpe_opt', 'cagr_opt', 'max_dd_opt',
                   'sharpe_test', 'cagr_test', 'max_dd_test']
        print(tp_sub[[c for c in tp_cols if c in tp_sub.columns]].to_string(index=False), flush=True)

    print(f"\n{'=' * 90}", flush=True)
    print("BEST COMBO per threshold (by test Sharpe)", flush=True)
    print(f"{'=' * 90}", flush=True)
    for thresh in THRESHOLDS:
        sub = opt_df[opt_df['threshold'] == thresh]
        if len(sub) > 0:
            best = sub.nlargest(1, 'sharpe_test').iloc[0]
            print(f"  thresh={thresh}: pos={int(best['max_pos'])}, sl={best['stop_loss']:+.0%}, "
                  f"tp={best['take_profit']:+.0%}, "
                  f"opt_sharpe={best['sharpe_opt']:.2f}, opt_cagr={best['cagr_opt']:+.1%}, opt_trade_ret={best.get('avg_trade_return_opt',0):.2%}, "
                  f"test_sharpe={best['sharpe_test']:.2f}, test_cagr={best['cagr_test']:+.1%}, test_trade_ret={best.get('avg_trade_return_test',0):.2%}", flush=True)

    print(f"\nTotal time: {time.time() - t0:.0f}s", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    run()
