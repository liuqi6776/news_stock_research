"""
网格搜索 v3: T+1约束 - 买入当天无法止损
优化期: 2022-2024, 测试期: 2025-2026

T+1止损逻辑:
- 买入日T: 以entry_price买入，无法卖出
- 卖出日T+1: 可以卖出
  - 若T+1开盘价 <= entry_price*(1+stop_loss): 跳空低开，以T+1开盘价卖出
  - 若T+1最低价 <= entry_price*(1+stop_loss) 但开盘价 > 止损价: 止损触发，以止损价卖出
  - 否则: 不触发止损，以exit_price_1d卖出(或止盈)

止盈逻辑:
- 若T+1最高价 >= entry_price*(1+take_profit): 止盈触发，以止盈价卖出
- 止盈和止损同日可能触发时，保守假设止损优先
"""
import os
import sys
import pandas as pd
import numpy as np
import time
from itertools import product

sys.stdout.reconfigure(line_buffering=True)

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_FILE = os.path.join(STUDY_DIR, 'predictions', 'predictions_1d_wf_monthly.parquet')
FEATURES_FILE = os.path.join(STUDY_DIR, 'data', 'all_features_v2.parquet')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRANSACTION_COST = 0.003

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
    feat['next_trade_date'] = feat.groupby('ts_code')['trade_date'].shift(-1)

    feat = feat.dropna(subset=['next_open', 'next_low', 'next_high'])
    print(f"  Features with T+1 OHLC: {len(feat)} rows", flush=True)
    return feat


def compute_realized_return(entry_price, next_open, next_high, next_low, next_close,
                             stop_loss, take_profit):
    if stop_loss == 0 and take_profit == 0:
        return (next_close - entry_price) / entry_price

    sl_price = entry_price * (1 + stop_loss) if stop_loss < 0 else 0
    tp_price = entry_price * (1 + take_profit) if take_profit > 0 else float('inf')

    sl_triggered = False
    tp_triggered = False

    if sl_price > 0 and next_open <= sl_price:
        return (next_open - entry_price) / entry_price

    if tp_price < float('inf') and next_open >= tp_price:
        return (next_open - entry_price) / entry_price

    if sl_price > 0 and next_low <= sl_price:
        sl_triggered = True

    if tp_price < float('inf') and next_high >= tp_price:
        tp_triggered = True

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
    pos_size = 1.0 / max_pos_val

    daily_pnl = np.zeros(len(trading_dates))
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

    return {
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'total_return': float(total_return),
        'n_trades': int(n_trades),
        'win_rate_days': float(win_rate_days),
        'monthly_win_rate': float(monthly_win_rate),
        'n_months': len(monthly_rets),
    }


def run():
    t0 = time.time()

    feat = load_next_day_ohlc()

    print("Loading predictions...", flush=True)
    pred = pd.read_parquet(PRED_FILE)
    pred['ds'] = pred['trade_date'].astype(str)
    print(f"Predictions: {len(pred)} rows", flush=True)

    df = pred.merge(feat[['trade_date', 'ts_code', 'entry_price',
                           'next_open', 'next_high', 'next_low', 'next_close']],
                     on=['trade_date', 'ts_code'], how='inner')
    df = df.dropna(subset=['actual_return', 'next_open', 'next_low', 'next_high']).copy()
    print(f"Merged with T+1 OHLC: {len(df)} rows, date range: {df['ds'].min()} - {df['ds'].max()}", flush=True)

    opt_sel = precompute_selections_t1(df, '20220101', '20241231')
    test_sel = precompute_selections_t1(df, '20250101', '20261231')
    print(f"  Precompute done in {time.time()-t0:.0f}s", flush=True)

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
            print(f"  [{i+1}/{total_combos}] elapsed={elapsed:.0f}s, remaining={remaining/60:.1f}min", flush=True)

    total_time = time.time() - start_time
    print(f"\nGrid search done in {total_time:.0f}s", flush=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'wf_monthly_grid_t1_full.csv'), index=False)
    print(f"Full results: {len(results_df)} rows -> wf_monthly_grid_t1_full.csv", flush=True)

    opt_df = results_df.dropna(subset=['cagr_opt']).copy()

    print(f"\n{'='*90}", flush=True)
    print("TOP 30 by Opt Sharpe (2022-2024) - T+1 Constraint", flush=True)
    print(f"{'='*90}", flush=True)
    top30 = opt_df.nlargest(30, 'sharpe_opt')
    cols = ['threshold', 'max_pos', 'stop_loss', 'take_profit',
            'cagr_opt', 'sharpe_opt', 'max_dd_opt', 'n_trades_opt', 'monthly_win_rate_opt',
            'cagr_test', 'sharpe_test', 'max_dd_test', 'n_trades_test', 'monthly_win_rate_test']
    avail_cols = [c for c in cols if c in top30.columns]
    print(top30[avail_cols].to_string(index=False), flush=True)

    print(f"\n{'='*90}", flush=True)
    print("STOP LOSS impact (threshold=0.58, max_pos=3, tp=0) - T+1 vs No T+1", flush=True)
    print(f"{'='*90}", flush=True)
    sl_sub = opt_df[(opt_df['threshold'] == 0.58) & (opt_df['max_pos'] == 3) & (opt_df['take_profit'] == 0)]
    if len(sl_sub) > 0:
        sl_cols = ['stop_loss', 'sharpe_opt', 'cagr_opt', 'max_dd_opt', 'n_trades_opt',
                   'sharpe_test', 'cagr_test', 'max_dd_test', 'n_trades_test']
        avail_sl = [c for c in sl_cols if c in sl_sub.columns]
        print(sl_sub[avail_sl].to_string(index=False), flush=True)

    print(f"\n{'='*90}", flush=True)
    print("COMBINED SL+TP (threshold=0.58, max_pos=3) - Top 10 by opt Sharpe", flush=True)
    print(f"{'='*90}", flush=True)
    combo_sub = opt_df[(opt_df['threshold'] == 0.58) & (opt_df['max_pos'] == 3)]
    if len(combo_sub) > 0:
        combo_cols = ['stop_loss', 'take_profit', 'sharpe_opt', 'cagr_opt', 'max_dd_opt',
                      'sharpe_test', 'cagr_test', 'max_dd_test']
        avail_combo = [c for c in combo_cols if c in combo_sub.columns]
        top_combo = combo_sub.nlargest(10, 'sharpe_opt')
        print(top_combo[avail_combo].to_string(index=False), flush=True)

    print(f"\n{'='*90}", flush=True)
    print("BEST COMBO per threshold (by opt Sharpe) - T+1 Constraint", flush=True)
    print(f"{'='*90}", flush=True)
    for thresh in THRESHOLDS:
        sub = opt_df[opt_df['threshold'] == thresh]
        if len(sub) > 0:
            best = sub.nlargest(1, 'sharpe_opt').iloc[0]
            test_sharpe = f"{best.get('sharpe_test', 0):.2f}" if pd.notna(best.get('sharpe_test')) else 'N/A'
            test_cagr = f"{best.get('cagr_test', 0):.1%}" if pd.notna(best.get('cagr_test')) else 'N/A'
            print(f"  thresh={thresh}: pos={int(best['max_pos'])}, sl={best['stop_loss']}, tp={best['take_profit']}, "
                  f"opt_sharpe={best['sharpe_opt']:.2f}, opt_cagr={best['cagr_opt']:.1%}, opt_dd={best['max_dd_opt']:.1%}, "
                  f"test_sharpe={test_sharpe}, test_cagr={test_cagr}", flush=True)

    print(f"\n{'='*90}", flush=True)
    print("Reasonable range: opt Sharpe 0.5-2.0, opt MaxDD > -50%", flush=True)
    print(f"{'='*90}", flush=True)
    reasonable = opt_df[(opt_df['sharpe_opt'] >= 0.5) & (opt_df['sharpe_opt'] <= 2.0) & (opt_df['max_dd_opt'] > -0.5)]
    reasonable = reasonable.sort_values('sharpe_test', ascending=False)
    print(f"Count: {len(reasonable)}", flush=True)
    if len(reasonable) > 0:
        r_cols = ['threshold', 'max_pos', 'stop_loss', 'take_profit',
                  'cagr_opt', 'sharpe_opt', 'max_dd_opt',
                  'cagr_test', 'sharpe_test', 'max_dd_test']
        avail_r = [c for c in r_cols if c in reasonable.columns]
        print(reasonable[avail_r].head(20).to_string(index=False), flush=True)

    print(f"\n{'='*90}", flush=True)
    print("T+1 Gap Risk Analysis: how often does gap-down bypass stop loss?", flush=True)
    print(f"{'='*90}", flush=True)
    t1_sub = df[(df['ds'] >= '20220101') & (df['ds'] <= '20241231')].copy()
    for sl_val in [-0.03, -0.05, -0.07]:
        sl_price = t1_sub['entry_price'] * (1 + sl_val)
        gap_down = (t1_sub['next_open'] <= sl_price).sum()
        total = len(t1_sub)
        sl_trigger = (t1_sub['next_low'] <= sl_price).sum()
        print(f"  SL={sl_val:+.0%}: gap_down_at_open={gap_down} ({gap_down/total:.1%}), "
              f"any_trigger={sl_trigger} ({sl_trigger/total:.1%}), "
              f"normal_trigger={sl_trigger-gap_down} ({(sl_trigger-gap_down)/total:.1%})", flush=True)

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    run()
