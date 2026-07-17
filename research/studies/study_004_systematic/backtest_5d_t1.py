"""
5日持有期 T+1约束 网格搜索

核心逻辑:
  买入日T: 以entry_price买入, 无法卖出 (A股T+1规则)
  T+1~T+5: 逐日检查止损/止盈触发
    - 跳空低开: 若开盘价 <= 止损价, 以开盘价卖出 (实际亏损 > 止损线)
    - 跳空高开: 若开盘价 >= 止盈价, 以开盘价卖出
    - 同日SL+TP触发: 保守假设止损优先
    - 正常止损: 最低价 <= 止损价, 以止损价卖出
    - 正常止盈: 最高价 >= 止盈价, 以止盈价卖出
  T+5收盘: 若未触发, 以收盘价强制平仓

严格防止的问题:
  1. 无 np.clip() - 所有收益基于实际OHLC价格
  2. T+1约束 - 买入日不可卖出
  3. 月度WF预测 - 训练数据不含未来信息
  4. 资金重叠说明 - 5d持有期存在仓位重叠, position_size=1/max_pos

运行命令:
  cd study_004_systematic
  python -u backtest_5d_t1.py 2>&1 | tee backtest_5d_t1_log.txt
"""
import os
import sys
import pandas as pd
import numpy as np
import time
from itertools import product

sys.stdout.reconfigure(line_buffering=True)

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_FILE = os.path.join(STUDY_DIR, 'predictions', 'predictions_5d_wf_monthly.parquet')
FEATURES_FILE = os.path.join(STUDY_DIR, 'data', 'all_features_v2.parquet')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRANSACTION_COST = 0.003

THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66]
MAX_POSITIONS = [1, 2, 3, 5, 10]
STOP_LOSSES = [0.0, -0.03, -0.05, -0.07, -0.10]
TAKE_PROFIT = [0.0, 0.05, 0.08, 0.10, 0.15]


def load_next_5d_ohlc():
    print("Loading features for T+1~T+5 OHLC...", flush=True)
    feat = pd.read_parquet(FEATURES_FILE)
    cols = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close']
    feat = feat[cols].copy()
    feat['trade_date'] = feat['trade_date'].astype(str)
    feat = feat.sort_values(['ts_code', 'trade_date'])

    for d in range(1, 6):
        feat[f'd{d}_open'] = feat.groupby('ts_code')['open'].shift(-d)
        feat[f'd{d}_high'] = feat.groupby('ts_code')['high'].shift(-d)
        feat[f'd{d}_low'] = feat.groupby('ts_code')['low'].shift(-d)
        feat[f'd{d}_close'] = feat.groupby('ts_code')['close'].shift(-d)

    ohlc_cols = ['trade_date', 'ts_code']
    for d in range(1, 6):
        ohlc_cols.extend([f'd{d}_open', f'd{d}_high', f'd{d}_low', f'd{d}_close'])

    feat = feat.dropna(subset=['d1_open', 'd1_low', 'd1_high'])
    print(f"  Features with T+1~T+5 OHLC: {len(feat)} rows", flush=True)
    return feat[ohlc_cols]


def compute_5d_realized_return(entry_price, d1_o, d1_h, d1_l, d1_c,
                                d2_o, d2_h, d2_l, d2_c,
                                d3_o, d3_h, d3_l, d3_c,
                                d4_o, d4_h, d4_l, d4_c,
                                d5_o, d5_h, d5_l, d5_c,
                                stop_loss, take_profit):
    if stop_loss == 0 and take_profit == 0:
        if pd.isna(d5_c):
            if pd.isna(d4_c):
                if pd.isna(d3_c):
                    if pd.isna(d2_c):
                        return (d1_c - entry_price) / entry_price if not pd.isna(d1_c) else 0.0
                    return (d2_c - entry_price) / entry_price
                return (d3_c - entry_price) / entry_price
            return (d4_c - entry_price) / entry_price
        return (d5_c - entry_price) / entry_price

    sl_price = entry_price * (1 + stop_loss) if stop_loss < 0 else 0
    tp_price = entry_price * (1 + take_profit) if take_profit > 0 else float('inf')

    days_ohlc = [
        (d1_o, d1_h, d1_l, d1_c),
        (d2_o, d2_h, d2_l, d2_c),
        (d3_o, d3_h, d3_l, d3_c),
        (d4_o, d4_h, d4_l, d4_c),
        (d5_o, d5_h, d5_l, d5_c),
    ]

    last_close = entry_price
    for day_idx, (o, h, l, c) in enumerate(days_ohlc):
        if pd.isna(o):
            return (last_close - entry_price) / entry_price

        if sl_price > 0 and o <= sl_price:
            return (o - entry_price) / entry_price
        if tp_price < float('inf') and o >= tp_price:
            return (o - entry_price) / entry_price

        sl_triggered = sl_price > 0 and l <= sl_price
        tp_triggered = tp_price < float('inf') and h >= tp_price

        if sl_triggered and tp_triggered:
            return stop_loss
        if sl_triggered:
            return stop_loss
        if tp_triggered:
            return take_profit

        last_close = c

    return (last_close - entry_price) / entry_price


def compute_5d_realized_return_vec(entry_prices, ohlc_array, stop_loss, take_profit):
    n = len(entry_prices)
    returns = np.zeros(n)

    sl_price = entry_prices * (1 + stop_loss) if stop_loss < 0 else np.zeros(n)
    tp_price = entry_prices * (1 + take_profit) if take_profit > 0 else np.full(n, np.inf)

    has_sl = stop_loss < 0
    has_tp = take_profit > 0

    if not has_sl and not has_tp:
        for i in range(n):
            last_c = ohlc_array[i, 3]
            for d in range(1, 5):
                idx = d * 4 + 3
                if idx < ohlc_array.shape[1] and not np.isnan(ohlc_array[i, idx]):
                    last_c = ohlc_array[i, idx]
                else:
                    break
            returns[i] = (last_c - entry_prices[i]) / entry_prices[i]
        return returns

    last_close = entry_prices.copy()

    for d in range(5):
        o_idx = d * 4
        h_idx = d * 4 + 1
        l_idx = d * 4 + 2
        c_idx = d * 4 + 3

        if c_idx >= ohlc_array.shape[1]:
            break

        o = ohlc_array[:, o_idx]
        h = ohlc_array[:, h_idx]
        l = ohlc_array[:, l_idx]
        c = ohlc_array[:, c_idx]

        active = returns == 0

        na_mask = np.isnan(o)
        suspended = active & na_mask
        returns[suspended] = (last_close[suspended] - entry_prices[suspended]) / entry_prices[suspended]
        active = returns == 0

        if has_sl:
            gap_down = active & (o <= sl_price)
            returns[gap_down] = (o[gap_down] - entry_prices[gap_down]) / entry_prices[gap_down]
            active = returns == 0

        if has_tp:
            gap_up = active & (o >= tp_price)
            returns[gap_up] = (o[gap_up] - entry_prices[gap_up]) / entry_prices[gap_up]
            active = returns == 0

        sl_trig = has_sl & active & (l <= sl_price)
        tp_trig = has_tp & active & (h >= tp_price)
        both = sl_trig & tp_trig

        returns[both] = stop_loss
        active = returns == 0

        sl_only = has_sl & active & (l <= sl_price)
        returns[sl_only] = stop_loss
        active = returns == 0

        tp_only = has_tp & active & (h >= tp_price)
        returns[tp_only] = take_profit
        active = returns == 0

        valid_c = ~np.isnan(c)
        last_close[valid_c] = c[valid_c]

    still_active = returns == 0
    returns[still_active] = (last_close[still_active] - entry_prices[still_active]) / entry_prices[still_active]

    return returns


def precompute_selections(df, start, end):
    print(f"  Precomputing selections for {start}-{end}...", flush=True)
    mask = (df['ds'] >= start) & (df['ds'] <= end)
    pdf = df[mask].copy()
    trading_dates = sorted(pdf['ds'].unique())

    ohlc_col_names = []
    for d in range(1, 6):
        ohlc_col_names.extend([f'd{d}_open', f'd{d}_high', f'd{d}_low', f'd{d}_close'])

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
                    day_groups[d] = (None, None, 0)
                else:
                    entry_prices = day_trades['entry_price'].values
                    ohlc_arr = day_trades[ohlc_col_names].values.astype(np.float64)
                    day_groups[d] = (entry_prices, ohlc_arr, len(day_trades))
            selections[key] = (day_groups, trading_dates)
    return selections


def backtest_5d(day_groups, trading_dates, max_pos_val, stop_loss, take_profit):
    pos_size = 1.0 / max_pos_val

    daily_pnl = np.zeros(len(trading_dates))
    for i, d in enumerate(trading_dates):
        entry_prices, ohlc_arr, n_trades = day_groups[d]
        if n_trades == 0 or entry_prices is None:
            daily_pnl[i] = 0.0
        else:
            trade_rets = compute_5d_realized_return_vec(
                entry_prices, ohlc_arr, stop_loss, take_profit
            )
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
        'win_rate_days': float(win_rate_days),
        'monthly_win_rate': float(monthly_win_rate),
        'n_months': len(monthly_rets),
    }


def run():
    t0 = time.time()

    ohlc_data = load_next_5d_ohlc()

    print("Loading 5d predictions...", flush=True)
    pred = pd.read_parquet(PRED_FILE)
    pred['ds'] = pred['trade_date'].astype(str)
    print(f"Predictions: {len(pred)} rows", flush=True)

    df = pred.merge(ohlc_data, on=['trade_date', 'ts_code'], how='inner', suffixes=('', '_ohlc'))
    if 'entry_price_ohlc' in df.columns:
        df = df.drop(columns=['entry_price_ohlc'])
    df = df.dropna(subset=['actual_return', 'd1_open', 'd1_low', 'd1_high']).copy()
    print(f"Merged with T+1~T+5 OHLC: {len(df)} rows, date range: {df['ds'].min()} - {df['ds'].max()}", flush=True)

    opt_sel = precompute_selections(df, '20220101', '20241231')
    test_sel = precompute_selections(df, '20250101', '20261231')
    print(f"  Precompute done in {time.time()-t0:.0f}s", flush=True)

    total_combos = len(THRESHOLDS) * len(MAX_POSITIONS) * len(STOP_LOSSES) * len(TAKE_PROFIT)
    print(f"\nGrid: {total_combos} combos ({len(THRESHOLDS)} thresh x {len(MAX_POSITIONS)} pos "
          f"x {len(STOP_LOSSES)} SL x {len(TAKE_PROFIT)} TP)", flush=True)

    all_results = []
    start_time = time.time()

    for i, (threshold, max_pos_val, sl, tp) in enumerate(
            product(THRESHOLDS, MAX_POSITIONS, STOP_LOSSES, TAKE_PROFIT)):
        key = (threshold, max_pos_val)
        opt_dg, opt_td = opt_sel[key]
        test_dg, test_td = test_sel[key]

        opt = backtest_5d(opt_dg, opt_td, max_pos_val, sl, tp)
        test = backtest_5d(test_dg, test_td, max_pos_val, sl, tp)

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
    results_df.to_csv(os.path.join(RESULTS_DIR, 'wf_5d_grid_t1_full.csv'), index=False)
    print(f"Full results: {len(results_df)} rows -> wf_5d_grid_t1_full.csv", flush=True)

    opt_df = results_df.dropna(subset=['cagr_opt']).copy()

    print(f"\n{'='*90}", flush=True)
    print("TOP 30 by Opt Sharpe (2022-2024) - 5d T+1 Constraint", flush=True)
    print(f"{'='*90}", flush=True)
    top30 = opt_df.nlargest(30, 'sharpe_opt')
    cols = ['threshold', 'max_pos', 'stop_loss', 'take_profit',
            'cagr_opt', 'sharpe_opt', 'max_dd_opt', 'monthly_win_rate_opt',
            'cagr_test', 'sharpe_test', 'max_dd_test', 'monthly_win_rate_test']
    avail_cols = [c for c in cols if c in top30.columns]
    print(top30[avail_cols].to_string(index=False), flush=True)

    print(f"\n{'='*90}", flush=True)
    print("STOP LOSS impact (threshold=0.58, max_pos=3, tp=0) - 5d T+1", flush=True)
    print(f"{'='*90}", flush=True)
    sl_sub = opt_df[(opt_df['threshold'] == 0.58) & (opt_df['max_pos'] == 3) & (opt_df['take_profit'] == 0)]
    if len(sl_sub) > 0:
        sl_cols = ['stop_loss', 'sharpe_opt', 'cagr_opt', 'max_dd_opt',
                   'sharpe_test', 'cagr_test', 'max_dd_test']
        avail_sl = [c for c in sl_cols if c in sl_sub.columns]
        print(sl_sub[avail_sl].to_string(index=False), flush=True)

    print(f"\n{'='*90}", flush=True)
    print("TAKE PROFIT impact (threshold=0.58, max_pos=3, sl=0) - 5d T+1", flush=True)
    print(f"{'='*90}", flush=True)
    tp_sub = opt_df[(opt_df['threshold'] == 0.58) & (opt_df['max_pos'] == 3) & (opt_df['stop_loss'] == 0)]
    if len(tp_sub) > 0:
        tp_cols = ['take_profit', 'sharpe_opt', 'cagr_opt', 'max_dd_opt',
                   'sharpe_test', 'cagr_test', 'max_dd_test']
        avail_tp = [c for c in tp_cols if c in tp_sub.columns]
        print(tp_sub[avail_tp].to_string(index=False), flush=True)

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
    print("BEST COMBO per threshold (by opt Sharpe) - 5d T+1", flush=True)
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
    print("5d T+1 Gap Risk Analysis: gap-down bypass on T+1", flush=True)
    print(f"{'='*90}", flush=True)
    t1_sub = df[(df['ds'] >= '20220101') & (df['ds'] <= '20241231')].copy()
    for sl_val in [-0.03, -0.05, -0.07]:
        sl_price = t1_sub['entry_price'] * (1 + sl_val)
        gap_down = (t1_sub['d1_open'] <= sl_price).sum()
        total = len(t1_sub)
        sl_trigger = (t1_sub['d1_low'] <= sl_price).sum()
        print(f"  SL={sl_val:+.0%}: gap_down_at_T+1_open={gap_down} ({gap_down/total:.1%}), "
              f"any_T+1_trigger={sl_trigger} ({sl_trigger/total:.1%}), "
              f"normal_T+1_trigger={sl_trigger-gap_down} ({(sl_trigger-gap_down)/total:.1%})", flush=True)

    print(f"\n{'='*90}", flush=True)
    print("5d vs 1d comparison note:", flush=True)
    print("  - 5d持有期: 每日选股top N, 持有5日", flush=True)
    print("  - 资金重叠: 同一天最多有5批仓位 (5 x max_pos)", flush=True)
    print("  - position_size = 1/max_pos (与1d一致, 但实际占用资金更大)", flush=True)
    print("  - 日收益 = sum(当日新开仓的5d实现收益 * position_size)", flush=True)
    print("  - 注意: 此方法将5d收益归因于入场日, 与1d方法一致但存在资金重叠问题", flush=True)
    print(f"{'='*90}", flush=True)

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    run()
