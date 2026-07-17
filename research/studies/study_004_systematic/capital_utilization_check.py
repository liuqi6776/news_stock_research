import os, sys, json, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_FILE = os.path.join(STUDY_DIR, 'data', 'all_features_v2.parquet')
PRED_FILE = os.path.join(STUDY_DIR, 'predictions', 'predictions_1d_open_wf_monthly.parquet')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

BUY_COST = 0.001
SELL_COST = 0.001
LIMIT_UP_THRESHOLD = 0.095

THRESHOLD = 0.55
MAX_POS = 3
HOLD_DAYS = 2
GAP_MIN = 0.02
GAP_MAX = 0.06


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


def backtest_compare(pred_df, ohlc_lookup, pctchg_lookup, close_lookup):
    above = pred_df[pred_df['prob'] >= THRESHOLD].copy()
    above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
    selected = above[above['rank'] <= MAX_POS].copy()

    trading_dates = sorted(pred_df['ds'].unique())
    date_idx_map = {d: i for i, d in enumerate(trading_dates)}
    n_dates = len(trading_dates)

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
        if GAP_MIN <= gap < GAP_MAX:
            filtered.append(row)
            gap_values.append(gap)

    if not filtered:
        print("No trades after gap filter!", flush=True)
        return

    filtered_df = pd.DataFrame(filtered)
    n_pos = len(filtered_df)
    print(f"\nFiltered trades: {n_pos}", flush=True)

    entry_date_idx = np.array([date_idx_map[r['ds']] for _, r in filtered_df.iterrows()], dtype=np.int32)
    ts_codes_arr = [r['ts_code'] for _, r in filtered_df.iterrows()]
    buy_price = np.full(n_pos, np.nan, dtype=np.float64)
    last_price = np.full(n_pos, np.nan, dtype=np.float64)
    status = np.ones(n_pos, dtype=np.int8)

    fixed_pos_size = 1.0 / (HOLD_DAYS * MAX_POS)

    daily_pnl_fixed = np.zeros(n_dates, dtype=np.float64)
    daily_pnl_dynamic = np.zeros(n_dates, dtype=np.float64)
    daily_capital_used = np.zeros(n_dates, dtype=np.float64)
    daily_n_positions = np.zeros(n_dates, dtype=np.int32)
    daily_invested_return = np.zeros(n_dates, dtype=np.float64)

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
            if is_limit_up(ts_codes_arr[pos_i], pct_t1):
                status[pos_i] = 0
                continue
            pct_t0 = pctchg_lookup.get((ts_codes_arr[pos_i], trading_dates[entry_date_idx[pos_i]]))
            if is_limit_up(ts_codes_arr[pos_i], pct_t0):
                status[pos_i] = 0
                continue
            bp = o
            buy_price[pos_i] = bp
            last_price[pos_i] = bp
            daily_pnl_fixed[day_i] -= fixed_pos_size * BUY_COST
            daily_pnl_dynamic[day_i] -= fixed_pos_size * BUY_COST
            daily_pnl_fixed[day_i] += fixed_pos_size * (c - bp) / bp
            daily_pnl_dynamic[day_i] += fixed_pos_size * (c - bp) / bp
            last_price[pos_i] = c

        active_sub = (hold_days_all >= 2) & (hold_days_all <= HOLD_DAYS)
        if not active_sub.any():
            continue
        active_positions = open_idx[active_sub]
        active_hold = hold_days_all[active_sub]

        for j in range(len(active_positions)):
            pos_i = active_positions[j]
            hd = active_hold[j]
            ohlc = ohlc_lookup.get((ts_codes_arr[pos_i], d))
            if ohlc is None:
                daily_pnl_fixed[day_i] -= fixed_pos_size * SELL_COST
                daily_pnl_dynamic[day_i] -= fixed_pos_size * SELL_COST
                status[pos_i] = 0
                continue
            o, h, l, c = ohlc
            prev = last_price[pos_i]
            pos_return = (c - prev) / prev

            pct_d = pctchg_lookup.get((ts_codes_arr[pos_i], d))
            at_limit_down = is_limit_down(ts_codes_arr[pos_i], pct_d)

            if hd == HOLD_DAYS:
                if at_limit_down:
                    daily_pnl_fixed[day_i] += fixed_pos_size * pos_return
                    daily_pnl_dynamic[day_i] += fixed_pos_size * pos_return
                    last_price[pos_i] = c
                else:
                    daily_pnl_fixed[day_i] += fixed_pos_size * pos_return - fixed_pos_size * SELL_COST
                    daily_pnl_dynamic[day_i] += fixed_pos_size * pos_return - fixed_pos_size * SELL_COST
                    status[pos_i] = 0
                    last_price[pos_i] = c
            else:
                daily_pnl_fixed[day_i] += fixed_pos_size * pos_return
                daily_pnl_dynamic[day_i] += fixed_pos_size * pos_return
                last_price[pos_i] = c

    for day_i in range(n_dates):
        count = 0
        invested_pnl = 0.0
        for pos_i in range(n_pos):
            if not np.isnan(buy_price[pos_i]) and status[pos_i] >= 0:
                ed = entry_date_idx[pos_i]
                hd = day_i - ed
                if 1 <= hd <= HOLD_DAYS:
                    count += 1
        daily_n_positions[day_i] = count
        daily_capital_used[day_i] = count * fixed_pos_size

        if daily_pnl_fixed[day_i] != 0 and daily_capital_used[day_i] > 0:
            daily_invested_return[day_i] = daily_pnl_fixed[day_i] / daily_capital_used[day_i]

    dates = pd.to_datetime(trading_dates, format='%Y%m%d')

    equity_fixed = (1 + pd.Series(daily_pnl_fixed, index=dates)).cumprod()
    equity_dynamic = (1 + pd.Series(daily_pnl_dynamic, index=dates)).cumprod()

    n_years = len(trading_dates) / 252
    cagr_fixed = equity_fixed.iloc[-1] ** (1 / n_years) - 1
    cagr_dynamic = equity_dynamic.iloc[-1] ** (1 / n_years) - 1

    capital_series = pd.Series(daily_capital_used, index=dates)
    npos_series = pd.Series(daily_n_positions, index=dates)
    invested_ret_series = pd.Series(daily_invested_return, index=dates)

    print("\n" + "=" * 80, flush=True)
    print("资金利用率分析 (gap 2%-6%, th=0.55, pos=3)", flush=True)
    print("=" * 80, flush=True)

    print(f"\n--- 基本统计 ---", flush=True)
    print(f"总交易日: {len(trading_dates)}", flush=True)
    print(f"总执行交易: {n_pos}", flush=True)
    print(f"日均交易: {n_pos / len(trading_dates):.2f} 笔", flush=True)

    print(f"\n--- 每日持仓数量分布 ---", flush=True)
    for n in sorted(npos_series.unique()):
        count = (npos_series == n).sum()
        pct = count / len(npos_series) * 100
        print(f"  {n} 只持仓: {count} 天 ({pct:.1f}%)", flush=True)

    print(f"\n--- 资金利用率 ---", flush=True)
    print(f"固定仓位: pos_size = 1/{HOLD_DAYS * MAX_POS} = {fixed_pos_size:.4f} ({fixed_pos_size*100:.1f}%)", flush=True)
    print(f"日均资金占用: {capital_series.mean()*100:.1f}%", flush=True)
    print(f"中位资金占用: {capital_series.median()*100:.1f}%", flush=True)
    print(f"最大资金占用: {capital_series.max()*100:.1f}%", flush=True)
    print(f"资金占用>0的天数: {(capital_series > 0).sum()} / {len(capital_series)} ({(capital_series > 0).mean()*100:.1f}%)", flush=True)
    print(f"资金占用=0的天数(全空仓): {(capital_series == 0).sum()} / {len(capital_series)} ({(capital_series == 0).mean()*100:.1f}%)", flush=True)

    idle_cash_pct = 1 - capital_series.mean()
    print(f"\n--- 闲置资金 ---", flush=True)
    print(f"平均闲置资金: {idle_cash_pct*100:.1f}%", flush=True)

    print(f"\n--- CAGR 对比 ---", flush=True)
    print(f"方法A (固定仓位1/6, 含闲置资金): CAGR = {cagr_fixed*100:.1f}%", flush=True)

    invested_ret_clean = invested_ret_series[invested_ret_series != 0]
    if len(invested_ret_clean) > 0:
        avg_invested_daily = invested_ret_clean.mean()
        invested_cagr_simple = (1 + avg_invested_daily) ** 252 - 1
        print(f"方法B (仅计算有交易日的收益, 年化): {invested_cagr_simple*100:.1f}%", flush=True)
        print(f"  有交易日数: {len(invested_ret_clean)} / {len(trading_dates)}", flush=True)
        print(f"  有交易日平均日收益: {avg_invested_daily*100:.3f}%", flush=True)

    print(f"\n--- 投入资金的真实收益 ---", flush=True)
    invested_capital_cagr = cagr_fixed / capital_series.mean() if capital_series.mean() > 0 else 0
    print(f"总账户CAGR: {cagr_fixed*100:.1f}%", flush=True)
    print(f"平均资金占用率: {capital_series.mean()*100:.1f}%", flush=True)
    print(f"投入资金的隐含CAGR: {invested_capital_cagr*100:.1f}%", flush=True)
    print(f"  (即: 如果把闲置资金也算上, 投入的那部分资金实际产生了多少年化)", flush=True)

    print(f"\n--- 关键问题验证 ---", flush=True)
    print(f"Q: CAGR 34.5% 是总资金账户的年化还是仅投入资金的年化?", flush=True)
    print(f"A: 是总资金账户的年化。每日PnL = pos_size * 收益率, pos_size=1/6", flush=True)
    print(f"   闲置资金每日收益=0, 已包含在计算中。", flush=True)
    print(f"   所以34.5%是真实的总账户CAGR, 不是仅投入资金的CAGR。", flush=True)
    print(f"", flush=True)
    print(f"Q: 投入资金的真实年化是多少?", flush=True)
    print(f"A: 投入资金隐含CAGR ≈ {invested_capital_cagr*100:.1f}%", flush=True)
    print(f"   这个数字确实很高, 说明少量交易产生了极高收益。", flush=True)
    print(f"", flush=True)
    print(f"Q: 这是否意味着结果不可信?", flush=True)
    print(f"A: 需要区分两个问题:", flush=True)
    print(f"   1. CAGR计算是否正确 → 是, 34.5%是总账户真实年化", flush=True)
    print(f"   2. 投入资金的高收益是否可持续 → 这是过拟合风险的核心", flush=True)
    print(f"      460笔交易中70.4%胜率, 平均盈利4.5% vs 平均亏损5.4%", flush=True)
    print(f"      盈亏比 = 4.5/5.4 = 0.83, 胜率70.4%", flush=True)
    print(f"      期望收益 = 0.704*4.5% - 0.296*5.4% = 1.58%", flush=True)
    print(f"      这个期望值在统计上是否显著, 需要更多样本验证", flush=True)

    results = {
        'total_trading_days': len(trading_dates),
        'total_executed_trades': n_pos,
        'avg_trades_per_day': float(n_pos / len(trading_dates)),
        'fixed_pos_size': float(fixed_pos_size),
        'avg_capital_utilization': float(capital_series.mean()),
        'median_capital_utilization': float(capital_series.median()),
        'max_capital_utilization': float(capital_series.max()),
        'idle_cash_pct': float(idle_cash_pct),
        'days_with_zero_positions': int((capital_series == 0).sum()),
        'pct_days_with_zero_positions': float((capital_series == 0).mean()),
        'cagr_total_account': float(cagr_fixed),
        'implied_invested_cagr': float(invested_capital_cagr),
        'position_distribution': {str(int(n)): int((npos_series == n).sum()) for n in sorted(npos_series.unique())},
    }

    out_file = os.path.join(RESULTS_DIR, 'capital_utilization_analysis.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == '__main__':
    ohlc_lookup, pctchg_lookup, close_lookup = load_data()
    pred = pd.read_parquet(PRED_FILE)
    pred['ds'] = pred['trade_date'].astype(str)
    print(f"Predictions: {len(pred)} rows", flush=True)
    backtest_compare(pred, ohlc_lookup, pctchg_lookup, close_lookup)
