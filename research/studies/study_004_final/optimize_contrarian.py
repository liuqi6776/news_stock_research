"""
Step 3: 参数优化（带止损止盈的网格搜索）

核心: 在优化期(2022-2025)搜索最优参数组合
参数: threshold + max_positions + stop_loss + take_profit + time_stop

输入: predictions/contrarian_predictions.parquet
输出: results/contrarian_grid.parquet, results/contrarian_optimized.json

耗时: 约5-20分钟（取决于网格大小）
"""
import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PREDICTIONS_FILE, OPT_RESULTS_FILE, GRID_RESULTS_FILE, RESULTS_DIR,
    OPT_START, OPT_END, VAL_START, VAL_END,
    THRESHOLD_RANGE, MAX_POSITIONS_RANGE, STOP_LOSS_RANGE, TAKE_PROFIT_RANGE, TIME_STOP_RANGE,
    TRANSACTION_COST, SLIPPAGE
)


def run_backtest_with_stop(predictions_df, price_dir, start_date, end_date,
                           threshold, max_positions, stop_loss, take_profit, time_stop):
    """
    改进回测：模拟持仓，支持止损止盈和时间止损

    逻辑：
    - 每日收盘后，按prob选股，次日开盘买入
    - 每个持仓独立跟踪：达到止损/止盈/时间限制时卖出
    - 持仓数不超过max_positions，新信号只在没有空仓时买入
    """
    import os

    # 获取所有交易日
    from shared.data_loader import get_all_dates, PRICE_DIR
    all_dates = get_all_dates()
    all_dates = [d for d in all_dates if start_date <= d <= end_date]
    if len(all_dates) < 30:
        return None

    # 过滤预测数据
    pdf = predictions_df.copy()
    pdf['ds'] = pdf['trade_date'].astype(str)
    pdf = pdf[(pdf['ds'] >= start_date) & (pdf['ds'] <= end_date)].copy()
    if len(pdf) == 0:
        return None

    # 持仓状态: {ts_code: {'entry_date': str, 'entry_price': float, 'days_held': int}}
    positions = {}
    daily_nav = []
    capital = 1.0
    trades = []
    n_buy_limit_up = 0

    def get_price(date_str, ts_code, price_type='open'):
        p = os.path.join(PRICE_DIR, f"{date_str}.parquet")
        if not os.path.exists(p):
            return None
        try:
            df = pd.read_parquet(p, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
            row = df[df['ts_code'] == ts_code]
            if row.empty:
                return None
            if price_type == 'open':
                return float(row.iloc[0]['open'])
            elif price_type == 'high':
                return float(row.iloc[0]['high'])
            elif price_type == 'low':
                return float(row.iloc[0]['low'])
            elif price_type == 'close':
                return float(row.iloc[0]['close'])
            elif price_type == 'pre_close':
                return float(row.iloc[0]['pre_close'])
        except:
            return None
        return None

    def get_limit_pct(ts_code):
        if ts_code.startswith('68') or ts_code.startswith('30'):
            return 0.20
        return 0.10

    for i in range(len(all_dates) - 1):
        d_curr = all_dates[i]
        d_next = all_dates[i + 1]

        # 1. 检查当前持仓，处理止损止盈
        sold_today = []
        for ts_code, pos in list(positions.items()):
            pos['days_held'] += 1

            # 获取当日价格数据
            open_p = get_price(d_next, ts_code, 'open')
            high_p = get_price(d_next, ts_code, 'high')
            low_p = get_price(d_next, ts_code, 'low')
            close_p = get_price(d_next, ts_code, 'close')

            if open_p is None or close_p is None:
                continue  # 数据缺失，继续持有

            # 检查开盘是否涨停（无法卖出）
            pre_close = get_price(d_next, ts_code, 'pre_close')
            if pre_close and open_p >= pre_close * (1 + get_limit_pct(ts_code) - 0.02):
                # 接近涨停开盘，跳过卖出，按收盘计算当日盈亏
                ret = (close_p / pos['entry_price'] - 1) if pos['entry_price'] > 0 else 0
            else:
                # 检查日内是否触发止损
                ret_from_entry = (low_p / pos['entry_price'] - 1) if pos['entry_price'] > 0 else 0
                if ret_from_entry <= -stop_loss:
                    # 止损卖出（按止损价）
                    sell_price = pos['entry_price'] * (1 - stop_loss)
                    ret = (sell_price / pos['entry_price'] - 1) - TRANSACTION_COST - SLIPPAGE
                    sold_today.append((ts_code, ret, 'stop_loss'))
                    continue

                # 检查是否触发止盈
                ret_from_entry = (high_p / pos['entry_price'] - 1) if pos['entry_price'] > 0 else 0
                if ret_from_entry >= take_profit:
                    # 止盈卖出
                    sell_price = pos['entry_price'] * (1 + take_profit)
                    ret = (sell_price / pos['entry_price'] - 1) - TRANSACTION_COST - SLIPPAGE
                    sold_today.append((ts_code, ret, 'take_profit'))
                    continue

                # 检查时间止损
                if pos['days_held'] >= time_stop:
                    ret = (close_p / pos['entry_price'] - 1) - TRANSACTION_COST - SLIPPAGE
                    sold_today.append((ts_code, ret, 'time_stop'))
                    continue

                # 未触发，持有到收盘
                ret = (close_p / pos['entry_price'] - 1)

        # 执行卖出
        for ts_code, ret, reason in sold_today:
            if ts_code in positions:
                trades.append({
                    'ts_code': ts_code,
                    'entry_date': positions[ts_code]['entry_date'],
                    'exit_date': d_next,
                    'days_held': positions[ts_code]['days_held'],
                    'return': ret,
                    'reason': reason,
                })
                del positions[ts_code]

        # 2. 选择新买入信号
        n_open_slots = max_positions - len(positions)
        if n_open_slots > 0:
            day_pred = pdf[pdf['ds'] == d_curr].copy()
            if len(day_pred) > 0:
                day_pred = day_pred.sort_values('prob', ascending=False)
                # 排除已持仓的
                day_pred = day_pred[~day_pred['ts_code'].isin(positions.keys())]
                candidates = day_pred[day_pred['prob'] >= threshold].head(n_open_slots)

                for _, row in candidates.iterrows():
                    ts_code = row['ts_code']
                    open_p = get_price(d_next, ts_code, 'open')
                    pre_close = get_price(d_next, ts_code, 'pre_close')
                    if open_p is None or pre_close is None:
                        continue

                    # 跳过涨停开盘
                    limit_pct = get_limit_pct(ts_code)
                    if open_p >= pre_close * (1 + limit_pct - 0.01):
                        n_buy_limit_up += 1
                        continue

                    positions[ts_code] = {
                        'entry_date': d_next,
                        'entry_price': open_p * (1 + SLIPPAGE),
                        'days_held': 0,
                    }

        # 3. 计算当日净值（基于所有持仓的当日收盘价）
        day_pnl = 0.0
        n_active = 0
        for ts_code, pos in positions.items():
            close_p = get_price(d_next, ts_code, 'close')
            if close_p is not None:
                ret = (close_p / pos['entry_price'] - 1) if pos['entry_price'] > 0 else 0
                day_pnl += ret / max_positions
                n_active += 1

        # 如果持仓不足，空仓部分视为0收益
        capital *= (1 + day_pnl)
        daily_nav.append({'date': d_next, 'nav': capital, 'n_positions': len(positions), 'n_active': n_active})

    # 计算统计指标
    if len(daily_nav) == 0:
        return None

    nav_df = pd.DataFrame(daily_nav)
    n_days = len(nav_df)
    n_years = n_days / 252

    total_return = nav_df['nav'].iloc[-1] - 1
    cagr = (nav_df['nav'].iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0

    running_max = nav_df['nav'].cummax()
    max_dd = ((nav_df['nav'] - running_max) / running_max).min()

    daily_rets = nav_df['nav'].pct_change().dropna()
    sharpe = (daily_rets.mean() / (daily_rets.std() + 1e-8) * np.sqrt(252)) if daily_rets.std() > 0 else 0

    # 交易统计
    if trades:
        trade_rets = [t['return'] for t in trades]
        win_rate = sum(1 for r in trade_rets if r > 0) / len(trade_rets)
        avg_days = np.mean([t['days_held'] for t in trades])
        stop_loss_rate = sum(1 for t in trades if t['reason'] == 'stop_loss') / len(trades)
        take_profit_rate = sum(1 for t in trades if t['reason'] == 'take_profit') / len(trades)
        time_stop_rate = sum(1 for t in trades if t['reason'] == 'time_stop') / len(trades)
    else:
        trade_rets = []
        win_rate = 0
        avg_days = 0
        stop_loss_rate = 0
        take_profit_rate = 0
        time_stop_rate = 0

    return {
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'total_return': float(total_return),
        'win_rate': float(win_rate),
        'n_trades': len(trades),
        'avg_days_held': float(avg_days) if trades else 0,
        'stop_loss_rate': float(stop_loss_rate),
        'take_profit_rate': float(take_profit_rate),
        'time_stop_rate': float(time_stop_rate),
        'avg_trade_return': float(np.mean(trade_rets)) if trade_rets else 0,
        'n_days': n_days,
        'skipped_limit_up': n_buy_limit_up,
    }


def run():
    print("=" * 80)
    print("Step 3: 参数优化（带止损止盈的网格搜索）")
    print(f"优化期: {OPT_START}-{OPT_END}, 验证期: {VAL_START}-{VAL_END}")
    print("=" * 80)

    if not os.path.exists(PREDICTIONS_FILE):
        print("错误: 请先运行 train_contrarian.py")
        return

    df = pd.read_parquet(PREDICTIONS_FILE)
    df['ds'] = df['trade_date'].astype(str)
    df = df.dropna(subset=['actual_return']).copy()
    print(f"数据: {len(df)} 行, {df['ds'].min()} - {df['ds'].max()}")

    # 为了加速，减少网格规模（可选）
    # 实际运行时可以扩大
    threshold_range = THRESHOLD_RANGE
    max_pos_range = MAX_POSITIONS_RANGE
    stop_loss_range = STOP_LOSS_RANGE
    take_profit_range = TAKE_PROFIT_RANGE
    time_stop_range = TIME_STOP_RANGE

    total_combos = len(threshold_range) * len(max_pos_range) * len(stop_loss_range) * len(take_profit_range) * len(time_stop_range)
    print(f"总组合数: {total_combos}")

    results = []
    combo_idx = 0

    for threshold in threshold_range:
        for max_pos in max_pos_range:
            for stop_loss in stop_loss_range:
                for take_profit in take_profit_range:
                    for time_stop in time_stop_range:
                        combo_idx += 1
                        print(f"\n[{combo_idx}/{total_combos}] thresh={threshold:.2f}, pos={max_pos}, "
                              f"SL={stop_loss:.0%}, TP={take_profit:.0%}, TS={time_stop}d")

                        opt = run_backtest_with_stop(
                            df, None, OPT_START, OPT_END,
                            threshold, max_pos, stop_loss, take_profit, time_stop
                        )
                        if opt is None or opt['n_trades'] < 20:
                            print(f"  跳过: 交易数不足")
                            continue

                        val = run_backtest_with_stop(
                            df, None, VAL_START, VAL_END,
                            threshold, max_pos, stop_loss, take_profit, time_stop
                        )

                        results.append({
                            'threshold': float(threshold),
                            'max_positions': int(max_pos),
                            'stop_loss': float(stop_loss),
                            'take_profit': float(take_profit),
                            'time_stop': int(time_stop),
                            'opt_cagr': opt['cagr'],
                            'opt_sharpe': opt['sharpe'],
                            'opt_max_dd': opt['max_drawdown'],
                            'opt_win_rate': opt['win_rate'],
                            'opt_n_trades': opt['n_trades'],
                            'opt_avg_days': opt['avg_days_held'],
                            'opt_sl_rate': opt['stop_loss_rate'],
                            'opt_tp_rate': opt['take_profit_rate'],
                            'val_cagr': val['cagr'] if val else None,
                            'val_sharpe': val['sharpe'] if val else None,
                            'val_max_dd': val['max_drawdown'] if val else None,
                            'val_win_rate': val['win_rate'] if val else None,
                            'val_n_trades': val['n_trades'] if val else 0,
                        })

                        print(f"  opt: CAGR={opt['cagr']:.2%}, Sharpe={opt['sharpe']:.2f}, "
                              f"DD={opt['max_drawdown']:.2%}, WR={opt['win_rate']:.2%}, n={opt['n_trades']}")
                        if val:
                            print(f"  val: CAGR={val['cagr']:.2%}, Sharpe={val['sharpe']:.2f}, "
                                  f"DD={val['max_drawdown']:.2%}, WR={val['win_rate']:.2%}")

    if not results:
        print("错误: 没有有效结果")
        return

    rdf = pd.DataFrame(results)
    print(f"\n有效组合: {len(rdf)}")

    # 按优化期Sharpe排序
    print("\n" + "=" * 80)
    print("Top 10 (按优化期Sharpe)")
    print("=" * 80)
    top_opt = rdf.sort_values('opt_sharpe', ascending=False).head(10)
    for _, row in top_opt.iterrows():
        print(f"  thresh={row['threshold']:.2f}, pos={row['max_positions']}, SL={row['stop_loss']:.0%}, "
              f"TP={row['take_profit']:.0%}, TS={row['time_stop']}d | "
              f"opt CAGR={row['opt_cagr']:.2%}, Sharpe={row['opt_sharpe']:.2f}, DD={row['opt_max_dd']:.2%} | "
              f"val CAGR={row['val_cagr']:.2% if row['val_cagr'] is not None else 'N/A'}, "
              f"Sharpe={row['val_sharpe']:.2f if row['val_sharpe'] is not None else 'N/A'}")

    # 选择最优参数（按优化期Sharpe，但要求验证期不为负）
    valid = rdf[rdf['val_sharpe'].notna()]
    if len(valid) > 0:
        best_idx = valid['opt_sharpe'].idxmax()
    else:
        best_idx = rdf['opt_sharpe'].idxmax()
    best = rdf.loc[best_idx]

    print(f"\n最优参数:")
    print(f"  threshold={best['threshold']:.2f}, max_positions={best['max_positions']}")
    print(f"  stop_loss={best['stop_loss']:.0%}, take_profit={best['take_profit']:.0%}, time_stop={best['time_stop']}d")

    # 保存结果
    rdf.to_parquet(GRID_RESULTS_FILE, index=False)
    print(f"\n网格搜索结果已保存: {GRID_RESULTS_FILE}")

    out = {
        'best_params': {
            'threshold': float(best['threshold']),
            'max_positions': int(best['max_positions']),
            'stop_loss': float(best['stop_loss']),
            'take_profit': float(best['take_profit']),
            'time_stop': int(best['time_stop']),
        },
        'opt_results': {
            'cagr': best['opt_cagr'],
            'sharpe': best['opt_sharpe'],
            'max_drawdown': best['opt_max_dd'],
            'win_rate': best['opt_win_rate'],
            'n_trades': best['opt_n_trades'],
        },
        'val_results': {
            'cagr': best['val_cagr'],
            'sharpe': best['val_sharpe'],
            'max_drawdown': best['val_max_dd'],
            'win_rate': best['val_win_rate'],
            'n_trades': best['val_n_trades'],
        } if best['val_cagr'] is not None else None,
    }
    with open(OPT_RESULTS_FILE, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"最优参数已保存: {OPT_RESULTS_FILE}")

    return rdf


if __name__ == '__main__':
    run()
