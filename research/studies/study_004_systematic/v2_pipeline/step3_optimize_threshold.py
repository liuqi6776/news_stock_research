"""
Step 3: 参数优化 v2 (无clip)

核心: 不做clip止盈止损，只优化threshold + max_positions
优化期: 2022-2025, 验证期: 2026

输入: predictions/predictions_1d_wf.parquet
输出: results/grid_search_v2_results.parquet, results/optimized_params_v2.json

耗时: 约5分钟
运行频率: Step 2 更新后
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (
    WF_PREDICTIONS_FILE, OPT_RESULTS_FILE, GRID_RESULTS_FILE,
    RESULTS_DIR, TRANSACTION_COST,
    THRESHOLD_RANGE_START, THRESHOLD_RANGE_END, THRESHOLD_RANGE_STEP,
    MAX_POSITIONS_RANGE, OPT_START, OPT_END, VAL_START, VAL_END
)


def backtest_simple(df, start, end, threshold, max_positions):
    mask = (df['ds'] >= start) & (df['ds'] <= end)
    pdf = df[mask]
    if len(pdf) == 0:
        return None

    above = pdf[pdf['prob'] >= threshold].copy()
    above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
    selected = above[above['rank'] <= max_positions]
    if len(selected) == 0:
        return None

    pos_size = 1.0 / max_positions
    rets = selected['actual_return'].values
    rets_after_cost = rets - TRANSACTION_COST

    daily_pnl = selected.groupby('ds').apply(
        lambda g: pos_size * (g['actual_return'].values - TRANSACTION_COST).sum()
    ).sort_index().values

    n_days = len(daily_pnl)
    n_years = n_days / 252
    if n_years <= 0:
        return None

    equity = np.cumprod(1 + daily_pnl)
    total_return = equity[-1] - 1
    cagr = (equity[-1] ** (1 / n_years) - 1) if n_years > 0 else 0

    running_max = np.maximum.accumulate(equity)
    max_dd = ((equity - running_max) / running_max).min()

    avg_daily = np.mean(daily_pnl)
    std_daily = np.std(daily_pnl)
    sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 1e-10 else 0

    win_rate = (rets_after_cost > 0).mean()
    n_trades = len(rets_after_cost)
    avg_ret = float(np.mean(rets_after_cost))

    return {
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'total_return': float(total_return),
        'win_rate': float(win_rate),
        'n_trades': n_trades,
        'avg_return': avg_ret,
        'n_days': n_days,
        'avg_win': float(np.mean(rets_after_cost[rets_after_cost > 0])) if np.any(rets_after_cost > 0) else 0,
        'avg_loss': float(np.mean(rets_after_cost[rets_after_cost <= 0])) if np.any(rets_after_cost <= 0) else 0,
        'pct_big_loss': float(np.mean(rets_after_cost < -0.05)),
        'pct_big_win': float(np.mean(rets_after_cost > 0.05)),
    }


def plot_equity(df, start, end, threshold, max_positions, period_name):
    mask = (df['ds'] >= start) & (df['ds'] <= end)
    pdf = df[mask]
    above = pdf[pdf['prob'] >= threshold].copy()
    above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
    selected = above[above['rank'] <= max_positions]

    pos_size = 1.0 / max_positions
    daily_pnl = selected.groupby('ds').apply(
        lambda g: pos_size * (g['actual_return'].values - TRANSACTION_COST).sum()
    ).sort_index().values

    equity = np.cumprod(1 + daily_pnl)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max

    trading_dates = sorted(pdf['ds'].unique())
    dates_pd = pd.to_datetime(trading_dates[:len(daily_pnl)], format='%Y%m%d')

    r = backtest_simple(df, start, end, threshold, max_positions)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1 = axes[0]
    ax1.plot(dates_pd, equity, 'b-', linewidth=1.0)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f'v2 No Clip - {period_name}\n'
                  f'thresh={threshold:.2f}, pos={max_positions}\n'
                  f'CAGR={r["cagr"]:.2%}, Sharpe={r["sharpe"]:.2f}, MaxDD={r["max_drawdown"]:.2%}',
                  fontsize=13)
    ax1.set_ylabel('Equity')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax2 = axes[1]
    ax2.fill_between(dates_pd, drawdown, 0, color='red', alpha=0.4)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    ax3 = axes[2]
    colors = ['green' if p > 0 else 'red' for p in daily_pnl]
    ax3.bar(dates_pd, daily_pnl, color=colors, alpha=0.6, width=1)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title('Daily PnL')
    ax3.set_ylabel('PnL')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f'equity_v2_{period_name.replace(" ", "_")}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表: {fig_path}")


def run():
    print("=" * 80)
    print("Step 3: 参数优化 v2 (无clip, 只优化 threshold + max_positions)")
    print(f"优化期: {OPT_START}-{OPT_END}, 验证期: {VAL_START}-{VAL_END}")
    print("=" * 80)

    if not os.path.exists(WF_PREDICTIONS_FILE):
        print("错误: 请先运行 step2_walkforward_predict.py")
        return None

    df = pd.read_parquet(WF_PREDICTIONS_FILE)
    df['ds'] = df['trade_date'].astype(str)
    df = df.dropna(subset=['actual_return']).copy()
    print(f"数据: {len(df)} 行, {df['ds'].min()} - {df['ds'].max()}")

    threshold_range = np.arange(THRESHOLD_RANGE_START, THRESHOLD_RANGE_END, THRESHOLD_RANGE_STEP)
    results = []

    for threshold in threshold_range:
        for max_pos in MAX_POSITIONS_RANGE:
            opt = backtest_simple(df, OPT_START, OPT_END, threshold, max_pos)
            if opt is None or opt['n_trades'] < 50:
                continue

            val = backtest_simple(df, VAL_START, VAL_END, threshold, max_pos)

            results.append({
                'threshold': float(threshold),
                'max_positions': int(max_pos),
                'opt_cagr': opt['cagr'],
                'opt_sharpe': opt['sharpe'],
                'opt_max_dd': opt['max_drawdown'],
                'opt_win_rate': opt['win_rate'],
                'opt_n_trades': opt['n_trades'],
                'opt_avg_return': opt['avg_return'],
                'opt_avg_win': opt['avg_win'],
                'opt_avg_loss': opt['avg_loss'],
                'opt_pct_big_loss': opt['pct_big_loss'],
                'opt_pct_big_win': opt['pct_big_win'],
                'val_cagr': val['cagr'] if val else None,
                'val_sharpe': val['sharpe'] if val else None,
                'val_max_dd': val['max_drawdown'] if val else None,
                'val_win_rate': val['win_rate'] if val else None,
                'val_n_trades': val['n_trades'] if val else 0,
                'val_avg_return': val['avg_return'] if val else None,
            })

    rdf = pd.DataFrame(results)
    print(f"\n有效组合: {len(rdf)}")

    print("\n" + "=" * 80)
    print("全部结果 (按优化期Sharpe排序)")
    print("=" * 80)
    rdf_sorted = rdf.sort_values('opt_sharpe', ascending=False)
    for _, row in rdf_sorted.iterrows():
        val_cagr_str = f"{row['val_cagr']:.2%}" if row['val_cagr'] is not None else "N/A"
        val_sharpe_str = f"{row['val_sharpe']:.2f}" if row['val_sharpe'] is not None else "N/A"
        print(f"  thresh={row['threshold']:.2f}, pos={row['max_positions']} | "
              f"opt: CAGR={row['opt_cagr']:.2%}, Sharpe={row['opt_sharpe']:.2f}, "
              f"DD={row['opt_max_dd']:.2%}, WR={row['opt_win_rate']:.2%}, n={row['opt_n_trades']} | "
              f"val: CAGR={val_cagr_str}, Sharpe={val_sharpe_str}, n={row['val_n_trades']}")

    print("\n" + "=" * 80)
    print("按验证期Sharpe排序 Top5")
    print("=" * 80)
    rdf_val = rdf[rdf['val_sharpe'].notna()].sort_values('val_sharpe', ascending=False)
    for _, row in rdf_val.head(5).iterrows():
        print(f"  thresh={row['threshold']:.2f}, pos={row['max_positions']} | "
              f"opt: Sharpe={row['opt_sharpe']:.2f}, CAGR={row['opt_cagr']:.2%} | "
              f"val: Sharpe={row['val_sharpe']:.2f}, CAGR={row['val_cagr']:.2%}, "
              f"DD={row['val_max_dd']:.2%}, n={row['val_n_trades']}")

    best_idx = rdf['opt_sharpe'].idxmax()
    best = rdf.loc[best_idx]
    print(f"\n最优参数 (按优化期Sharpe): threshold={best['threshold']:.2f}, max_positions={best['max_positions']}")

    for period_name, start, end in [('优化期_2022_2025', OPT_START, OPT_END),
                                     ('验证期_2026', VAL_START, VAL_END)]:
        r = backtest_simple(df, start, end, best['threshold'], int(best['max_positions']))
        if r:
            print(f"\n  {period_name}:")
            print(f"    CAGR={r['cagr']:.2%}, Sharpe={r['sharpe']:.2f}, "
                  f"MaxDD={r['max_drawdown']:.2%}, WinRate={r['win_rate']:.2%}")
            print(f"    交易数={r['n_trades']}, 平均收益={r['avg_return']:.2%}")
            print(f"    平均盈利={r['avg_win']:.2%}, 平均亏损={r['avg_loss']:.2%}")
            print(f"    大亏损(>5%)占比={r['pct_big_loss']:.1%}, 大盈利(>5%)占比={r['pct_big_win']:.1%}")
            plot_equity(df, start, end, best['threshold'], int(best['max_positions']), period_name)

    rdf.to_parquet(GRID_RESULTS_FILE, index=False)
    print(f"\n网格搜索结果已保存: {GRID_RESULTS_FILE}")

    out = {
        'best_params': {
            'threshold': float(best['threshold']),
            'max_positions': int(best['max_positions']),
        },
        'opt_results': backtest_simple(df, OPT_START, OPT_END, best['threshold'], int(best['max_positions'])),
        'val_results': backtest_simple(df, VAL_START, VAL_END, best['threshold'], int(best['max_positions'])),
    }
    with open(OPT_RESULTS_FILE, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"最优参数已保存: {OPT_RESULTS_FILE}")

    return rdf


if __name__ == '__main__':
    run()
