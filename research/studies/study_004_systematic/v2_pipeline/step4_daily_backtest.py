"""
Step 4: 逐日回测 + 收益分布分析

使用Step 3的最优参数，生成详细的逐日回测结果和收益分布分析

输入: predictions/predictions_1d_wf.parquet, results/optimized_params_v2.json
输出: results/daily_backtest_*.json, results/equity_curve_*.png, results/return_distribution.json

耗时: 约1分钟
运行频率: Step 3 更新后
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
    WF_PREDICTIONS_FILE, OPT_RESULTS_FILE, RESULTS_DIR,
    TRANSACTION_COST, OPT_START, OPT_END, VAL_START, VAL_END
)


def run_daily_backtest(df, start, end, threshold, max_positions, period_name):
    mask = (df['ds'] >= start) & (df['ds'] <= end)
    pdf = df[mask].copy()
    if len(pdf) == 0:
        return None, None

    trading_dates = sorted(pdf['ds'].unique())
    selected = pdf[pdf['prob'] >= threshold].copy()
    selected['rank'] = selected.groupby('ds')['prob'].rank(ascending=False, method='first')
    selected = selected[selected['rank'] <= max_positions].copy()

    pos_size = 1.0 / max_positions

    daily_pnl_list = []
    for d in trading_dates:
        day_trades = selected[selected['ds'] == d]
        if len(day_trades) == 0:
            daily_pnl_list.append(0.0)
        else:
            day_pnl = pos_size * (day_trades['actual_return'].values - TRANSACTION_COST).sum()
            daily_pnl_list.append(day_pnl)

    daily_returns = np.array(daily_pnl_list)
    equity_curve = np.cumprod(1 + daily_returns)

    n_trading_days = len(trading_dates)
    n_years = n_trading_days / 252

    total_return = equity_curve[-1] - 1
    cagr = (equity_curve[-1] ** (1 / n_years) - 1) if n_years > 0 else 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    win_days = int(np.sum(daily_returns > 0))
    win_rate = win_days / n_trading_days

    avg_daily = np.mean(daily_returns)
    std_daily = np.std(daily_returns)
    sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 1e-10 else 0

    n_trades = len(selected)
    trade_rets = selected['actual_return'] - TRANSACTION_COST
    trade_win = (trade_rets > 0).mean()

    stats = {
        'period': period_name,
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'win_rate_days': float(win_rate),
        'avg_daily_return': float(avg_daily),
        'n_trading_days': n_trading_days,
        'n_trades': n_trades,
        'trade_win_rate': float(trade_win),
        'threshold': float(threshold),
        'max_positions': int(max_positions),
    }

    dates_pd = pd.to_datetime(trading_dates, format='%Y%m%d')

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1 = axes[0]
    ax1.plot(dates_pd, equity_curve, 'b-', linewidth=1.0)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f'v2 No Clip - {period_name}\n'
                  f'thresh={threshold:.2f}, pos={max_positions}\n'
                  f'CAGR={cagr:.2%}, Sharpe={sharpe:.2f}, MaxDD={max_drawdown:.2%}',
                  fontsize=14)
    ax1.set_ylabel('Equity')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax2 = axes[1]
    ax2.fill_between(dates_pd, drawdown, 0, color='red', alpha=0.4)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    ax3 = axes[2]
    colors = ['green' if r > 0 else 'red' for r in daily_returns]
    ax3.bar(dates_pd, daily_returns, color=colors, alpha=0.6, width=1)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title('Daily Returns')
    ax3.set_ylabel('Return')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f'equity_curve_{period_name}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    return stats, fig_path


def analyze_return_distribution(df, start, end, threshold, max_positions, period_name):
    mask = (df['ds'] >= start) & (df['ds'] <= end)
    pdf = df[mask]
    above = pdf[pdf['prob'] >= threshold].copy()
    above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
    selected = above[above['rank'] <= max_positions]
    if len(selected) == 0:
        return None

    rets = selected['actual_return']
    dist = {
        'period': period_name,
        'n_trades': len(rets),
        'mean': float(rets.mean()),
        'median': float(rets.median()),
        'win_rate': float((rets > 0).mean()),
        'q5': float(rets.quantile(0.05)),
        'q25': float(rets.quantile(0.25)),
        'q75': float(rets.quantile(0.75)),
        'q95': float(rets.quantile(0.95)),
        'pct_big_loss': float((rets < -0.05).mean()),
        'pct_big_win': float((rets > 0.05).mean()),
        'max_loss': float(rets.min()),
        'max_win': float(rets.max()),
    }

    print(f"\n  {period_name} 收益分布 (thresh={threshold}, top{max_positions}):")
    print(f"    交易数: {dist['n_trades']}")
    print(f"    均值: {dist['mean']:.4f}, 中位数: {dist['median']:.4f}")
    print(f"    胜率: {dist['win_rate']:.2%}")
    print(f"    分位数: 5%={dist['q5']:.4f}, 25%={dist['q25']:.4f}, "
          f"75%={dist['q75']:.4f}, 95%={dist['q95']:.4f}")
    print(f"    极端亏损(>5%): {dist['pct_big_loss']:.2%}, "
          f"极端盈利(>5%): {dist['pct_big_win']:.2%}")
    print(f"    最大亏损: {dist['max_loss']:.4f}, 最大盈利: {dist['max_win']:.4f}")

    return dist


def run():
    print("=" * 80)
    print("Step 4: 逐日回测 + 收益分布分析")
    print("=" * 80)

    if not os.path.exists(WF_PREDICTIONS_FILE):
        print("错误: 请先运行 step2_walkforward_predict.py")
        return

    if not os.path.exists(OPT_RESULTS_FILE):
        print("错误: 请先运行 step3_optimize_threshold.py")
        return

    with open(OPT_RESULTS_FILE, 'r') as f:
        opt_params = json.load(f)

    threshold = opt_params['best_params']['threshold']
    max_positions = opt_params['best_params']['max_positions']
    print(f"使用最优参数: threshold={threshold:.2f}, max_positions={max_positions}")

    df = pd.read_parquet(WF_PREDICTIONS_FILE)
    df['ds'] = df['trade_date'].astype(str)
    df = df.dropna(subset=['actual_return']).copy()
    print(f"数据: {len(df)} 行, {df['ds'].min()} - {df['ds'].max()}")

    all_stats = {}
    all_dists = {}

    for period_name, start, end in [('opt_2022_2025', OPT_START, OPT_END),
                                     ('val_2026', VAL_START, VAL_END)]:
        print(f"\n--- {period_name} ---")
        stats, fig_path = run_daily_backtest(df, start, end, threshold, max_positions, period_name)
        if stats:
            all_stats[period_name] = stats
            print(f"  CAGR={stats['cagr']:.2%}, Sharpe={stats['sharpe']:.2f}, "
                  f"MaxDD={stats['max_drawdown']:.2%}, WinRate={stats['win_rate_days']:.2%}")
            print(f"  交易数={stats['n_trades']}, 交易胜率={stats['trade_win_rate']:.2%}")
            print(f"  图表: {fig_path}")

        dist = analyze_return_distribution(df, start, end, threshold, max_positions, period_name)
        if dist:
            all_dists[period_name] = dist

    out_path = os.path.join(RESULTS_DIR, 'step4_daily_backtest.json')
    with open(out_path, 'w') as f:
        json.dump({'stats': all_stats, 'distributions': all_dists}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n回测结果已保存: {out_path}")


if __name__ == '__main__':
    run()
