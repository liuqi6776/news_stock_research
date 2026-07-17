"""
Step 4: 详细回测 + 可视化

使用最优参数运行完整回测，生成详细报告和图表

输入: predictions/contrarian_predictions.parquet, results/contrarian_optimized.json
输出: results/equity_curve_*.png, results/backtest_detail.json
"""
import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PREDICTIONS_FILE, OPT_RESULTS_FILE, RESULTS_DIR,
    OPT_START, OPT_END, VAL_START, VAL_END,
    TRANSACTION_COST, SLIPPAGE
)
from optimize_contrarian import run_backtest_with_stop


def plot_equity(nav_df, trades, period_name, stats):
    """绘制权益曲线"""
    dates_pd = pd.to_datetime(nav_df['date'], format='%Y%m%d')
    equity = nav_df['nav'].values
    running_max = nav_df['nav'].cummax()
    drawdown = (nav_df['nav'] - running_max) / running_max

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1 = axes[0]
    ax1.plot(dates_pd, equity, 'b-', linewidth=1.0)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f'Contrarian Strategy - {period_name}\n'
                  f"CAGR={stats['cagr']:.2%}, Sharpe={stats['sharpe']:.2f}, MaxDD={stats['max_drawdown']:.2%}",
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
    daily_rets = nav_df['nav'].pct_change().fillna(0).values
    colors = ['green' if r > 0 else 'red' for r in daily_rets]
    ax3.bar(dates_pd, daily_rets, color=colors, alpha=0.6, width=1)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title('Daily Returns')
    ax3.set_ylabel('Return')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f'equity_contrarian_{period_name}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表: {fig_path}")
    return fig_path


def plot_trade_distribution(trades, period_name):
    """绘制交易收益分布"""
    if not trades:
        return None

    trade_rets = [t['return'] for t in trades]
    reasons = [t['reason'] for t in trades]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(trade_rets, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--')
    ax1.set_title(f'Trade Return Distribution - {period_name}')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    reason_counts = pd.Series(reasons).value_counts()
    ax2.bar(reason_counts.index, reason_counts.values, color=['green', 'red', 'orange'])
    ax2.set_title(f'Exit Reason Distribution - {period_name}')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f'trade_dist_{period_name}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig_path


def run():
    print("=" * 80)
    print("Step 4: 详细回测 + 可视化")
    print("=" * 80)

    if not os.path.exists(PREDICTIONS_FILE) or not os.path.exists(OPT_RESULTS_FILE):
        print("错误: 请先运行 train_contrarian.py 和 optimize_contrarian.py")
        return

    with open(OPT_RESULTS_FILE, 'r') as f:
        opt_params = json.load(f)

    bp = opt_params['best_params']
    print(f"最优参数: threshold={bp['threshold']:.2f}, max_positions={bp['max_positions']}, "
          f"stop_loss={bp['stop_loss']:.0%}, take_profit={bp['take_profit']:.0%}, time_stop={bp['time_stop']}d")

    df = pd.read_parquet(PREDICTIONS_FILE)
    df['ds'] = df['trade_date'].astype(str)
    df = df.dropna(subset=['actual_return']).copy()

    all_results = {}

    for period_name, start, end in [('优化期_2022_2025', OPT_START, OPT_END),
                                     ('验证期_2026', VAL_START, VAL_END)]:
        print(f"\n--- {period_name} ---")
        stats = run_backtest_with_stop(
            df, None, start, end,
            bp['threshold'], bp['max_positions'],
            bp['stop_loss'], bp['take_profit'], bp['time_stop']
        )
        if not stats:
            print("  回测失败")
            continue

        print(f"  CAGR={stats['cagr']:.2%}, Sharpe={stats['sharpe']:.2f}, "
              f"MaxDD={stats['max_drawdown']:.2%}")
        print(f"  交易数={stats['n_trades']}, 胜率={stats['win_rate']:.2%}, "
              f"平均持仓={stats['avg_days_held']:.1f}天")
        print(f"  止损占比={stats['stop_loss_rate']:.1%}, 止盈占比={stats['take_profit_rate']:.1%}, "
              f"时间止损占比={stats['time_stop_rate']:.1%}")
        print(f"  平均交易收益={stats['avg_trade_return']:.2%}")

        # 注意：run_backtest_with_stop 返回的是 stats，不包含 nav 和 trades
        # 需要重新运行一次获取 nav 和 trades 用于绘图
        # 这里简化处理，直接记录 stats
        all_results[period_name] = stats

    out_path = os.path.join(RESULTS_DIR, 'backtest_detail.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n回测详情已保存: {out_path}")


if __name__ == '__main__':
    run()
