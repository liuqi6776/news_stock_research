import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FINAL_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_strategy(name, trades_path, equity_path):
    trades = pd.read_csv(trades_path)
    equity = pd.read_csv(equity_path)

    trades['ret'] = (trades['sell_close'] / trades['buy_price']) - 1 - 0.0015
    trades['raw_ret'] = (trades['sell_close'] / trades['buy_price']) - 1

    print(f"\n{'='*70}")
    print(f"  {name} - 详细分析")
    print(f"{'='*70}")

    print(f"\n--- 基本统计 ---")
    print(f"  总交易次数: {len(trades)}")
    print(f"  交易天数: {trades['date_t2'].nunique()}")
    print(f"  平均每天交易: {len(trades) / trades['date_t2'].nunique():.1f} 只")
    print(f"  胜率: {(trades['ret'] > 0).mean():.2%}")
    print(f"  平均收益: {trades['ret'].mean():.4f} ({trades['ret'].mean()*100:.2f}%)")
    print(f"  中位收益: {trades['ret'].median():.4f} ({trades['ret'].median()*100:.2f}%)")
    print(f"  盈利交易平均: {trades[trades['ret']>0]['ret'].mean():.4f}")
    print(f"  亏损交易平均: {trades[trades['ret']<=0]['ret'].mean():.4f}")
    print(f"  盈亏比: {trades[trades['ret']>0]['ret'].mean() / abs(trades[trades['ret']<=0]['ret'].mean()):.2f}")

    print(f"\n--- 概率分布 ---")
    print(f"  prob均值: {trades['prob'].mean():.4f}")
    print(f"  prob中位: {trades['prob'].median():.4f}")
    print(f"  prob > 0.8: {(trades['prob'] > 0.8).sum()} ({(trades['prob'] > 0.8).mean():.1%})")
    print(f"  prob > 0.6: {(trades['prob'] > 0.6).sum()} ({(trades['prob'] > 0.6).mean():.1%})")
    print(f"  prob > 0.5: {(trades['prob'] > 0.5).sum()} ({(trades['prob'] > 0.5).mean():.1%})")

    print(f"\n--- 按概率分组的收益 ---")
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
        sub = trades[(trades['prob'] >= lo) & (trades['prob'] < hi)]
        if len(sub) > 0:
            print(f"  prob [{lo:.1f}-{hi:.1f}): n={len(sub)}, mean_ret={sub['ret'].mean():.4f}, win_rate={(sub['ret']>0).mean():.2%}")

    print(f"\n--- 收益分布 ---")
    for lo, hi in [(-1, -0.1), (-0.1, -0.05), (-0.05, 0), (0, 0.03), (0.03, 0.05), (0.05, 0.1), (0.1, 1)]:
        sub = trades[(trades['ret'] >= lo) & (trades['ret'] < hi)]
        print(f"  [{lo:+.0%} ~ {hi:+.0%}): {len(sub)} ({len(sub)/len(trades):.1%})")

    print(f"\n--- 月度表现 ---")
    trades['month'] = trades['date_t2'].astype(str).str[:6]
    monthly = trades.groupby('month').agg(
        n=('ret', 'count'),
        mean_ret=('ret', 'mean'),
        win_rate=('ret', lambda x: (x > 0).mean()),
        total_ret=('ret', 'sum')
    )
    pos_months = (monthly['mean_ret'] > 0).sum()
    print(f"  正收益月: {pos_months}/{len(monthly)} ({pos_months/len(monthly):.1%})")
    print(f"  最佳月: {monthly['mean_ret'].idxmax()} ({monthly['mean_ret'].max():.4f})")
    print(f"  最差月: {monthly['mean_ret'].idxmin()} ({monthly['mean_ret'].min():.4f})")

    print(f"\n--- 大亏损交易 ---")
    big_loss = trades.nsmallest(10, 'ret')
    for _, t in big_loss.iterrows():
        print(f"  {t['ts_code']} {t['date_t2']}: ret={t['ret']:.4f}, prob={t['prob']:.4f}")

    print(f"\n--- 大盈利交易 ---")
    big_win = trades.nlargest(10, 'ret')
    for _, t in big_win.iterrows():
        print(f"  {t['ts_code']} {t['date_t2']}: ret={t['ret']:.4f}, prob={t['prob']:.4f}")

    return trades, equity

def improvement_analysis(db_trades, s2_trades, s3_trades):
    print(f"\n{'='*70}")
    print(f"  提升方向分析")
    print(f"{'='*70}")

    print(f"\n--- 1. doubao_result的高收益来源 ---")
    db_trades['ret'] = (db_trades['sell_close'] / db_trades['buy_price']) - 1 - 0.0015
    high_prob = db_trades[db_trades['prob'] > 0.8]
    low_prob = db_trades[db_trades['prob'] <= 0.8]
    print(f"  高概率(>0.8): {len(high_prob)} trades, mean_ret={high_prob['ret'].mean():.4f}, win_rate={(high_prob['ret']>0).mean():.2%}")
    print(f"  低概率(<=0.8): {len(low_prob)} trades, mean_ret={low_prob['ret'].mean():.4f}, win_rate={(low_prob['ret']>0).mean():.2%}")
    print(f"  → 高概率筛选是doubao_result的核心优势")

    print(f"\n--- 2. S2 vs doubao: 多出的交易质量 ---")
    s2_trades['ret'] = (s2_trades['sell_close'] / s2_trades['buy_price']) - 1 - 0.0015
    db_set = set(zip(db_trades['date_t'], db_trades['ts_code']))
    s2_extra = s2_trades[~s2_trades.set_index(['date_t', 'ts_code']).index.isin(db_set)]
    print(f"  S2独有交易: {len(s2_extra)} trades")
    if len(s2_extra) > 0:
        print(f"  S2独有交易 mean_ret: {s2_extra['ret'].mean():.4f}")
        print(f"  S2独有交易 win_rate: {(s2_extra['ret']>0).mean():.2%}")
        print(f"  S2独有交易 prob均值: {s2_extra['prob'].mean():.4f}")
        print(f"  → S2多选的低概率股票拉低了整体收益")

    print(f"\n--- 3. 止盈策略对比 ---")
    for tp in [0.03, 0.05, 0.08, 0.10, 0.15]:
        tp_ret = []
        for _, t in db_trades.iterrows():
            if t['sell_high'] >= t['buy_price'] * (1 + tp):
                tp_ret.append(tp - 0.0015)
            else:
                tp_ret.append(t['ret'])
        tp_ret = np.array(tp_ret)
        cum = np.prod(1 + tp_ret) - 1
        print(f"  止盈{tp:.0%}: 累计收益={cum:.2%}, 触发率={np.mean([1 if t['sell_high'] >= t['buy_price']*(1+tp) else 0 for _,t in db_trades.iterrows()]):.1%}")

    print(f"\n--- 4. 概率阈值优化 ---")
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        sub = s2_trades[s2_trades['prob'] >= thresh]
        if len(sub) > 0:
            cum = np.prod(1 + sub['ret']) - 1
            print(f"  prob>={thresh}: n={len(sub)}, cum_ret={cum:.2%}, mean_ret={sub['ret'].mean():.4f}, win_rate={(sub['ret']>0).mean():.2%}")

    print(f"\n--- 5. 仓位优化建议 ---")
    db_trades['prob_weight'] = db_trades['prob'] / db_trades['prob'].sum() * len(db_trades)
    print(f"  当前: 等权分配")
    print(f"  建议: 按概率加权分配仓位 (prob越高仓位越重)")
    print(f"  效果: 提高高概率股票的资金利用效率")

    print(f"\n--- 6. 综合提升建议 ---")
    print(f"  A. 保持doubao_result的prob>0.8阈值筛选")
    print(f"  B. 引入动态止盈: 大概率股票用更高止盈(10-15%), 小概率用8%")
    print(f"  C. 仓位管理: 按概率加权分配，高概率重仓")
    print(f"  D. 风控: 单日亏损超5%降低次日仓位")
    print(f"  E. S3的score可优化: 调整权重或引入更多因子")
    print(f"  F. 考虑加入成交量/换手率等过滤条件")

def main():
    db_trades, db_eq = analyze_strategy(
        'doubao_result',
        os.path.join(FINAL_DIR, 'doubao', 'trades.csv'),
        os.path.join(FINAL_DIR, 'doubao', 'equity.csv')
    )
    s2_trades, s2_eq = analyze_strategy(
        'NewIdea S2',
        os.path.join(FINAL_DIR, 'NewIdea_S2', 'trades.csv'),
        os.path.join(FINAL_DIR, 'NewIdea_S2', 'equity.csv')
    )
    s3_trades, s3_eq = analyze_strategy(
        'NewIdea S3',
        os.path.join(FINAL_DIR, 'NewIdea_S3', 'trades.csv'),
        os.path.join(FINAL_DIR, 'NewIdea_S3', 'equity.csv')
    )

    improvement_analysis(db_trades, s2_trades, s3_trades)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    for name, trades, color in [('doubao_result', db_trades, '#1f77b4'), ('NewIdea S2', s2_trades, '#2ca02c'), ('NewIdea S3', s3_trades, '#d62728')]:
        trades['ret'].hist(bins=50, alpha=0.5, label=name, color=color, ax=ax1)
    ax1.set_title('Return Distribution')
    ax1.legend()
    ax1.axvline(x=0, color='black', linestyle='--')

    ax2 = axes[0, 1]
    for name, trades, color in [('doubao_result', db_trades, '#1f77b4'), ('NewIdea S2', s2_trades, '#2ca02c'), ('NewIdea S3', s3_trades, '#d62728')]:
        ax2.scatter(trades['prob'], trades['ret'], alpha=0.3, s=10, label=name, color=color)
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Return')
    ax2.set_title('Prob vs Return')
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='--')

    ax3 = axes[1, 0]
    for name, trades, color in [('doubao_result', db_trades, '#1f77b4'), ('NewIdea S2', s2_trades, '#2ca02c'), ('NewIdea S3', s3_trades, '#d62728')]:
        trades['month'] = trades['date_t2'].astype(str).str[:6]
        monthly = trades.groupby('month')['ret'].mean()
        monthly.cumsum().plot(ax=ax3, label=name, color=color)
    ax3.set_title('Cumulative Monthly Mean Return')
    ax3.legend()

    ax4 = axes[1, 1]
    prob_bins = np.arange(0, 1.01, 0.1)
    for name, trades, color in [('doubao_result', db_trades, '#1f77b4'), ('NewIdea S2', s2_trades, '#2ca02c'), ('NewIdea S3', s3_trades, '#d62728')]:
        trades['prob_bin'] = pd.cut(trades['prob'], bins=prob_bins)
        grp = trades.groupby('prob_bin')['ret'].mean()
        ax4.bar(range(len(grp)), grp.values, alpha=0.5, label=name, color=color, width=0.25)
    ax4.set_xticks(range(len(grp)))
    ax4.set_xticklabels([str(b) for b in grp.index], rotation=45, fontsize=7)
    ax4.set_xlabel('Prob Bin')
    ax4.set_ylabel('Mean Return')
    ax4.set_title('Mean Return by Prob Bin')
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_DIR, 'analysis_charts.png'), dpi=150, bbox_inches='tight')
    print(f"\nAnalysis charts saved to analysis_charts.png")

if __name__ == "__main__":
    main()
