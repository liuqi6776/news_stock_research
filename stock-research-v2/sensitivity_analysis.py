# -*- coding: utf-8 -*-
"""
sensitivity_analysis.py - 融券对冲成本敏感度分析与费率门槛测算
============================================================

本脚本计算当做空工具（融券/期货）年化摩擦成本从 2% 递增至 10% 时，
对冲组合 (Long A0, Short CSI 1000 Spot) 的年化收益率 (CAGR) 与夏普比率 (Sharpe) 的变化情况，
并测算出能够维持夏普比率在 0.5 以上的最大容忍对冲成本。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 确保中文字体显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r"C:\Users\liuqi\quant_system_v2"
DAILY_CSV = os.path.join(ROOT, "research", "studies", "study_007_cross_sectional", "fix", "results_fixed", "hedged_backtest_daily.csv")
OUTPUT_DIR = os.path.join(ROOT, "research", "studies", "study_007_cross_sectional", "fix", "results_fixed")

def run_sensitivity_analysis():
    print("=" * 70)
    print(" 启动融券对冲成本敏感度与费率门槛分析 ")
    print("=" * 70)

    if not os.path.exists(DAILY_CSV):
        raise FileNotFoundError(f"需要先运行 backtest_hedged.py 生成: {DAILY_CSV}")

    df = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)
    
    # 纯选股 Alpha 收益率（现货对冲）
    ret_alpha = df["ret_hedged_spot"]
    
    costs = np.arange(0.01, 0.11, 0.005) # 1% 到 10%，步长 0.5%
    records = []
    
    n_days = len(ret_alpha)
    years = n_days / 242.0

    for c in costs:
        # 每日扣除对冲成本 (c / 242)
        daily_ret = ret_alpha - c / 242.0
        nav = (1 + daily_ret).cumprod()
        
        cagr = nav.iloc[-1] ** (1.0 / years) - 1.0
        vol = daily_ret.std() * np.sqrt(242)
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(242)) if daily_ret.std() > 0 else 0
        cummax = nav.cummax()
        mdd = (nav / cummax - 1.0).min()
        
        records.append({
            "annual_cost": c,
            "cagr": cagr,
            "volatility": vol,
            "sharpe": sharpe,
            "max_drawdown": mdd
        })

    res_df = pd.DataFrame(records)
    
    # 寻找夏普比率 >= 0.5 的临界点
    threshold_sharpe = 0.5
    under_threshold = res_df[res_df["sharpe"] < threshold_sharpe]
    
    if len(under_threshold) > 0:
        idx = under_threshold.index[0]
        if idx == 0:
            max_cost = 0.0
            print("警告: 即使在最低成本下，夏普比率也未达到 0.5。")
        else:
            # 线性插值估算精确临界值
            c0, s0 = res_df.loc[idx-1, "annual_cost"], res_df.loc[idx-1, "sharpe"]
            c1, s1 = res_df.loc[idx, "annual_cost"], res_df.loc[idx, "sharpe"]
            max_cost = c0 + (threshold_sharpe - s0) * (c1 - c0) / (s1 - s0)
            print(f">>> 测算结论: 能够维持 Sharpe >= {threshold_sharpe} 的最大年化对冲成本上限为: {max_cost:.2%}")
    else:
        max_cost = res_df["annual_cost"].max()
        print(f"在测试的最高成本 {max_cost:.2%} 下，夏普比率仍保持在 0.5 以上。")

    # 导出敏感度分析结果
    res_df.to_csv(os.path.join(OUTPUT_DIR, "hedging_cost_sensitivity.csv"), index=False)
    
    # 绘制敏感度分析图
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = '#1f77b4'
    ax1.set_xlabel('年化做空/融券摩擦成本 (Annualized Shorting Cost)', fontsize=12)
    ax1.set_ylabel('年化收益率 (CAGR)', color=color, fontsize=12)
    line1 = ax1.plot(res_df['annual_cost'], res_df['cagr'], color=color, marker='o', label='CAGR')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2 = ax1.twinx()  
    color = '#ff7f0e'
    ax2.set_ylabel('夏普比率 (Sharpe Ratio)', color=color, fontsize=12)
    line2 = ax2.plot(res_df['annual_cost'], res_df['sharpe'], color=color, marker='s', label='Sharpe Ratio')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 标出 0.5 Sharpe 临界点线
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Sharpe = 0.5 警戒线')
    if max_cost > 0:
        ax1.axvline(x=max_cost, color='g', linestyle='-.', alpha=0.7, label=f'最大成本上限 ({max_cost:.2%})')
    
    # 合并图例
    lines = line1 + line2 + [plt.Line2D([0], [0], color='r', linestyle='--'), plt.Line2D([0], [0], color='g', linestyle='-.')]
    labels = ['CAGR', 'Sharpe Ratio', 'Sharpe = 0.5 警戒线', f'最大成本上限 ({max_cost:.2%})']
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title('做空对冲成本对组合绩效的敏感度分析 (2023 - 2025)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, "hedging_cost_sensitivity.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print(f"分析结果图表已保存至 {fig_path}")

if __name__ == "__main__":
    run_sensitivity_analysis()
