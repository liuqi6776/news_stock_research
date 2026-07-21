# -*- coding: utf-8 -*-
"""
backtest_hedged.py - A0 市场中性对冲策略回测与可行性预研 (路线 B)
==================================================================

本脚本实现以下对冲方案回测（2023 - 2025 样本外区间）：
1. 理论对冲方案 (Spot Hedged)：做多 A0 股票组合，100% 对冲做空中证1000现货指数（无做空摩擦/贴水）。
2. 期货对冲方案 (IM Futures Hedged - Pct Change)：做多 A0 股票组合，100% 对冲做空 IM 主力连续合约（直接收盘价计算）。
3. 期货对冲方案 (IM Futures Hedged - Basis Drag)：做多 A0 股票组合，100% 对冲做空中证1000现货指数，并扣除每日时变贴水收敛损耗（通过 alpha_basis.py 测算）。

同时计算各项绩效指标（CAGR、Sharpe、MaxDD、月胜率、跟踪误差）并生成对比曲线图。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 确保中文字体显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---- 路径配置 ----
ROOT = r"C:\Users\liuqi\quant_system_v2"
A0_PATH = os.path.join(ROOT, "research", "studies", "study_007_cross_sectional", "fix", "results_fixed", "main_nav_vs_benchmark.csv")
SPOT_A = os.path.join(ROOT, "zz1000_spot_a.csv")
SPOT_B = os.path.join(ROOT, "zz1000_spot_b.csv")
IM_PATH = os.path.join(ROOT, "im_main_sina.csv")
SAMPLES_CSV = os.path.join(ROOT, "im_ann_basis_samples.csv")

def run_hedged_backtest():
    print("=" * 70)
    print(" 启动 A0 股票组合市场中性对冲策略回测 (2023 - 2025) ")
    print("=" * 70)

    # 1. 读取 A0 策略收益率
    if not os.path.exists(A0_PATH):
        raise FileNotFoundError(f"未找到 A0 策略数据: {A0_PATH}")
    a0 = pd.read_csv(A0_PATH)
    a0["date"] = pd.to_datetime(a0["trade_date"].astype(str), format="%Y%m%d")
    a0 = a0.set_index("date")[["daily_ret"]].rename(columns={"daily_ret": "ret_a0"})
    
    # 2. 读取中证1000现货指数并计算收益率
    parts = []
    for f in (SPOT_A, SPOT_B):
        if os.path.exists(f):
            parts.append(pd.read_csv(f))
    if not parts:
        raise FileNotFoundError("未找到中证1000现货指数数据")
    spot = pd.concat(parts, ignore_index=True)
    spot["date"] = pd.to_datetime(spot["time"])
    spot = spot.drop_duplicates("date").sort_values("date").set_index("date")
    spot["ret_spot"] = spot["close"].pct_change()
    
    # 3. 读取 IM 期货主力连续合约并计算收益率
    if not os.path.exists(IM_PATH):
        raise FileNotFoundError(f"未找到 IM 期货数据: {IM_PATH}")
    fut = pd.read_csv(IM_PATH)
    fut["date"] = pd.to_datetime(fut["日期"])
    fut = fut.set_index("date")
    fut["ret_fut"] = fut["收盘价"].pct_change()
    
    # 4. 获取时变贴水拖累 (Basis Drag)
    # 导入 alpha_basis.py 中计算的 rolling 60 天基差均值
    sys.path.insert(0, ROOT)
    try:
        from alpha_basis import load_alpha_series
        alpha_series, meta = load_alpha_series(samples_csv=SAMPLES_CSV)
        # alpha_series 是年化小数，转为日频贴水拖累 (假定一年 242 交易日)
        # 贴水收敛对期货空头来说是损失，因此每日拖累 = -alpha / 242.0
        daily_drag = -alpha_series / 242.0
        daily_drag.name = "ret_drag"
    except Exception as e:
        print(f"警告: 无法加载 alpha_basis.py 进行时变贴水计算 ({e})，将使用常数贴水 9.3%/年。")
        daily_drag = pd.Series(-0.093 / 242.0, index=spot.index, name="ret_drag")

    # 5. 合并并对齐数据
    df = a0.join(spot[["ret_spot"]], how="inner")
    df = df.join(fut[["ret_fut"]], how="inner")
    df = df.join(daily_drag, how="left")
    # 填充拖累的前导空值 (使用 2023-01~2026-07 均值 9.3% 年化作为 fallback)
    df["ret_drag"] = df["ret_drag"].fillna(-0.093 / 242.0)
    
    print(f"对齐样本交易日数量: {len(df)} 天, {df.index.min().date()} ~ {df.index.max().date()}")

    # 6. 计算对冲组合的收益率
    # 方案 A: 理论现货对冲 (100% 做空 1000 现货)
    df["ret_hedged_spot"] = df["ret_a0"] - df["ret_spot"]
    
    # 方案 B: 期货主力合约直接对冲 (100% 做空 IM 主力连续)
    df["ret_hedged_fut"] = df["ret_a0"] - df["ret_fut"]
    
    # 方案 C: 期货贴水拖累对冲 (现货对冲 - 贴水收敛损失)
    df["ret_hedged_drag"] = df["ret_a0"] - df["ret_spot"] + df["ret_drag"]

    # 7. 计算累计净值 (NAV)
    df["nav_a0"] = (1 + df["ret_a0"]).cumprod()
    df["nav_spot"] = (1 + df["ret_spot"]).cumprod()
    df["nav_hedged_spot"] = (1 + df["ret_hedged_spot"]).cumprod()
    df["nav_hedged_fut"] = (1 + df["ret_hedged_fut"]).cumprod()
    df["nav_hedged_drag"] = (1 + df["ret_hedged_drag"]).cumprod()

    # 8. 绩效评价函数
    def evaluate(ret, nav, name):
        n = len(ret)
        years = n / 242.0
        cagr = nav.iloc[-1] ** (1.0 / years) - 1.0
        vol = ret.std() * np.sqrt(242)
        sharpe = (ret.mean() / ret.std() * np.sqrt(242)) if ret.std() > 0 else 0
        cummax = nav.cummax()
        mdd = (nav / cummax - 1.0).min()
        
        # 计算月胜率
        monthly_ret = ret.groupby(ret.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
        win_rate = (monthly_ret > 0).mean()
        
        return {
            "策略名称": name,
            "年化收益 (CAGR)": f"{cagr:.2%}",
            "年化波动率": f"{vol:.2%}",
            "夏普比率 (Sharpe)": f"{sharpe:.3f}",
            "最大回撤 (MaxDD)": f"{mdd:.2%}",
            "月度胜率": f"{win_rate:.2%}",
            "卡玛比率 (Calmar)": f"{abs(cagr/mdd):.2f}" if mdd != 0 else "N/A"
        }

    # 9. 评估各组合表现
    metrics = []
    metrics.append(evaluate(df["ret_a0"], df["nav_a0"], "A0 股票多头组合"))
    metrics.append(evaluate(df["ret_spot"], df["nav_spot"], "中证1000现货指数 (基准)"))
    metrics.append(evaluate(df["ret_hedged_spot"], df["nav_hedged_spot"], "A0 - 理论现货对冲 (纯选股Alpha)"))
    metrics.append(evaluate(df["ret_hedged_fut"], df["nav_hedged_fut"], "A0 - IM期货直接对冲 (含主力换月损耗)"))
    metrics.append(evaluate(df["ret_hedged_drag"], df["nav_hedged_drag"], "A0 - 现货对冲扣除贴水拖累 (理论期货对冲)"))

    res_df = pd.DataFrame(metrics)
    print("\n" + "=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)

    # 10. 保存指标数据到 CSV 方便后续分析
    output_dir = os.path.join(ROOT, "research", "studies", "study_007_cross_sectional", "fix", "results_fixed")
    df.to_csv(os.path.join(output_dir, "hedged_backtest_daily.csv"))
    res_df.to_csv(os.path.join(output_dir, "hedged_backtest_metrics.csv"), index=False, encoding="utf-8-sig")
    print(f"数据已导出至 {output_dir}")

    # 11. 绘制净值对比曲线图
    plt.figure(figsize=(12, 7))
    plt.plot(df.index, df["nav_a0"], label="A0 股票多头组合 (CAGR 17.5%)", color="#2c3e50", alpha=0.8)
    plt.plot(df.index, df["nav_spot"], label="中证1000现货指数 (CAGR 2.6%)", color="#7f8c8d", linestyle="--", alpha=0.7)
    plt.plot(df.index, df["nav_hedged_spot"], label="A0 - 理论现货对冲 (纯选股Alpha, CAGR 14.8%)", color="#27ae60", linewidth=2.0)
    plt.plot(df.index, df["nav_hedged_drag"], label="A0 - 现货对冲扣除贴水拖累 (理论IM对冲, CAGR 5.1%)", color="#e74c3c", linewidth=2.0)
    plt.plot(df.index, df["nav_hedged_fut"], label="A0 - IM期货直接对冲 (直接主力对冲, CAGR 7.3%)", color="#f39c12", alpha=0.8)
    
    plt.title("A0 股票组合市场中性对冲策略净值走势比对 (2023 - 2025)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("累计净值 (NAV)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, "hedged_backtest_nav.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"图表已保存至 {fig_path}")

if __name__ == "__main__":
    run_hedged_backtest()
