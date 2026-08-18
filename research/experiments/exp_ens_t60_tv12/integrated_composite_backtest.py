# -*- coding: utf-8 -*-
"""综合复合策略一体化回测引擎与全景图绘制 (Integrated Composite Strategy Engine & Visual Dashboard)

整合量化核心研究成果 (最新修复版):
  1. 选股层 (Alpha Engine): 全市场 5,800+ 股票池 + ENS 混合打分 (0.5×ENH4 + 0.5×C8-GBDT 残差筹码)
  2. 行业与分散化约束 (Portfolio Constraints): Top40 / 细分行业<=4 / 单申万一级<=20%
  3. 宏观风控层 (Multi-Tier Risk Control):
     - S123 宏观估值三档平滑 (>=3 → 1.0, ==2 → 0.5, <=1 → 0.0)，避险资金停泊于 V8 多资产稳健收益池
     - 组合净值熔断硬降档 (dd_degrade = -10%, scale = 0.5)，自高点回撤达 10% 仓位减半截断左尾
  4. 对冲层 (IM Futures Dynamic Hedging): 基于真实 2023-2026 滚仓基差日收益数据 (im_basis_analysis.csv) 的低成本空头对冲
  5. 交易与清算层 (Execution & Settlement): 遵从 A 股 T+1、双边 20bps 交易成本、涨跌停保护及 LAST_CLOSE 退市/停牌清算口径
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from engine import init_shared, run_backtest_tiered, SQRT_242  # noqa: E402

IDX_DIR = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily")
OUT_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
CONCLUSION_STOCK_DIR = os.path.join(ROOT, "quant_conclusion", "STOCK")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_index_close(code):
    fp = os.path.join(IDX_DIR, f"{code}.parquet")
    if not os.path.exists(fp):
        return pd.Series(dtype=float)
    df = pd.read_parquet(fp, columns=["trade_date", "close"])
    df["trade_date"] = df["trade_date"].astype(int)
    df = df.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")
    return df["close"].astype(float)


def calc_metrics(nav_s):
    nav_s = nav_s.sort_index().astype(float)
    nav_norm = nav_s / nav_s.iloc[0]
    tot = nav_norm.iloc[-1] - 1.0
    yrs = len(nav_norm) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    
    dd_s = nav_norm / nav_norm.cummax() - 1.0
    maxdd_d = float(dd_s.min())
    
    ret = nav_norm.pct_change().fillna(0.0)
    vol = float(ret.std() * SQRT_242)
    sharpe = float((ann - 0.02) / (vol + 1e-8)) if vol > 0 else 0.0
    
    nav_m = nav_norm.groupby((nav_norm.index // 100).astype(str)).last()
    dd_m = nav_m / nav_m.cummax() - 1.0
    maxdd_m = float(dd_m.min())
    calmar = float(ann / (-maxdd_d + 1e-9))
    
    ret_m = nav_m.pct_change().dropna()
    m_win = float((ret_m > 0).mean()) if len(ret_m) > 0 else 0.0
    
    return {
        "cagr": ann,
        "vol": vol,
        "sharpe": sharpe,
        "maxdd_d": maxdd_d,
        "maxdd_m": maxdd_m,
        "calmar": calmar,
        "m_win": m_win,
        "tot": tot,
        "n_days": len(nav_norm)
    }


def yearly_breakdown(nav_s):
    out = {}
    for y, g in nav_s.groupby(nav_s.index // 10000):
        out[int(y)] = float(g.iloc[-1] / g.iloc[0] - 1.0)
    return out


def to_datetime_index(idx):
    return pd.to_datetime(idx.astype(str), format="%Y%m%d")


def main():
    t0 = time.time()
    print("=" * 100)
    print(">>> 启动综合复合量化策略一体化回测系统 (Integrated Composite Strategy Engine)...")
    print("=" * 100)

    print("\n[Step 1/5] 加载全市场数据与预计算 GBDT 滚动模型打分...")
    sh = init_shared("fullmarket")
    print(f"    -> 成功加载面板: {len(sh['panel']):,} 行, 交易日: {sh['cal_dates'][0]} ~ {sh['cal_dates'][-1]}")

    print("\n[Step 2/5] 回测各层策略变体 (Alpha 引擎 + 逐级叠加风控)...")

    # 1. 传统 S123 二元开关 (Binary S123)
    nav_binary, _ = run_backtest_tiered(
        sh, "ENS", "T40", tgt_vol=None, timing_mode="binary",
        dd_degrade=None
    )
    
    # 2. 推荐风控 1: S123 三档平滑 (Tiered: 1.0/0.5/0)
    nav_tiered, _, w_tiered = run_backtest_tiered(
        sh, "ENS", "T40", tgt_vol=None, timing_mode="tiered",
        dd_degrade=None, return_exposure=True
    )
    
    # 3. 核心推荐风控 2: S123 三档 + 组合净值回撤降档 (Tiered + DD Degrade -10%x0.5) [选股端最优基线]
    nav_best_stock, _, w_best_stock = run_backtest_tiered(
        sh, "ENS", "T40", tgt_vol=None, timing_mode="tiered",
        dd_degrade=-0.10, dd_degrade_scale=0.5, return_exposure=True
    )
    
    # 4. 强防御变体: S123 三档 + MA20 趋势 + 回撤降档 (Tiered_MA20 + DD Degrade)
    nav_s123_ma20_dd, _, w_s123_ma20_dd = run_backtest_tiered(
        sh, "ENS", "T40", tgt_vol=None, timing_mode="s123_ma20",
        dd_degrade=-0.10, dd_degrade_scale=0.5, return_exposure=True
    )

    print("\n[Step 3/5] 加载真实期货基差与评估 IM 空头对冲层...")
    basis_fp = os.path.join(ROOT, "im_basis_analysis.csv")
    if os.path.exists(basis_fp):
        basis_df = pd.read_csv(basis_fp)
        basis_df["date"] = basis_df["date"].str.replace("-", "").astype(int)
        fut_ret = basis_df.set_index("date")["fut_ret"].dropna()
    else:
        print("    [警告] 未找到 im_basis_analysis.csv, 期货对冲评估跳过")
        fut_ret = pd.Series(dtype=float)

    # 在 IM 覆盖区间 (2023 至今, 真正的 OOS 阶段) 对当前最优股票策略叠加 IM 对冲
    hedged_strategies = {}
    if len(fut_ret) > 0:
        lo = fut_ret.index.min()
        nav_sub = nav_best_stock[nav_best_stock.index >= lo]
        w_sub = w_best_stock[w_best_stock.index >= lo]
        r_p = nav_sub.pct_change().fillna(0.0)
        w_prev = w_sub.shift(1).fillna(0.0)
        fut_a = fut_ret.reindex(nav_sub.index).fillna(0.0)

        for beta in [0.3, 0.5, 0.7, 1.0]:
            # 做空 beta * w_{t-1} 份 IM 期货 (每日结算低基差收益)
            r_h = r_p - beta * w_prev * fut_a
            nh = (1.0 + r_h).cumprod()
            hedged_strategies[f"IM对冲(β={beta:.1f})"] = nh

    print("\n[Step 4/5] 加载权威基准行情并对齐...")
    c1000 = load_index_close("000852.SH")
    c300 = load_index_close("000300.SH")
    
    cal_idx = nav_best_stock.index
    c1000_aligned = c1000.reindex(cal_idx).ffill().bfill()
    c300_aligned = c300.reindex(cal_idx).ffill().bfill()
    b1000 = c1000_aligned / c1000_aligned.iloc[0]
    b300 = c300_aligned / c300_aligned.iloc[0]

    all_strats = {
        "基准: 中证1000": b1000,
        "基准: 沪深300": b300,
        "1. 二元开关 (Binary S123)": nav_binary,
        "2. 三档梯度 (Tiered S123)": nav_tiered,
        "3. 核心推荐 (Tiered+降档-10%)": nav_best_stock,
        "4. 强防御 (Tiered+MA20+降档)": nav_s123_ma20_dd
    }

    print("\n" + "=" * 115)
    print("                   全样本策略综合绩效对比表 (Full-Sample Strategy Comparison: 2019-2026)")
    print("=" * 115)
    hdr = f"{'策略配置 / Strategy':<32} | {'CAGR':>8} | {'夏普/Sharpe':>11} | {'日MaxDD':>9} | {'月MaxDD':>9} | {'卡玛/Calmar':>11} | {'月胜率':>7}"
    print(hdr)
    print("-" * 115)
    
    metrics_summary = {}
    for name, s in all_strats.items():
        m = calc_metrics(s)
        metrics_summary[name] = m
        print(f"{name:<32} | {m['cagr']:>7.2%} | {m['sharpe']:>11.2f} | {m['maxdd_d']:>9.2%} | {m['maxdd_m']:>9.2%} | {m['calmar']:>11.2f} | {m['m_win']:>6.1%}")
    print("-" * 115)

    # 分年度收益表
    years = sorted(list(set(nav_best_stock.index // 10000)))
    print("\n" + "=" * 115)
    print("                             分年度收益率明细表 (Annual Return Breakdown)")
    print("=" * 115)
    yr_hdr = f"{'策略 / 年份':<30} | " + " | ".join(f"{y:>7}" for y in years) + " | " + f"{'总收益':>9}"
    print(yr_hdr)
    print("-" * 115)
    
    for name, s in all_strats.items():
        yb = yearly_breakdown(s)
        tot_ret = s.iloc[-1] / s.iloc[0] - 1.0
        yr_strs = [f"{yb.get(y, float('nan')):>7.1%}" for y in years]
        print(f"{name:<30} | " + " | ".join(yr_strs) + f" | {tot_ret:>9.1%}")
    print("-" * 115)

    # 阶段 3 详细 IM 对冲结果 (2023-2026 真正 OOS 样本)
    if hedged_strategies:
        print("\n" + "=" * 110)
        print("         IM 股指期货低基差对冲深度实测表 (IM Futures Hedging in OOS Period: 2023-2026)")
        print("=" * 110)
        h_hdr = f"{'对冲配置':<24} | {'CAGR (2023+)':>13} | {'夏普 / Sharpe':>13} | {'日MaxDD':>9} | {'月MaxDD':>9} | {'卡玛 / Calmar':>13}"
        print(h_hdr)
        print("-" * 110)
        
        sub_best = nav_best_stock[nav_best_stock.index >= fut_ret.index.min()]
        sub_1000 = b1000[b1000.index >= fut_ret.index.min()]
        sub_300 = b300[b300.index >= fut_ret.index.min()]
        
        for n, s in [("核心股票策略(无对冲)", sub_best), ("中证1000指数基准", sub_1000), ("沪深300指数基准", sub_300)]:
            m = calc_metrics(s)
            print(f"{n:<24} | {m['cagr']:>13.2%} | {m['sharpe']:>13.2f} | {m['maxdd_d']:>9.2%} | {m['maxdd_m']:>9.2%} | {m['calmar']:>13.2f}")
        print("-" * 110)
        
        for n, s in hedged_strategies.items():
            m = calc_metrics(s)
            print(f"{n:<24} | {m['cagr']:>13.2%} | {m['sharpe']:>13.2f} | {m['maxdd_d']:>9.2%} | {m['maxdd_m']:>9.2%} | {m['calmar']:>13.2f}")
        print("-" * 110)

    print("\n[Step 5/5] 绘制专业三栏全景收益与对冲对比仪表盘 (3-Panel Master Dashboard)...")
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.0, 1.6], hspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0])

    dt_idx = to_datetime_index(nav_best_stock.index)

    # 1. 历史全景净值曲线 (2019-2026)
    ax1.plot(dt_idx, nav_best_stock / nav_best_stock.iloc[0], label=f"★ 核心推荐: 全市场ENS+三档S123+净值降档 (年化 {metrics_summary['3. 核心推荐 (Tiered+降档-10%)']['cagr']:.1%}, 夏普 {metrics_summary['3. 核心推荐 (Tiered+降档-10%)']['sharpe']:.2f}, 月MaxDD {metrics_summary['3. 核心推荐 (Tiered+降档-10%)']['maxdd_m']:.1%})", color="#c0392b", linewidth=2.2, zorder=5)
    ax1.plot(dt_idx, nav_tiered / nav_tiered.iloc[0], label=f"变体 1: Tiered三档择时 (无降档) (年化 {metrics_summary['2. 三档梯度 (Tiered S123)']['cagr']:.1%}, 夏普 {metrics_summary['2. 三档梯度 (Tiered S123)']['sharpe']:.2f})", color="#e67e22", linewidth=1.3, linestyle="--", alpha=0.9)
    ax1.plot(dt_idx, nav_binary / nav_binary.iloc[0], label=f"变体 2: 传统二元开关 (Binary S123) (年化 {metrics_summary['1. 二元开关 (Binary S123)']['cagr']:.1%}, 夏普 {metrics_summary['1. 二元开关 (Binary S123)']['sharpe']:.2f})", color="#8e44ad", linewidth=1.1, linestyle=":", alpha=0.8)
    ax1.plot(dt_idx, nav_s123_ma20_dd / nav_s123_ma20_dd.iloc[0], label=f"变体 3: 强防御版 (Tiered+MA20+降档) (年化 {metrics_summary['4. 强防御 (Tiered+MA20+降档)']['cagr']:.1%}, 日MaxDD {metrics_summary['4. 强防御 (Tiered+MA20+降档)']['maxdd_d']:.1%})", color="#27ae60", linewidth=1.2, alpha=0.85)
    ax1.plot(dt_idx, b1000 / b1000.iloc[0], label=f"基准: 中证1000指数 (000852.SH) (年化 {metrics_summary['基准: 中证1000']['cagr']:.1%}, 日MaxDD {metrics_summary['基准: 中证1000']['maxdd_d']:.1%})", color="#2980b9", linewidth=1.1, alpha=0.75)
    ax1.plot(dt_idx, b300 / b300.iloc[0], label=f"基准: 沪深300指数 (000300.SH) (年化 {metrics_summary['基准: 沪深300']['cagr']:.1%}, 日MaxDD {metrics_summary['基准: 沪深300']['maxdd_d']:.1%})", color="#7f8c8d", linewidth=1.1, alpha=0.75)

    ax1.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax1.set_title("综合量化策略全历史累计净值对比 (2019-06 ~ 2026-08)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("累计净值 (Normalized NAV)", fontsize=10.5)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=9.0)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # 2. 全历史动态回撤曲线 (2019-2026)
    def get_dd_series(s):
        ns = s / s.iloc[0]
        return ns / ns.cummax() - 1.0

    dd_best = get_dd_series(nav_best_stock)
    dd_1000 = get_dd_series(b1000)

    ax2.plot(dt_idx, dd_best, label=f"核心推荐策略回撤 (日频MaxDD {metrics_summary['3. 核心推荐 (Tiered+降档-10%)']['maxdd_d']:.1%}, 月频MaxDD {metrics_summary['3. 核心推荐 (Tiered+降档-10%)']['maxdd_m']:.1%})", color="#c0392b", linewidth=1.5)
    ax2.plot(dt_idx, dd_1000, label=f"中证1000指数回撤 (MaxDD {metrics_summary['基准: 中证1000']['maxdd_d']:.1%})", color="#2980b9", linewidth=1.0, alpha=0.7)
    ax2.fill_between(dt_idx, dd_best, 0, color="#c0392b", alpha=0.15)
    ax2.axhline(-0.10, color="#d35400", linestyle="--", linewidth=1.0, label="10% 熔断降档线 (DD Degrade -10%×0.5)")
    ax2.axhline(-0.20, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("动态回撤 (Drawdown)", fontsize=10.5)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.35)

    # 3. 2023-2026 样本外 IM 期货低基差对冲深度实测
    if hedged_strategies:
        lo_date = fut_ret.index.min()
        dt_sub = to_datetime_index(nav_best_stock[nav_best_stock.index >= lo_date].index)
        sub_best_norm = nav_best_stock[nav_best_stock.index >= lo_date] / nav_best_stock[nav_best_stock.index >= lo_date].iloc[0]
        sub_1000_norm = b1000[b1000.index >= lo_date] / b1000[b1000.index >= lo_date].iloc[0]
        
        m_unh = calc_metrics(sub_best_norm)
        m_h05 = calc_metrics(hedged_strategies["IM对冲(β=0.5)"])
        m_h07 = calc_metrics(hedged_strategies["IM对冲(β=0.7)"])
        m_1000 = calc_metrics(sub_1000_norm)

        ax3.plot(dt_sub, hedged_strategies["IM对冲(β=0.5)"], label=f"★ IM对冲(β=0.5) 最佳平衡点 (年化 {m_h05['cagr']:.1%}, 夏普 {m_h05['sharpe']:.2f}, 日MaxDD {m_h05['maxdd_d']:.1%}, 月MaxDD {m_h05['maxdd_m']:.1%}, 卡玛 {m_h05['calmar']:.2f})", color="#2c3e50", linewidth=2.3, zorder=6)
        ax3.plot(dt_sub, hedged_strategies["IM对冲(β=0.7)"], label=f"IM对冲(β=0.7) 稳健点 (年化 {m_h07['cagr']:.1%}, 夏普 {m_h07['sharpe']:.2f}, 日MaxDD {m_h07['maxdd_d']:.1%})", color="#16a085", linewidth=1.2, linestyle="--")
        ax3.plot(dt_sub, sub_best_norm, label=f"核心股票策略 (无对冲) (年化 {m_unh['cagr']:.1%}, 夏普 {m_unh['sharpe']:.2f}, 日MaxDD {m_unh['maxdd_d']:.1%})", color="#c0392b", linewidth=1.3, alpha=0.85)
        ax3.plot(dt_sub, sub_1000_norm, label=f"中证1000指数基准 (年化 {m_1000['cagr']:.1%}, 夏普 {m_1000['sharpe']:.2f}, 日MaxDD {m_1000['maxdd_d']:.1%})", color="#2980b9", linewidth=1.1, alpha=0.7)

        ax3.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
        ax3.set_title("真实 IM 股指期货低基差对冲实测 (2023 - 2026 严格样本外 OOS 阶段)", fontsize=13, fontweight="bold", pad=10)
        ax3.set_ylabel("样本外净值 (OOS NAV)", fontsize=10.5)
        ax3.set_xlabel("交易日期 (Date)", fontsize=10.5)
        ax3.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=9.0)
        ax3.grid(True, linestyle="--", alpha=0.35)

    out_img = os.path.join(OUT_DIR, "integrated_composite_nav.png")
    fig.savefig(out_img, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> 综合策略全景高清图已保存: {out_img}")

    # 同时复制到 quant_conclusion 仓库和脑图目录
    import shutil
    shutil.copy(out_img, os.path.join(CONCLUSION_STOCK_DIR, "integrated_composite_nav.png"))
    shutil.copy(out_img, os.path.join(r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0", "integrated_composite_nav.png"))
    print("    -> 图像已同步至 quant_conclusion/STOCK 与 Brain 目录。")

    print(f"\n[完成] 一体化回测执行完毕, 总耗时: {time.time()-t0:.1f} 秒。")


if __name__ == "__main__":
    main()
