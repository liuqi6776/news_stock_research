# -*- coding: utf-8 -*-
"""
最终方案收益曲线: 9资产组合 + VolTarget7% (floor=0.5)
==========================================================
输出:
  results/final_curve.png  (净值曲线 + 回撤曲线, 一次性100万 + 定投场景)
用法:
  C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/final_curve.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vol_target as vt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

def draw_curve(eq, title, path, dca_days=None, lump=1_000_000, dca=0):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    # 净值曲线
    ax = axes[0]
    nav = eq / eq.iloc[0]
    ax.plot(nav.index, nav.values, lw=1.6, color="#1663b3")
    ax.set_ylabel("净值(起始=1.0)")
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3)
    # 标注期末
    ax.annotate(f"期末净值 {nav.iloc[-1]:.2f}", xy=(nav.index[-1], nav.iloc[-1]),
                xytext=(-120, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#333"), fontsize=10)
    # 回撤曲线
    ax2 = axes[1]
    dd = eq / eq.cummax() - 1
    ax2.fill_between(dd.index, dd.values * 100, 0, color="#d33", alpha=0.45)
    ax2.set_ylabel("回撤 (%)")
    ax2.grid(alpha=0.3)
    mdd = dd.min()
    ax2.annotate(f"最大回撤 {mdd:.1%}", xy=(dd.idxmin(), mdd * 100),
                 xytext=(20, -30), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="#333"), fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()

def main():
    navs = vt.load_navs()
    weights = {n: w for n, (_, w) in vt.ASSETS.items()}
    TGT, FLOOR = 0.07, 0.50

    # 场景1: 一次性100万
    eq1, _ = vt.run_backtest(navs, weights, tgt_vol=TGT, floor_w=FLOOR,
                             lump=1_000_000, dca=0)
    m1 = vt.calc_metrics(eq1, 1_000_000)
    draw_curve(eq1, f"场景A: 一次性100万 | 年化{m1['年化']:.1%} 回撤{m1['回撤']:.1%} 夏普{m1['夏普']:.2f}",
               os.path.join(OUT_DIR, "final_curve_lump.png"))

    # 场景2: 一次性100万 + 月定投1万
    eq2, dca_days = vt.run_backtest(navs, weights, tgt_vol=TGT, floor_w=FLOOR,
                                    lump=1_000_000, dca=10_000)
    total_in = 1_000_000 + 10_000 * len(dca_days)
    m2 = vt.calc_metrics(eq2, 1_000_000)
    draw_curve(eq2, f"场景B: 一次性100万+月定投1万 | 期末{m2['期末']/1e4:.0f}万 总收益{m2['期末']/total_in-1:.1%} 回撤{m2['回撤']:.1%}",
               os.path.join(OUT_DIR, "final_curve_dca.png"), dca_days)

    # 场景3: OOS区间 一次性100万
    eq3, _ = vt.run_backtest(navs, weights, tgt_vol=TGT, floor_w=FLOOR,
                             lump=1_000_000, dca=0, start=vt.OOS_START, end=vt.END)
    m3 = vt.calc_metrics(eq3, 1_000_000)
    draw_curve(eq3, f"OOS区间(2023-2026) 一次性100万 | 年化{m3['年化']:.1%} 回撤{m3['回撤']:.1%} 夏普{m3['夏普']:.2f}",
               os.path.join(OUT_DIR, "final_curve_oos.png"))

    # 逐年收益 (场景A)
    print("=" * 90)
    print("最终方案: 9资产 + VolTarget7% (floor=0.5)")
    print("=" * 90)
    yr = eq1.resample("Y").last().pct_change().dropna()
    print(f"场景A(一次性100万) 逐年收益:")
    print("  " + "  ".join(f"{y.year}:{v:+.1%}" for y, v in yr.items()))
    print(f"  全样本: 年化{m1['年化']:.1%} 回撤{m1['回撤']:.1%} 波动{m1['波动']:.1%} 夏普{m1['夏普']:.2f} 期末{m1['期末']/1e4:.0f}万")
    print(f"  OOS(2023-2026): 年化{m3['年化']:.1%} 回撤{m3['回撤']:.1%} 夏普{m3['夏普']:.2f} 期末{m3['期末']/1e4:.0f}万")
    print(f"\n场景B(一次性+定投) : 期末{m2['期末']/1e4:.0f}万 总收益{m2['期末']/total_in-1:.1%} 回撤{m2['回撤']:.1%} (投入{total_in/1e4:.0f}万)")
    print(f"\n[图] {OUT_DIR}/final_curve_lump.png, final_curve_dca.png, final_curve_oos.png")

if __name__ == "__main__":
    main()
