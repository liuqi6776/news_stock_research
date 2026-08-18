# -*- coding: utf-8 -*-
"""
4433 轮换定投 - 参数敏感性分析 (对比月度 / 季度 / 半年度轮换)
================================================================

在**年度定投总额固定**的前提下, 对比轮换频率对收益/回撤/换手成本的影响:
  - 月度   每期 1000 元 (年度 12000)
  - 季度   每期 3000 元 (年度 12000)
  - 半年度 每期 6000 元 (年度 12000)

每个频率跑 4 个策略 (与 run_rotation.py 相同口径):
  A. 动态4433-全仓再平衡   (主策略)
  B. 动态4433-只轮新资金   (无赎回费损耗)
  C. 静态206-全仓再平衡    (幸存者偏差对照)
  D. 沪深300联接-定投      (市场基准)

复用 run_rotation.py 的选基/组合/净值曲线函数, 只替换轮换日期序列与每期金额。

用法:
  C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/sensitivity.py \
    --start 2021-01-01 --end 2026-08-06 --annual 12000
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fund_research
sys.path.insert(0, ROOT)
import run_rotation as rr  # noqa: E402

# 频率配置: (显示名, pandas freq, 每期=年度/12*mult)
FREQS = [
    ("月度", "M", 1),
    ("季度", "Q", 3),
    ("半年度", "2Q", 6),
]

STRATEGIES = ["动态4433-全仓再平衡", "动态4433-只轮新资金",
              "静态206-全仓再平衡", "沪深300联接-定投"]


def main():
    ap = argparse.ArgumentParser(description="4433 轮换频率敏感性分析")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-08-06")
    ap.add_argument("--annual", type=float, default=12000, help="年度定投总额")
    ap.add_argument("--rebuild-panel", action="store_true", help="强制重建窗口收益缓存")
    args = ap.parse_args()

    t0 = time.time()
    basic = rr.load_basic()
    active = basic[basic["fund_type"].isin(rr.ACTIVE_TYPES)]
    codes = active["code"].astype(str).tolist()
    basic_ft = basic.set_index("code")["fund_type"]
    print(f"动态池: {len(codes)} 只主动权益基金")

    # 静态 206 池 (2026 年 4433 结果)
    static206 = []
    if os.path.exists(rr.STATIC_CSV):
        csv = pd.read_csv(rr.STATIC_CSV, dtype={"code": str})
        csv = csv[csv["is_4433"] == True]
        static206 = [str(int(c)).zfill(6) for c in csv["code"]]
    print(f"静态池: {len(static206)} 只")

    # 指数基准
    idx_code = None
    for c in rr.INDEX_CANDIDATES:
        if os.path.exists(os.path.join(rr.NAV_DIR, f"{c}.parquet")):
            idx_code = c
            break
    print(f"指数基准: {idx_code or '无'}")

    rows = []
    curves = {}          # "频率:策略" -> 净值 Series
    sel_stats = []       # 每频率每期入选数
    for freq_name, freq, mult in FREQS:
        rdates = [pd.Timestamp(d) for d in pd.date_range(args.start, args.end, freq=freq)]
        per = args.annual / 12.0 * mult  # 每期金额, 保证年度总投入一致
        rr.CONTRIBUTION = per
        rr.END = pd.Timestamp(args.end)
        n_sel = []
        print(f"\n=== {freq_name}轮换: {len(rdates)} 期 x {per:.0f} 元 (年度 {args.annual:.0f}) ===")

        out, union = rr.compute_window_sums(codes, rdates, rebuild=args.rebuild_panel)
        sel_dyn = [rr.select_4433(out, codes, basic_ft, i) for i in range(len(rdates))]
        n_sel = [len(s) for s in sel_dyn]
        print(f"  平均每期通过4433: {np.mean(n_sel):.0f} 只")

        eqA, heldA, turnA = rr.run_strategy(STRATEGIES[0], rdates, sel_dyn, "full", rr.Portfolio(), union)
        eqB, heldB, _ = rr.run_strategy(STRATEGIES[1], rdates, sel_dyn, "new", rr.Portfolio(), union)
        sel_static = [static206] * len(rdates)
        eqC, heldC, turnC = rr.run_strategy(STRATEGIES[2], rdates, sel_static, "full", rr.Portfolio(), union)
        sel_idx = [[idx_code] for _ in rdates] if idx_code else [[]] * len(rdates)
        eqD, heldD, _ = rr.run_strategy(STRATEGIES[3], rdates, sel_idx, "new", rr.Portfolio(), union)

        for name, eq, held, turn in [
            (STRATEGIES[0], eqA, heldA, turnA),
            (STRATEGIES[1], eqB, heldB, None),
            (STRATEGIES[2], eqC, heldC, turnC),
            (STRATEGIES[3], eqD, heldD, None),
        ]:
            r = rr.summarize(f"{freq_name}:{name}", eq, per, rdates)
            r["freq"] = freq_name
            r["n_periods"] = len(rdates)
            r["per_amount"] = per
            r["avg_held"] = round(float(np.mean(held)), 1) if held else np.nan
            r["avg_turnover_pct"] = round(float(np.mean(turn)) * 100, 1) if turn is not None else np.nan
            rows.append(r)
            curves[f"{freq_name}:{name}"] = eq

        for i, d in enumerate(rdates):
            sel_stats.append({"freq": freq_name, "date": d.date(), "n_selected": n_sel[i]})

    summary = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print("\n" + "=" * 120)
    print(f"4433 轮换频率敏感性分析 ({args.start} ~ {args.end}, 年度定投 {args.annual:.0f} 元, "
          f"申购费 {rr.SG_FEE:.2f}%, 赎回费按持有期阶梯)")
    print("=" * 120)
    show = summary[["freq", "strategy", "total_invested", "final_value", "total_return_pct",
                    "xirr_pct", "mdd_pct", "avg_held", "avg_turnover_pct"]]
    print(show.to_string(index=False))
    print("=" * 120)

    # 导出
    summary_out = os.path.join(rr.RESULTS_DIR, "rotation_sensitivity_summary.csv")
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")
    print(f"汇总表: {summary_out}")

    sel_df = pd.DataFrame(sel_stats)
    sel_df.to_csv(os.path.join(rr.RESULTS_DIR, "rotation_sensitivity_selection.csv"),
                  index=False, encoding="utf-8-sig")

    # 画图: 每策略一个子图, 3 条频率曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        for ax, strat in zip(axes.flat, STRATEGIES):
            for freq_name, _, _ in FREQS:
                eq = curves.get(f"{freq_name}:{strat}")
                if eq is not None and not eq.empty:
                    ax.plot(eq.index, eq.values, label=freq_name, linewidth=1.6)
            ax.set_title(strat)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.suptitle(f"4433 轮换频率敏感性 ({args.start} ~ {args.end}, 年度定投 {args.annual:.0f} 元)",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        png = os.path.join(rr.RESULTS_DIR, "rotation_sensitivity.png")
        fig.savefig(png, dpi=130)
        print(f"对比图: {png}")
    except Exception as e:
        print(f"画图跳过: {e}")

    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
