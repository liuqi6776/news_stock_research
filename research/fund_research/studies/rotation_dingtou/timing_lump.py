# -*- coding: utf-8 -*-
"""
10 万资金安排的两种方式对比 (择时一次性买持 vs 季度定投只轮新资金)
================================================================

A 择时一次性买持: 10 万资金等待 4 信号首次进入低估区 (n_sig>=2, 2021-09-30)
                  一次性买入当季 4433 池等权, 此后拿住不动 (无轮换/无卖出)
B 季度定投-只轮新资金: 10 万分 22 期每季定投, 动态 4433 每季选基,
                  新资金只在当季池等权买入 (invest_new_money), 旧仓永不卖
C 对照-无择时一次性买持: 2021-03-31 起点一次性买入当季 4433 池拿住
                  (复现 lump_sum.py 动态买持 +31.77%)

指标: 期末市值 / 总收益 / XIRR / 最大回撤 (申购费 0.15%, 无赎回费摩擦: A/C 不卖, B 只买)

用法:
  C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/timing_lump.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fund_research
sys.path.insert(0, ROOT)
import run_rotation as rr  # noqa: E402
import timing_dingtou as td  # noqa: E402

START = "2021-01-01"
END = "2026-08-06"
TOTAL = 100000.0


def buy_and_hold(qdates, sel_dyn, union, i0, amount):
    """从第 i0 季度一次性买入当季 4433 池等权, 之后拿住不动. 返回净值 Series"""
    port = rr.Portfolio()
    rr.invest_new_money(port, sel_dyn[i0], qdates[i0], amount)
    eq_parts = []
    for i in range(i0, len(qdates)):
        d_end = qdates[i + 1] if i + 1 < len(qdates) else rr.END
        cash, shares = port.snapshot()
        seg = rr.segment_equity(cash, shares, qdates[i], d_end, union)
        if not seg.empty:
            eq_parts.append(seg)
    return pd.concat(eq_parts)


def quarterly_dca(qdates, sel_dyn, union, per):
    """每季度定投 per 元, 只轮新资金 (旧仓永不卖). 返回净值 Series"""
    port = rr.Portfolio()
    eq_parts = []
    for i, d in enumerate(qdates):
        rr.invest_new_money(port, sel_dyn[i], d, per)
        d_end = qdates[i + 1] if i + 1 < len(qdates) else rr.END
        cash, shares = port.snapshot()
        seg = rr.segment_equity(cash, shares, d, d_end, union)
        if not seg.empty:
            eq_parts.append(seg)
    return pd.concat(eq_parts)


def summarize(name, equity, cf):
    final = float(equity.iloc[-1])
    invested = sum(-a for _, a in cf if a < 0)
    irr = rr.xirr(cf) if len(cf) >= 2 else np.nan
    mdd = float((equity / equity.cummax() - 1.0).min())
    return {"strategy": name,
            "invested": round(invested, 2),
            "final_value": round(final, 2),
            "total_return_pct": round((final / invested - 1.0) * 100, 2),
            "xirr_pct": round(irr * 100, 2) if np.isfinite(irr) else np.nan,
            "mdd_pct": round(mdd * 100, 2)}


def main():
    basic = rr.load_basic()
    active = basic[basic["fund_type"].isin(rr.ACTIVE_TYPES)]
    codes = active["code"].astype(str).tolist()
    basic_ft = basic.set_index("code")["fund_type"]
    qdates = [pd.Timestamp(d) for d in pd.date_range(START, END, freq="Q")]
    rr.END = pd.Timestamp(END)
    print(f"动态池 {len(codes)} 只, 季度点 {len(qdates)} 个, 总资金 {TOTAL:.0f}")

    out, union = rr.compute_window_sums(codes, qdates)
    sel_dyn = [rr.select_4433(out, codes, basic_ft, i) for i in range(len(qdates))]
    sig = td.compute_signals(qdates, sel_dyn, out, codes)
    n_sig = sig["n_sig"].tolist()

    # 首个低估区季度 (n_sig >= 2)
    first_low = next(i for i, n in enumerate(n_sig) if n >= td.LOW_SIG)
    print(f"首个低估区季度: {qdates[first_low].date()} (n_sig={n_sig[first_low]})")

    results = []
    curves = {}

    # A 择时一次性买持
    eqA = buy_and_hold(qdates, sel_dyn, union, first_low, TOTAL)
    cfA = [(qdates[first_low], -TOTAL), (eqA.index[-1], float(eqA.iloc[-1]))]
    r = summarize("A 择时一次性买持(低估区入场)", eqA, cfA)
    results.append(r)
    curves["A 择时一次性买持(低估区入场)"] = eqA
    print(f"  A: {r}")

    # B 季度定投只轮新资金
    per = TOTAL / len(qdates)
    eqB = quarterly_dca(qdates, sel_dyn, union, per)
    cfB = [(d, -per) for d in qdates] + [(eqB.index[-1], float(eqB.iloc[-1]))]
    r = summarize("B 季度定投-只轮新资金", eqB, cfB)
    results.append(r)
    curves["B 季度定投-只轮新资金"] = eqB
    print(f"  B: {r}")

    # C 对照: 无择时一次性买持 (2021-03-31 起点)
    eqC = buy_and_hold(qdates, sel_dyn, union, 0, TOTAL)
    cfC = [(qdates[0], -TOTAL), (eqC.index[-1], float(eqC.iloc[-1]))]
    r = summarize("C 对照-无择时一次性买持", eqC, cfC)
    results.append(r)
    curves["C 对照-无择时一次性买持"] = eqC
    print(f"  C: {r}")

    summary = pd.DataFrame(results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 110)
    print(f"10 万资金安排对比 ({START} ~ {END}, 低估区={td.LOW_SIG} 信号)")
    print("=" * 110)
    print(summary.to_string(index=False))
    print("=" * 110)
    summary.to_csv(os.path.join(rr.RESULTS_DIR, "timing_lump_summary.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame(curves).to_csv(os.path.join(rr.RESULTS_DIR, "timing_lump_equity.csv"), encoding="utf-8-sig")

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [2.4, 1]})
        for name, eq in curves.items():
            axes[0].plot(eq.index, eq.values, label=name, linewidth=1.5)
        axes[0].axvline(qdates[first_low], color="#4B3FE3", linestyle="--", linewidth=1,
                        label=f"低估区入场 {qdates[first_low].date()}")
        axes[0].set_ylabel("组合市值 (元)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_title(f"10 万资金安排对比: 择时一次性买持 vs 季度定投只轮新资金 ({START} ~ {END})")

        sigd = sig.set_index("date")
        axes[1].bar(sigd.index, sigd["n_sig"], color="#4B3FE3", alpha=0.8, width=20)
        axes[1].axhline(td.LOW_SIG - 0.5, color="#E8463A", linestyle="--", linewidth=1)
        axes[1].set_ylabel("满足信号数")
        axes[1].set_yticks([0, 1, 2, 3, 4])
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        png = os.path.join(rr.RESULTS_DIR, "timing_lump_compare.png")
        fig.savefig(png, dpi=130)
        print(f"对比图: {png}")
    except Exception as e:
        print(f"画图跳过: {e}")


if __name__ == "__main__":
    main()
