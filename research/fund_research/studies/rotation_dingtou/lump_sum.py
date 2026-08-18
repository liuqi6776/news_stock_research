# -*- coding: utf-8 -*-
"""
4433 季度轮换 - 一次性投入 10 万回测 (非定投)
================================================

假设: 2021-03-31 (首个季度轮换点) 一次性投入 100000 元, 之后不再追加资金。

策略:
  A. 动态4433-季度全仓再平衡  每季末动态重算4433, 卖出去年名单外基金, 全部资产在当季名单内等权
  B. 动态4433-买入持有       2021-03-31 选一次后一直持有 (不轮换, 无赎回费损耗, 回答"轮换值不值")
  C. 静态206池-季度全仓再平衡 2026年4433静态池 (幸存者偏差对照)
  D. 沪深300联接-买入持有    一次性买入 110020 持有

费用: 申购费 0.15%, 赎回费按持有期阶梯, 与定投回测口径一致。

用法:
  C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/lump_sum.py \
    --start 2021-01-01 --end 2026-08-06 --initial 100000
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

STRATEGIES = ["动态4433-季度全仓再平衡", "动态4433-买入持有",
              "静态206-季度全仓再平衡", "沪深300联接-买入持有"]


def run_lump(qdates, selections, mode, initial, dates_idx):
    """一次性投入: 首期投入 initial, 之后每期 contribution=0。
    mode=full -> 每期全仓再平衡; mode=new -> 建仓后不动(买入持有)。"""
    port = rr.Portfolio()
    eq_parts = []
    for i, d in enumerate(qdates):
        contrib = initial if i == 0 else 0.0
        if mode == "full":
            rr.rebalance_full(port, selections[i], d, contrib)
        else:
            rr.invest_new_money(port, selections[i], d, contrib)
        d_end = qdates[i + 1] if i + 1 < len(qdates) else rr.END
        cash, shares = port.snapshot()
        seg = rr.segment_equity(cash, shares, d, d_end, dates_idx)
        if not seg.empty:
            eq_parts.append(seg)
    return pd.concat(eq_parts) if eq_parts else pd.Series(dtype=float)


def summarize_lump(name, equity, initial, start):
    final = float(equity.iloc[-1])
    total_ret = final / initial - 1.0
    cf = [(start, -initial), (equity.index[-1], final)]
    irr = rr.xirr(cf)
    years = (equity.index[-1] - start).days / 365.25
    cagr = (final / initial) ** (1.0 / years) - 1.0 if final > 0 else np.nan
    mdd = float((equity / equity.cummax() - 1.0).min())
    return {
        "strategy": name,
        "initial": initial,
        "final_value": round(final, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2) if np.isfinite(cagr) else np.nan,
        "xirr_pct": round(irr * 100, 2) if np.isfinite(irr) else np.nan,
        "mdd_pct": round(mdd * 100, 2),
    }


def main():
    ap = argparse.ArgumentParser(description="4433 季度轮换一次性投入回测")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-08-06")
    ap.add_argument("--initial", type=float, default=100000, help="一次性投入金额")
    ap.add_argument("--rebuild-panel", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    basic = rr.load_basic()
    active = basic[basic["fund_type"].isin(rr.ACTIVE_TYPES)]
    codes = active["code"].astype(str).tolist()
    basic_ft = basic.set_index("code")["fund_type"]
    print(f"动态池: {len(codes)} 只")

    static206 = []
    if os.path.exists(rr.STATIC_CSV):
        csv = pd.read_csv(rr.STATIC_CSV, dtype={"code": str})
        csv = csv[csv["is_4433"] == True]
        static206 = [str(int(c)).zfill(6) for c in csv["code"]]

    idx_code = None
    for c in rr.INDEX_CANDIDATES:
        if os.path.exists(os.path.join(rr.NAV_DIR, f"{c}.parquet")):
            idx_code = c
            break

    qdates = [pd.Timestamp(d) for d in pd.date_range(args.start, args.end, freq="Q")]
    if not qdates:
        print("无季度日期")
        return
    print(f"季度轮换点 {len(qdates)} 个: {qdates[0].date()} ~ {qdates[-1].date()}")
    rr.CONTRIBUTION = 0.0
    rr.END = pd.Timestamp(args.end)

    out, union = rr.compute_window_sums(codes, qdates, rebuild=args.rebuild_panel)
    sel_dyn = [rr.select_4433(out, codes, basic_ft, i) for i in range(len(qdates))]
    print(f"平均每季度通过4433: {np.mean([len(s) for s in sel_dyn]):.0f} 只")

    start = qdates[0]
    curves = {
        STRATEGIES[0]: run_lump(qdates, sel_dyn, "full", args.initial, union),
        STRATEGIES[1]: run_lump(qdates, sel_dyn, "new", args.initial, union),
        STRATEGIES[2]: run_lump(qdates, [static206] * len(qdates), "full", args.initial, union),
        STRATEGIES[3]: run_lump(qdates, [[idx_code]] * len(qdates), "new", args.initial, union),
    }

    rows = [summarize_lump(name, eq, args.initial, start) for name, eq in curves.items()]
    summary = pd.DataFrame(rows)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n" + "=" * 100)
    print(f"4433 季度轮换一次性投入回测 ({args.start} ~ {args.end}, 一次性 {args.initial:.0f} 元)")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("=" * 100)

    summary_out = os.path.join(rr.RESULTS_DIR, "lump_sum_summary.csv")
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")
    print(f"汇总表: {summary_out}")

    eq_df = pd.DataFrame(curves)
    eq_df.index.name = "date"
    eq_df.to_csv(os.path.join(rr.RESULTS_DIR, "lump_sum_equity.csv"), encoding="utf-8-sig")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(11, 6))
        for col in eq_df.columns:
            ax.plot(eq_df.index, eq_df[col], label=col, linewidth=1.6)
        ax.set_title(f"4433 轮换一次性投入 {args.initial:.0f} 元 ({args.start} ~ {args.end})")
        ax.set_ylabel("组合市值 (元)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png = os.path.join(rr.RESULTS_DIR, "lump_sum_compare.png")
        fig.savefig(png, dpi=130)
        print(f"对比图: {png}")
    except Exception as e:
        print(f"画图跳过: {e}")

    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
