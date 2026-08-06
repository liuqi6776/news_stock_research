# -*- coding: utf-8 -*-
"""
P2-r5: 实际换手成本模型 vs 固定 20bps 敏感性（审查 P2「继续研究」第二项）

上游 run_validation.run_fast 使用「固定双边 20bps」成本（net = gross - 20bps）。
本脚本用每期实际换手率（turnover_<factor>.csv, 由 run_fast 新增持久化）建模:

  - 固定成本场景: net = gross - c/10000, c ∈ {0, 5, 10, 20, 30, 50} bps
  - 换手率驱动场景: net = gross - turnover × (2 × per_side_bps)/10000,
    per_side ∈ {2.5, 5, 10, 15} bps（双边即 5/10/20/30 bps @ 100% 换手）
  - 恢复 gross: saved returns 为固定 20bps 净值 → gross = net20 + 20bps

输出: cost_sensitivity_report.json + 控制台
结论判读: 若换手驱动成本 < 固定 20bps（实际月换手 < 100%）, 则固定 20bps 假设偏保守,
策略真实表现应不劣于文档数字; 关注指标翻盘点（如超额转负、Sharpe 崩坏）。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "research", "factor_dic", "results")

FIXED_BPS_BASELINE = 20
FIXED_BPS_GRID = (0, 5, 10, 20, 30, 50)
PER_SIDE_BPS_GRID = (2.5, 5.0, 10.0, 15.0)  # 双边 = 2×


def metrics(net_rets):
    net_rets = np.asarray(net_rets, float)
    nav = np.cumprod(1 + net_rets)
    n = len(net_rets)
    years = n / 12.0
    cagr = nav[-1] ** (1 / years) - 1 if nav[-1] > 0 else np.nan
    sharpe = net_rets.mean() / net_rets.std(ddof=1) * np.sqrt(12) if net_rets.std(ddof=1) > 0 else np.nan
    mdd = ((np.maximum.accumulate(nav) - nav) / np.maximum.accumulate(nav)).max()
    win = (net_rets > 0).mean()
    return {"cagr": float(cagr), "sharpe": float(sharpe), "mdd": float(mdd), "win": float(win)}


def main():
    factor = "turnover_vol_20"
    ret_fp = os.path.join(RES, f"returns_{factor}.csv")
    to_fp = os.path.join(RES, f"turnover_{factor}.csv")
    bench_fp = os.path.join(RES, "returns_bench.csv")
    if not (os.path.exists(ret_fp) and os.path.exists(to_fp)):
        print(f"缺少 {ret_fp} / {to_fp}: 请先运行 run_validation.py {factor} 生成收益与换手率序列")
        sys.exit(2)

    net20 = pd.read_csv(ret_fp, index_col=0).iloc[:, 0]
    gross = net20 + FIXED_BPS_BASELINE / 10000.0   # 恢复毛收益
    turn = pd.read_csv(to_fp, index_col=0).iloc[:, 0]
    bench = pd.read_csv(bench_fp, index_col=0).iloc[:, 0].reindex(net20.index)

    idx = gross.index
    n = len(gross)
    print(f"[load] {factor}: {n} 个月 | 换手率均值={turn.mean():.1%} 中位={turn.median():.1%} "
          f"P90={turn.quantile(0.9):.1%} 最大={turn.max():.1%}")

    rows = {}
    # 1) 固定成本场景
    for c in FIXED_BPS_GRID:
        r = gross - c / 10000.0
        m = metrics(r)
        m["excess_vs_bench"] = float((1 + r).prod() / (1 + bench).prod() - 1)
        m["cost_note"] = f"固定 {c}bps"
        rows[f"fixed_{c}bps"] = m

    # 2) 换手率驱动成本场景
    for ps in PER_SIDE_BPS_GRID:
        cost = turn * (2 * ps) / 10000.0
        r = gross - cost
        m = metrics(r)
        m["excess_vs_bench"] = float((1 + r).prod() / (1 + bench).prod() - 1)
        m["cost_note"] = f"换手驱动 per_side={ps}bps（双边{2*ps}bps@100%换手）"
        m["mean_cost_bps"] = float(cost.mean() * 10000)
        rows[f"turnover_{ps}bps_side"] = m

    # 3) 隐含均衡: 固定 20bps 等价于多少 per_side bps？
    #    gross - 20bps ≡ gross - turnover×2×x → x = 20bps/(2×mean_turnover)
    mean_to = float(turn.mean())
    implied_per_side = (FIXED_BPS_BASELINE / 2.0) / mean_to if mean_to > 0 else np.nan

    report = {
        "scope": "P2-r5 cost sensitivity (2026-08-06)",
        "factor": factor,
        "n_month": int(n),
        "baseline": rows["fixed_20bps"],
        "scenarios": rows,
        "turnover_stats": {
            "mean": float(turn.mean()), "median": float(turn.median()),
            "p90": float(turn.quantile(0.9)), "max": float(turn.max()),
        },
        "implied_per_side_bps_eq_fixed20": implied_per_side,
        "interpretation": ("换手率驱动成本在 per_side≥10bps（双边≥20bps@100%换手）时与固定 20bps 相当; "
                           "若真实双边成本低于该隐含值, 固定 20bps 假设偏保守, 结论更稳健。"),
    }
    with open(os.path.join(HERE, "cost_sensitivity_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # ---- 控制台 ----
    print(f"\n== {factor} 成本敏感性（毛收益基准, {n} 个月）==")
    print(f"{'场景':<34}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'月胜率':>7}{'超额v基准':>10}{'均成本bps':>10}")
    for k, m in rows.items():
        mc = m.get("mean_cost_bps", float(k.split("_")[1].replace("bps", "")))
        print(f"{m['cost_note']:<34}{m['cagr']:>8.2%}{m['sharpe']:>8.2f}{m['mdd']:>8.2%}"
              f"{m['win']:>7.1%}{m['excess_vs_bench']:>10.2%}{mc:>10.2f}")
    print(f"\n换手率均值 {mean_to:.1%} → 固定 20bps 等价于 per_side ≈ {implied_per_side:.1f}bps"
          f"（双边 {2*implied_per_side:.1f}bps @ 均值换手）")

    base = rows["fixed_20bps"]
    worst = min((m for k, m in rows.items() if k != "fixed_20bps"), key=lambda m: m["sharpe"])
    print(f"\n基线(固定20bps): CAGR={base['cagr']:.2%} Sharpe={base['sharpe']:.2f} "
          f"超额={base['excess_vs_bench']:.2%}")
    print(f"最差场景({worst['cost_note']}): CAGR={worst['cagr']:.2%} Sharpe={worst['sharpe']:.2f} "
          f"超额={worst['excess_vs_bench']:.2%}")

    print(f"\n[保存] {os.path.join(HERE, 'cost_sensitivity_report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
