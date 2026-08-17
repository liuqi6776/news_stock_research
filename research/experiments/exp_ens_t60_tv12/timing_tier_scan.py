# -*- coding: utf-8 -*-
"""杠杆二深化：细扫降档参数 + MA20 快信号合成对照。

阶段1: 细扫组合回撤降档参数 (阈值 × 幅度)，binary/tiered 两套择时
阶段2: MA20 三档接入对照 (ma20 纯 / s123_ma20 合成)，+ 最优降档叠加
基准: 进取版 no-TV, ENS_T60/T40
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from engine import init_shared, run_backtest, run_backtest_tiered, SQRT_242  # noqa: E402

t0 = time.time()


def metrics(nav_s):
    nav_s = nav_s.sort_index().astype(float)
    tot = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
    yrs = len(nav_s) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dd_s = nav_s / nav_s.cummax() - 1.0
    maxdd = dd_s.min()
    trough = dd_s.idxmin()
    peak = nav_s.loc[:trough].idxmax()
    ret = nav_s.pct_change().fillna(0.0)
    sharpe = ret.mean() / (ret.std() + 1e-8) * SQRT_242
    nav_m = nav_s.groupby((nav_s.index // 100).astype(str)).last()
    dd_m = nav_m / nav_m.cummax() - 1.0
    return {"ann": ann, "maxdd": maxdd, "maxdd_m": dd_m.min(),
            "calmar": ann / (-maxdd + 1e-9), "sharpe": sharpe,
            "peak": int(peak), "trough": int(trough)}


def fmt(d):
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def run_variant(shared, top_tag, timing_mode, dd_degrade=None, dd_degrade_scale=0.5):
    nav, _ = run_backtest_tiered(shared, "ENS", top_tag, tgt_vol=None,
                                 timing_mode=timing_mode, dd_degrade=dd_degrade,
                                 dd_degrade_scale=dd_degrade_scale)
    return metrics(nav)


def main():
    print("[1] init_shared...", flush=True)
    shared = init_shared()
    print(f"    完成 {time.time()-t0:.0f}s", flush=True)

    DEG_THR = [-0.08, -0.10, -0.12, -0.15]
    DEG_SC = [0.3, 0.5, 0.7]

    for top_tag in ["T60", "T40"]:
        base = metrics(run_backtest(shared, "ENS", top_tag, tgt_vol=None)[0])
        print("\n" + "=" * 110)
        print(f"=== 杠杆二深化 ENS_{top_tag}_S123 (no-TV)  基线: "
              f"CAGR {base['ann']:.2%} MaxDD {base['maxdd']:.2%} 月DD {base['maxdd_m']:.2%} "
              f"Calmar {base['calmar']:.2f} Sharpe {base['sharpe']:.2f} ===")
        print("=" * 110)

        # ---------- 阶段1: 细扫降档参数 ----------
        for timing_mode in ["binary", "tiered"]:
            print(f"\n--- 阶段1 细扫降档 [{timing_mode}]  (Calmar / MaxDD / CAGR) ---")
            print(f"{'阈值':>8}", end="")
            for sc in DEG_SC:
                print(f" {'幅度'+str(sc):>22}", end="")
            print()
            best = None
            for thr in DEG_THR:
                print(f"{thr:>8.0%}", end="")
                for sc in DEG_SC:
                    m = run_variant(shared, top_tag, timing_mode, dd_degrade=thr,
                                    dd_degrade_scale=sc)
                    print(f"  {m['calmar']:.2f}/{m['maxdd']:.2%}/{m['ann']:.2%}", end="")
                    if best is None or m["calmar"] > best["m"]["calmar"]:
                        best = {"thr": thr, "sc": sc, "m": m}
                print()
            print(f"  → 最优: 阈值{best['thr']:.0%} 幅度{best['sc']}  "
                  f"CAGR {best['m']['ann']:.2%} MaxDD {best['m']['maxdd']:.2%} "
                  f"月DD {best['m']['maxdd_m']:.2%} Calmar {best['m']['calmar']:.2f} "
                  f"Sharpe {best['m']['sharpe']:.2f}")

        # ---------- 阶段2: MA20 合成对照 ----------
        print(f"\n--- 阶段2 MA20 合成对照 (无降档) ---")
        print(f"{'配置':<22} {'CAGR':>8} {'MaxDD':>8} {'月DD':>8} {'Calmar':>6} {'Sharpe':>7} {'峰':>10} {'谷':>10}")
        print("-" * 88)
        print(f"{'基线(binary 冻结)':<22} {base['ann']:7.2%} {base['maxdd']:7.2%} "
              f"{base['maxdd_m']:7.2%} {base['calmar']:5.2f} {base['sharpe']:6.2f} "
              f"{fmt(base['peak']):>10} {fmt(base['trough']):>10}")
        for tm, label in [("tiered", "s123三档(1.0/0.5/0)"),
                          ("ma20", "纯MA20三档(0.98)"),
                          ("s123_ma20", "s123×MA20合成")]:
            m = run_variant(shared, top_tag, tm)
            print(f"{label:<22} {m['ann']:7.2%} {m['maxdd']:7.2%} "
                  f"{m['maxdd_m']:7.2%} {m['calmar']:5.2f} {m['sharpe']:6.2f} "
                  f"{fmt(m['peak']):>10} {fmt(m['trough']):>10}")

        # ---------- 阶段3: 最优合成 + 最优降档叠加 ----------
        print(f"\n--- 阶段3 s123×MA20 合成 + 降档细扫 (最优合成路径) ---")
        print(f"{'阈值':>8}", end="")
        for sc in DEG_SC:
            print(f" {'幅度'+str(sc):>22}", end="")
        print()
        best2 = None
        for thr in DEG_THR:
            print(f"{thr:>8.0%}", end="")
            for sc in DEG_SC:
                m = run_variant(shared, top_tag, "s123_ma20", dd_degrade=thr,
                                dd_degrade_scale=sc)
                print(f"  {m['calmar']:.2f}/{m['maxdd']:.2%}/{m['ann']:.2%}", end="")
                if best2 is None or m["calmar"] > best2["m"]["calmar"]:
                    best2 = {"thr": thr, "sc": sc, "m": m}
            print()
        print(f"  → s123×MA20+降档最优: 阈值{best2['thr']:.0%} 幅度{best2['sc']}  "
              f"CAGR {best2['m']['ann']:.2%} MaxDD {best2['m']['maxdd']:.2%} "
              f"月DD {best2['m']['maxdd_m']:.2%} Calmar {best2['m']['calmar']:.2f} "
              f"Sharpe {best2['m']['sharpe']:.2f}")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
