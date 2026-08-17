# -*- coding: utf-8 -*-
"""杠杆三·池子放宽: 中证1000 → 全市场 (ENS/C8, 冻结 binary s123 + 杠杆二 tiered 配置)。

背景:
  - 杠杆三因子扩容(VAL价值)已证无增量, 方向关闭。
  - 转向池子放宽: 全市场面板 stock_ml_panel_fullmarket_72m.parquet 已存在(5869只/73月/含C8全特征),
    且 project_memory 既有结论"全市场选股回撤从-40.1%收到-27.9%"是明确增益方向。
  - 本脚本在 ENS 引擎上对照: 中证1000(冻结基线) vs 全市场, 同口径打分/择时/成本。

步骤:
  1. init_shared("csi1000") 与 init_shared("fullmarket") 各建一次打分(ENH4+GBDT滚动WFO+ENS)。
  2. 干净对照: run_backtest(binary s123, no-TV), T40/T60。
  3. 杠杆二收口配置对照: run_backtest_tiered(tiered + dd_degrade=-10%×0.5), T40。
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

from engine import init_shared, run_backtest, run_backtest_tiered  # noqa: E402

t0 = time.time()


def metrics(nav_s):
    nav_s = nav_s.sort_index().astype(float)
    tot = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
    yrs = len(nav_s) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dd_s = nav_s / nav_s.cummax() - 1.0
    ret = nav_s.pct_change().fillna(0.0)
    sharpe = ret.mean() / (ret.std() + 1e-8) * np.sqrt(242.0)
    nav_m = nav_s.groupby((nav_s.index // 100).astype(str)).last()
    dd_m = nav_m / nav_m.cummax() - 1.0
    return {"ann": ann, "maxdd": dd_s.min(), "maxdd_m": dd_m.min(),
            "calmar": ann / (-dd_s.min() + 1e-9), "sharpe": sharpe}


def main():
    print("[1] 加载中证1000 shared (冻结基线)...", flush=True)
    sh_csi = init_shared("csi1000")
    print(f"    csi1000 完成 {time.time()-t0:.0f}s, panel {len(sh_csi['panel']):,} 行", flush=True)

    print("[2] 加载全市场 shared...", flush=True)
    sh_full = init_shared("fullmarket")
    print(f"    fullmarket 完成 {time.time()-t0:.0f}s, panel {len(sh_full['panel']):,} 行", flush=True)

    hdr = f"{'universe':<11} {'config':<22} {'CAGR':>8} {'MaxDD':>8} {'月DD':>8} {'Calmar':>6} {'Sharpe':>7}"
    print("\n" + hdr)
    print("-" * 70)

    # ---- 干净对照: binary s123, no-TV ----
    for uni, sh in [("csi1000", sh_csi), ("fullmarket", sh_full)]:
        for tag in ["T40", "T60"]:
            nav, _ = run_backtest(sh, "ENS", tag, tgt_vol=None)
            m = metrics(nav)
            print(f"{uni:<11} {'binary_'+tag:<22} {m['ann']:7.2%} {m['maxdd']:7.2%} "
                  f"{m['maxdd_m']:7.2%} {m['calmar']:5.2f} {m['sharpe']:6.2f}", flush=True)

    # ---- 杠杆二收口配置: tiered + dd_degrade=-10%×0.5 ----
    print("\n# 杠杆二收口配置 (tiered + dd_degrade -10% x 0.5), T40")
    print(hdr)
    print("-" * 70)
    for uni, sh in [("csi1000", sh_csi), ("fullmarket", sh_full)]:
        nav, _ = run_backtest_tiered(sh, "ENS", "T40", tgt_vol=None,
                                     timing_mode="tiered", dd_degrade=-0.10,
                                     dd_degrade_scale=0.5)
        m = metrics(nav)
        print(f"{uni:<11} {'tiered_dd_T40':<22} {m['ann']:7.2%} {m['maxdd']:7.2%} "
              f"{m['maxdd_m']:7.2%} {m['calmar']:5.2f} {m['sharpe']:6.2f}", flush=True)

    print(f"\n[完成] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
