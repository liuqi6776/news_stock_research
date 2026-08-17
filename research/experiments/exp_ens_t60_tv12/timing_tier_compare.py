# -*- coding: utf-8 -*-
"""杠杆二对照：择时四档梯度 + 组合回撤硬降档。

对比 (进取版 no-TV, ENS_T60/T40):
  1. 基线   : run_backtest          (滞回二元开关, 冻结基线)
  2. 校验   : run_backtest_tiered   timing_mode='binary'  (应与基线一致, 回归检查)
  3. 梯度   : run_backtest_tiered   timing_mode='tiered'  (s123 1.0/0.5/0)
  4. 二元+降档: run_backtest_tiered timing_mode='binary' dd_degrade=-0.10
  5. 梯度+降档: run_backtest_tiered timing_mode='tiered'  dd_degrade=-0.10
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


def main():
    print("[1] init_shared...", flush=True)
    shared = init_shared()
    print(f"    完成 {time.time()-t0:.0f}s", flush=True)

    # 回撤期 s123 信号路径（辅助解读）
    sig = shared["sig_df"]
    print("\n=== 回撤期(2023-07~2024-05) s123 信号 ===")
    for ym, row in sig.loc[(sig.index >= 202307) & (sig.index <= 202405)].iterrows():
        print(f"  {ym}: s123={int(row['s123'])}")

    for top_tag in ["T60", "T40"]:
        print("\n" + "=" * 100)
        print(f"=== 杠杆二对照 ENS_{top_tag}_S123 (no-TV) ===")
        print("=" * 100)

        # 1. 基线 (冻结版 run_backtest)
        nav_base, _ = run_backtest(shared, "ENS", top_tag, tgt_vol=None)
        b = metrics(nav_base)

        # 2. 回归校验: tiered 引擎的 binary 模式
        nav_bin, _ = run_backtest_tiered(shared, "ENS", top_tag, tgt_vol=None,
                                         timing_mode="binary")
        binm = metrics(nav_bin)

        # 3-5. 变体
        variants = {
            "梯度(1.0/0.5/0)": dict(timing_mode="tiered", dd_degrade=None),
            "二元+回撤-10%降档": dict(timing_mode="binary", dd_degrade=-0.10),
            "梯度+回撤-10%降档": dict(timing_mode="tiered", dd_degrade=-0.10),
        }
        res = {}
        for name, kw in variants.items():
            nav, _ = run_backtest_tiered(shared, "ENS", top_tag, tgt_vol=None, **kw)
            res[name] = (metrics(nav), nav)

        # 打印
        hdr = f"{'配置':<20} {'CAGR':>8} {'MaxDD':>8} {'MaxDD月':>8} {'Calmar':>6} {'Sharpe':>7} {'峰':>10} {'谷':>10}"
        print(hdr)
        print("-" * len(hdr))
        print(f"{'基线(冻结)':<20} {b['ann']:7.2%} {b['maxdd']:7.2%} {b['maxdd_m']:7.2%} "
              f"{b['calmar']:5.2f} {b['sharpe']:6.2f} {fmt(b['peak']):>10} {fmt(b['trough']):>10}")
        print(f"{'校验binary(应≈基线)':<20} {binm['ann']:7.2%} {binm['maxdd']:7.2%} "
              f"{binm['maxdd_m']:7.2%} {binm['calmar']:5.2f} {binm['sharpe']:6.2f} "
              f"{fmt(binm['peak']):>10} {fmt(binm['trough']):>10}")
        for name, (m, _nav) in res.items():
            print(f"{name:<20} {m['ann']:7.2%} {m['maxdd']:7.2%} {m['maxdd_m']:7.2%} "
                  f"{m['calmar']:5.2f} {m['sharpe']:6.2f} {fmt(m['peak']):>10} {fmt(m['trough']):>10}")

        # 回归校验偏差
        d_ann = binm["ann"] - b["ann"]
        d_dd = binm["maxdd"] - b["maxdd"]
        print(f"\n  [回归校验] binary vs 基线: CAGR {d_ann:+.4%}, MaxDD {d_dd:+.4%} "
              f"{'(OK)' if abs(d_ann) < 1e-4 and abs(d_dd) < 1e-4 else '(差异!)'}")

        # 各变体相对基线的回撤改善 + CAGR 变化
        print(f"\n  [相对基线(冻结)的变化]")
        for name, (m, _nav) in res.items():
            print(f"    {name:<20} CAGR {m['ann']-b['ann']:+7.2%} | MaxDD {m['maxdd']-b['maxdd']:+7.2%} "
                  f"| 月频DD {m['maxdd_m']-b['maxdd_m']:+7.2%} | Calmar {m['calmar']-b['calmar']:+6.2f}")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
