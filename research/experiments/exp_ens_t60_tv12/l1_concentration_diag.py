# -*- coding: utf-8 -*-
"""杠杆一收尾诊断：真实持仓的行业集中度口径溯源。

回答 "Q9 的'工业/公用事业/医药 3 行业 55%' 与申万一级采样 12-14% 的矛盾"：
1. 通达信细分(110) 与 申万一级(31) 两个口径下的单行业/TOP3 集中度
2. 用 max-per-date（单日峰值）而非均值，避免被平均稀释
3. 结论判定: "55%" 是 TOP3 行业合计 还是 单行业集中
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

from engine import init_shared, run_backtest  # noqa: E402
from industry_l1 import TDX_TO_SW_L1  # noqa: E402

t0 = time.time()


def concentration(holdings, key_map, label):
    """key_map: ts_code -> 行业键。返回每调仓日的单行业最大权重 + top3合计 + 明细。"""
    rows = []
    for d, codes in holdings.items():
        if not codes:
            continue
        cnt = {}
        for c in codes:
            k = key_map.get(c, "其他")
            cnt[k] = cnt.get(k, 0) + 1
        n = len(codes)
        w = {k: v / n for k, v in cnt.items()}
        top3 = sum(sorted(w.values(), reverse=True)[:3])
        rows.append({"date": d, "n": n, "max_w": max(w.values()),
                     "top3_w": top3, "top_w": sorted(w.items(), key=lambda x: -x[1])[0]})
    df = pd.DataFrame(rows)
    return df


def report(df, label, st, ed):
    win = df[(df["date"] >= st) & (df["date"] <= ed)]
    print(f"\n=== {label} (回撤区间 {st}~{ed}, {len(win)} 个调仓日) ===")
    if len(win) == 0:
        print("  无数据")
        return
    # 单日峰值
    peak_max = win.loc[win["max_w"].idxmax()]
    peak_top3 = win.loc[win["top3_w"].idxmax()]
    print(f"  单行业权重: 均值 {win['max_w'].mean()*100:.1f}% | 单日峰值 {peak_max['max_w']*100:.1f}% "
          f"({peak_max['date']}, {peak_max['top_w'][0]}={peak_max['top_w'][1]*100:.1f}%)")
    print(f"  TOP3 合计: 均值 {win['top3_w'].mean()*100:.1f}% | 单日峰值 {peak_top3['top3_w']*100:.1f}% "
          f"({peak_top3['date']})")


def main():
    print("[1] init_shared...", flush=True)
    shared = init_shared()
    ind_map = shared["ind_map"]
    ind_l1_map = shared["ind_l1_map"]
    print(f"    完成 {time.time()-t0:.0f}s", flush=True)

    # 先自检: 映射覆盖
    missing = sorted(set(v for v in ind_map.values() if v not in TDX_TO_SW_L1))
    print(f"[自检] 通达信细分行业数={pd.Series(list(ind_map.values())).nunique()}, "
          f"TDX_TO_SW_L1 未覆盖={missing if missing else '无'}")

    # 回放 ENS_T60 no-TV (进取版), 记持仓
    nav_s, _, hold = run_backtest(shared, "ENS", "T60", tgt_vol=None, log_holdings=True)
    print(f"[回放] {len(hold)} 个调仓日有持仓, 完成 {time.time()-t0:.0f}s", flush=True)

    # 回撤区间
    cum = nav_s / nav_s.iloc[0]
    peak = cum.cummax()
    dd = cum / peak - 1
    trough = dd.idxmin()
    peak_date = nav_s.loc[:trough].idxmax()
    st, ed = int(peak_date), int(trough)
    print(f"\n最大回撤: {dd.min()*100:.2f}% | 峰 {st} → 谷 {ed}")
    # 2024H1
    st1, ed1 = 20240101, 20240630

    # 通达信细分口径
    df_tdx = concentration(hold, ind_map, "通达信细分")
    report(df_tdx, "通达信细分(110)", st, ed)
    report(df_tdx, "通达信细分(110) @2024H1", st1, ed1)

    # 申万一级口径
    df_l1 = concentration(hold, ind_l1_map, "申万一级")
    report(df_l1, "申万一级(31)", st, ed)
    report(df_l1, "申万一级(31) @2024H1", st1, ed1)

    # 直接打印回撤区间内 申万一级 单日峰值那天的完整分布
    win = df_l1[(df_l1["date"] >= st) & (df_l1["date"] <= ed)]
    if len(win):
        d_peak = win.loc[win["max_w"].idxmax(), "date"]
        codes = hold[d_peak]
        dist = {}
        for c in codes:
            k = ind_l1_map.get(c, "其他")
            dist[k] = dist.get(k, 0) + 1
        dist = {k: v / len(codes) for k, v in sorted(dist.items(), key=lambda x: -x[1])}
        print(f"\n申万一级 单日峰值分布 @ {d_peak} (共{len(codes)}只):")
        for k, v in dist.items():
            print(f"    {k:<10} {v*100:5.1f}%")
        top3 = sum(sorted(dist.values(), reverse=True)[:3])
        print(f"    → TOP3 合计 = {top3*100:.1f}%")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
