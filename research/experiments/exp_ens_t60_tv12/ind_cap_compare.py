# -*- coding: utf-8 -*-
"""杠杆一对照（修正版）：进取版(no-TV) 单申万一级 <=20% 约束 vs 冻结基线。

上一版误把对照跑在 TV12 均衡版上（CAGR 8.4%/-21.7%），而用户讨论的 30% 回撤
是进取版 ENS_T40/T60_S123（11.5%/-31%）。本版改为 tgt_vol=None 对照，并采样
真实持仓的申万一级集中度，确认 <=20% 约束是否真的绑定。

判断标准（用户给定）:
  若 2024H1 段回撤砍掉 3-5pp 且 CAGR 不掉 >=0.5pp, 直接纳入。
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

from engine import init_shared, run_backtest, SQRT_242  # noqa: E402

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
    # 月频回撤
    nav_m = nav_s.groupby((nav_s.index // 100).astype(str)).last()
    dd_m = nav_m / nav_m.cummax() - 1.0
    return {"ann": ann, "maxdd": maxdd, "maxdd_m": dd_m.min(),
            "calmar": ann / (-maxdd + 1e-9), "sharpe": sharpe,
            "peak": int(peak), "trough": int(trough)}


def window_dd(nav_s, start, end):
    win = nav_s[(nav_s.index >= start) & (nav_s.index <= end)]
    if len(win) == 0:
        return np.nan
    wc = win / win.iloc[0]
    return float((wc / wc.cummax() - 1).min())


def fmt(d):
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def l1_concentration(holdings, ind_l1_map, ind_map, panel):
    """返回每个调仓日的申万一级权重分布（等权近似: 每级股数/总股数）。"""
    rows = []
    for d, codes in holdings.items():
        if not codes:
            continue
        l1_cnt = {}
        for c in codes:
            l1 = ind_l1_map.get(c, ind_map.get(c, "其他"))
            l1_cnt[l1] = l1_cnt.get(l1, 0) + 1
        n = len(codes)
        for l1, cnt in l1_cnt.items():
            rows.append({"date": d, "l1": l1, "weight": cnt / n, "cnt": cnt, "total": n})
    return pd.DataFrame(rows)


def main():
    print("[1] init_shared (加载面板+行情+重训GBDT, 一次)...", flush=True)
    shared = init_shared()
    ind_l1_map = shared["ind_l1_map"]
    ind_map = shared["ind_map"]
    panel = shared["panel"]
    print(f"    完成, 耗时{time.time()-t0:.0f}s", flush=True)

    # 进取版(no-TV) 对照: T60 与 T40
    for top_tag in ["T60", "T40"]:
        print("\n" + "=" * 100)
        print(f"=== 进取版 ENS_{top_tag}_S123 (no-TV) 对照 ===")
        print("=" * 100)
        res = {}
        for tag, cap in [("基线(无约束)", None), ("单申万一级<=20%", 0.20)]:
            nav_s, _, hold = run_backtest(shared, "ENS", top_tag, tgt_vol=None,
                                          cap_ind_l1=cap, log_holdings=True)
            m = metrics(nav_s)
            m["nav"] = nav_s
            m["holdings"] = hold
            res[tag] = m
            print(f"    {tag:<14} CAGR={m['ann']:7.2%} MaxDD={m['maxdd']:7.2%} "
                  f"(月频{m['maxdd_m']:7.2%}) Calmar={m['calmar']:5.2f} Sharpe={m['sharpe']:5.2f} "
                  f"峰{fmt(m['peak'])} 谷{fmt(m['trough'])}", flush=True)

        base, capr = res["基线(无约束)"], res["单申万一级<=20%"]
        print(f"\n{'指标':<12} {'基线':>16} {'<=20%':>16} {'差值':>16}")
        print("-" * 62)
        for k, label, pct in [("ann", "CAGR", True), ("maxdd", "MaxDD(日)", True),
                              ("maxdd_m", "MaxDD(月)", True),
                              ("sharpe", "Sharpe", False), ("calmar", "Calmar", False)]:
            a, b = base[k], capr[k]
            print(f"{label:<12} {a:>15.2%} {b:>15.2%} {b-a:>+15.2%}")

        # 2024H1 与全回撤窗口
        dd_b = window_dd(base["nav"], 20240101, 20240630)
        dd_c = window_dd(capr["nav"], 20240101, 20240630)
        print(f"\n  2024H1 (2024-01~06) 窗口回撤: 基线 {dd_b:.2%} | <=20% {dd_c:.2%} | 改善 {dd_b-dd_c:+.2%}")
        st, ed = base["peak"], base["trough"]
        dd_b2 = window_dd(base["nav"], st, ed)
        dd_c2 = window_dd(capr["nav"], st, ed)
        print(f"  全回撤区间 {fmt(st)}~{fmt(ed)} 窗口回撤: 基线 {dd_b2:.2%} | <=20% {dd_c2:.2%} | 改善 {dd_b2-dd_c2:+.2%}")

        # 集中度采样: 基线(无约束) 在回撤区间内的申万一级分布
        hold = base["holdings"]
        conc = l1_concentration(hold, ind_l1_map, ind_map, panel)
        if len(conc):
            win = conc[(conc["date"] >= st) & (conc["date"] <= ed)]
            if len(win):
                top5 = (win.groupby("l1")["weight"].mean().sort_values(ascending=False).head(5) * 100).round(1)
                max_w = win.groupby("l1")["weight"].mean().max() * 100
                over = win.groupby("l1")["weight"].mean()[win.groupby("l1")["weight"].mean() > 0.20]
                print(f"\n  基线回撤区间内申万一级平均权重 Top5: {dict(top5)}")
                print(f"  单一申万一级最高平均权重: {max_w:.1f}%")
                print(f"  超过20%的申万一级: {dict((over*100).round(1)) if len(over) else '无'}")
            # 2024H1 单独
            win1 = conc[(conc["date"] >= 20240101) & (conc["date"] <= 20240630)]
            if len(win1):
                top5_1 = (win1.groupby("l1")["weight"].mean().sort_values(ascending=False).head(5) * 100).round(1)
                max_w1 = win1.groupby("l1")["weight"].mean().max() * 100
                print(f"  2024H1 申万一级平均权重 Top5: {dict(top5_1)} | 最高 {max_w1:.1f}%")


if __name__ == "__main__":
    main()
