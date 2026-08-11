# -*- coding: utf-8 -*-
"""T9 卖出信号研究 v2: 月频低吸进场(s123>=3) + 卖出变体对比

v2 修复:
1. 月频信号出场(s123le1)改为"月末评估+次日生效" — 原实现 (sig_daily<=1).shift(1) 是双重shift,
   使 201902 的信号在 2019-02-12 就触发出场(应 2019-03-01), 导致日频T7 2019-02只赚3.7% vs 月频17%
2. 技术卖出(MA/布林)加 min_hold 缓冲期 — 低吸点(s123>=3)必然已跌破均线, 不加缓冲=进即秒杀
3. 新增"持仓高点回撤"卖出变体 — 低吸策略最自然的卖出: 进场后从持有期最高点回撤 X% 离场

统一: 进场=月末 s123>=3 (次日生效), 出场=变体(次日生效), 空仓期=V8, 30bps, 无前视
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fund_research", "studies", "rotation_dingtou"))

from etf_optimize_backtest2 import (  # noqa: E402
    load_industry_daily, load_hv_daily, build_series, monthly_from_daily,
    calc_stats, COST, OUT_DIR, HV_WEIGHTS,
)
from sector_rotation_traditional import TRADITIONAL_ETFS, build_signals4  # noqa: E402
from timing_dingtou import fetch_pe_csi300, _rolling_pct  # noqa: E402

# ---- 日频数据 ----
panel = load_industry_daily()
trad_codes = [c for _, c in TRADITIONAL_ETFS]
trad_panel = {c: s for c, s in panel.items() if c in set(trad_codes)}
ew_trad = build_series(trad_panel)            # 传统行业日收益
nav_trad = (1 + ew_trad).cumprod()            # 传统行业日净值

hv = load_hv_daily()
all_dates = sorted(set().union(*[set(s.index) for s in hv.values()]))
hdf = pd.DataFrame(index=all_dates)
for code, s in hv.items():
    hdf[code] = s.reindex(all_dates)
v8_daily = (hdf * pd.Series(HV_WEIGHTS)).sum(axis=1).fillna(0)  # V8日收益

dates = ew_trad.index  # 全部交易日 (str YYYYMMDD)

# ---- 月频信号 ----
monthly_nav = {}
for code, s in panel.items():
    monthly_nav[code] = (1 + s).cumprod().groupby(s.index.str[:6]).last()
nav_panel = pd.DataFrame(monthly_nav).sort_index()
sig = build_signals4(list(nav_panel.index), nav_panel, trad_codes)

# 月末信号(月频): 每交易日映射当月信号值 (未shift, T-1 由调用方处理)
sig_daily = sig["s123"].reindex(pd.Index(dates.str[:6])).fillna(0).astype(int)
sig_daily.index = dates

# ---- 日频卖出指标 ----
mas = {n: nav_trad.rolling(n).mean() for n in (3, 5, 8, 10, 15, 20, 30, 45, 60, 120, 250)}
boll_mid = nav_trad.rolling(20).mean()
boll_std = nav_trad.rolling(20).std()
boll_low = boll_mid - 2 * boll_std

# PE 分位 (月末评估)
pe = fetch_pe_csi300()
pe_pct_m = pd.Series(
    {ym: _rolling_pct(pe["pe_ttm"], pd.Timestamp(ym + "01") + pd.offsets.MonthEnd(0))
     for ym in nav_panel.index}).sort_index()
# 当月值全月沿用 (月末日检查用)
pe_pct_daily = pe_pct_m.reindex(pd.Index(dates.str[:6])).fillna(0)
pe_pct_daily.index = dates
# 上月值逐日生效 (无前视): T日 = 上月末PE分位 (月频信号 T-1 生效)
pe_pct_prev_daily = pe_pct_m.shift(1).reindex(pd.Index(dates.str[:6])).fillna(0)
pe_pct_prev_daily.index = dates

month_end_dates = set(dates.to_series().groupby(dates.str[:6]).max().values)


def build_entry_raw():
    """进场: 月末 s123>=3 -> 次日(T+1)进场"""
    e = pd.Series(False, index=dates)
    for d in month_end_dates:
        if d in sig_daily.index and sig_daily.loc[d] >= 3:
            e.loc[d] = True
    return e.shift(1).fillna(False)   # 已生效: 月末信号 T+1 进场


def build_exit_raw(exit_kind, exit_th=None):
    """出场原始信号 (T日评估, 不含 shift):
    - s123le1: 月频信号, 仅月末评估 (避免当月值日频shift提前触发)
    - ma/boll: 每日评估
    - pe_gt: 月末PE分位 -> 映射到日 (全月沿用)
    - peak_dd: 持仓高点回撤, 在回测循环内跟踪 (此处不适用)
    """
    if exit_kind == "s123le1":
        out = pd.Series(False, index=dates)
        for d in month_end_dates:
            if d in sig_daily.index and sig_daily.loc[d] <= 1:
                out.loc[d] = True
    elif exit_kind.startswith("ma"):
        n = int(exit_kind[2:])
        out = (nav_trad < mas[n]).fillna(False)
    elif exit_kind == "boll_mid":
        out = (nav_trad < boll_mid).fillna(False)
    elif exit_kind == "boll_low":
        out = (nav_trad < boll_low).fillna(False)
    elif exit_kind == "pe_gt":
        # 月频信号: 月末评估 PE分位>阈值 -> 次月生效 (无前视)
        out = pd.Series(False, index=dates)
        for d in month_end_dates:
            if d in pe_pct_daily.index and pe_pct_daily.loc[d] > exit_th:
                out.loc[d] = True
    else:
        raise ValueError(exit_kind)
    return out


def run_backtest(exit_kind=None, exit_th=None, min_hold=0, peak_dd=None,
                 pe_trim=None, pe_clear=None):
    """进场: 月末 s123>=3 (T+1). 出场: exit_kind (T+1), 进场后至少持 min_hold 交易日.
    peak_dd: 持仓期 nav_trad 高点回撤 > peak_dd 离场 (替代 exit_kind).
    pe_trim: 持仓期 上月末PE分位>阈值 -> 减半仓(0.5), 回落恢复满仓.
    pe_clear: 持仓期 上月末PE分位>阈值 -> 清仓 (与 MA 破位同样优先)."""
    entry = build_entry_raw()
    exit_raw = None if (peak_dd is not None or exit_kind is None) \
        else build_exit_raw(exit_kind, exit_th).shift(1).fillna(False)

    nav = 1.0
    state = "out"
    prev_w = 0.0
    hold_days = 0
    peak = 1.0
    records = []
    for i, d in enumerate(dates):
        if i > 0:
            d_prev = dates[i - 1]
            if state == "out":
                if entry.loc[d_prev]:
                    state = "in"
                    hold_days = 0
                    peak = float(nav_trad.loc[d])
            else:  # in
                hold_days += 1
                # 当日净值(持仓)更新高点
                cur = float(nav_trad.loc[d])
                peak = max(peak, cur)
                trig = False
                if peak_dd is not None:
                    trig = cur < peak * (1 - peak_dd)          # 高点回撤
                elif exit_raw is not None and hold_days >= min_hold and exit_raw.loc[d_prev]:
                    trig = True                                  # 均线/布林/PE(月末)
                if not trig and pe_clear is not None and \
                        float(pe_pct_prev_daily.loc[d]) > pe_clear:
                    trig = True                                  # PE高估清仓
                if trig:
                    state = "out"
        # 确定目标仓位 (PE 高估减半, 回落恢复)
        if state == "in" and pe_trim is not None:
            w = 0.5 if float(pe_pct_prev_daily.loc[d]) > pe_trim else 1.0
        else:
            w = 1.0 if state == "in" else 0.0
        r = w * float(ew_trad.loc[d]) + (1 - w) * float(v8_daily.get(d, 0.0))
        c = abs(w - prev_w) * COST
        nav *= (1 + r - c)
        records.append({"d": d, "nav": nav, "w": w})
        prev_w = w
    out = pd.DataFrame(records).set_index("d")
    out["ym"] = out.index.str[:6]
    return out


VARIANTS = [
    ("T7 月频s123<=1出",        dict(exit_kind="s123le1")),
    ("MA3 +20日缓冲",           dict(exit_kind="ma3", min_hold=20)),
    ("MA5 +10日缓冲",           dict(exit_kind="ma5", min_hold=10)),
    ("MA5 +20日缓冲",           dict(exit_kind="ma5", min_hold=20)),
    ("MA5 +30日缓冲",           dict(exit_kind="ma5", min_hold=30)),
    ("MA5 +40日缓冲",           dict(exit_kind="ma5", min_hold=40)),
    ("MA8 +20日缓冲",           dict(exit_kind="ma8", min_hold=20)),
    ("MA10+20日缓冲",           dict(exit_kind="ma10", min_hold=20)),
    ("MA15+20日缓冲",           dict(exit_kind="ma15", min_hold=20)),
    ("MA20+20日缓冲",           dict(exit_kind="ma20", min_hold=20)),
    ("MA30+20日缓冲",           dict(exit_kind="ma30", min_hold=20)),
]


def monthly_view(daily_df):
    m = daily_df.groupby("ym").agg(nav=("nav", "last"), w=("w", "last"))
    m["ret"] = m["nav"].pct_change().fillna(0)
    return m


if __name__ == "__main__":
    print("=" * 122)
    print(f"{'版本':<24} {'NAV':>6} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'仓位':>6} {'进出':>5}")
    print("-" * 122)
    results = {}
    monthlies = {}
    for v in VARIANTS:
        name = v[0]
        dd = run_backtest(**v[1])
        m = monthly_view(dd)
        monthlies[name] = m
        st = calc_stats(m)
        results[name] = st
        n_sw = int((m["w"].diff().abs() > 0).sum())
        print(f"{name:<24} {st['FinalNAV']:>6.2f} {st['CAGR']:>7.2%} {st['MaxDD']:>7.2%} "
              f"{st['Sharpe']:>6.2f} {st['Calmar']:>6.2f} {st['avg_w']:>5.0%} {n_sw:>4}次")

    for label, start in [("2021-01起", "2021-01"), ("2024-01起", "2024-01")]:
        print(f"\n{label}:")
        for v in VARIANTS:
            name = v[0]
            sub = monthlies[name]
            sub = sub[sub.index >= start]
            st = calc_stats(sub)
            print(f"  {name:<24} NAV={st['FinalNAV']:>5.2f} CAGR={st['CAGR']:>7.2%} "
                  f"MaxDD={st['MaxDD']:>7.2%} Calmar={st['Calmar']:>5.2f} 仓位={st['avg_w']:.0%}")

    print("\n=== 各卖出变体的进出场时点 (月频视图) ===")
    for v in VARIANTS:
        name = v[0]
        m = monthlies[name]
        chg = m[m["w"].diff().fillna(0) != 0]
        pairs = list(zip(chg.index, chg["w"]))
        evs = []
        for i in range(len(pairs)):
            ym, w = pairs[i]
            if w > 0 and (i == 0 or pairs[i - 1][1] == 0):
                evs.append(f"{ym}进")
            elif w == 0 and i > 0 and pairs[i - 1][1] > 0:
                evs.append(f"{ym}出")
        print(f"  {name:<24} " + " -> ".join(evs))

    print("\n=== 踏空检查 (出场后6个月传统行业涨幅>10% 记踏空) ===")
    for v in VARIANTS:
        name = v[0]
        m = monthlies[name]
        exits = m.index[(m["w"].diff() < 0)]
        misses = 0
        for ym in exits:
            idx = list(m.index).index(ym)
            if idx + 6 < len(m):
                fwd = m["nav"].iloc[idx + 6] / m["nav"].iloc[idx] - 1
                if fwd > 0.10:
                    misses += 1
        print(f"  {name:<24} 出场{len(exits)}次 踏空{misses}次")

    # 纯V8对照
    v8_nav = (1 + v8_daily).prod()
    print(f"\n纯V8 全期 NAV = {v8_nav:.4f} (CAGR={(v8_nav**(12/len(monthlies['T7 月频s123<=1出']))) - 1:.2%})")

    pd.DataFrame(results).T.to_csv(os.path.join(OUT_DIR, "traditional_t9_exit_stats.csv"),
                                   encoding="utf-8-sig")
    print(f"\n[saved] traditional_t9_exit_stats.csv")
