# -*- coding: utf-8 -*-
"""T9 诊断: 
1) T9a(MA20出) 是否"进场即被秒杀" -> 打印每日进出明细
2) 纯V8最终NAV (验证T9a是否从未真正持有)
3) T7日频 vs 月频 月度收益差异对比
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
    load_industry_daily, load_hv_daily, build_series, hv_monthly_ret,
    monthly_from_daily, calc_stats, COST, HV_WEIGHTS,
)
from sector_rotation_traditional import (  # noqa: E402
    TRADITIONAL_ETFS, build_signals4, run_graded,
)
from diag_t9_exit import (  # noqa: E402
    nav_trad, mas, boll_mid, boll_low, sig_daily, month_end_dates,
    dates, ew_trad, v8_daily, pe_pct_daily,
)

print("=" * 90)
print("[1] 纯V8最终NAV (验证T9a是否从未真正持有)")
v8_nav = (1 + v8_daily).prod()
print(f"  纯V8 全期 NAV = {v8_nav:.4f}  (T9a NAV={1.27})")

print("\n[2] T9a(跌破MA20出) 每日进出明细 (前80行)")
exit_raw = (nav_trad < mas[20]).fillna(False)
entry_raw = pd.Series(False, index=dates)
for d in month_end_dates:
    if d in sig_daily.index and sig_daily.loc[d] >= 3:
        entry_raw.loc[d] = True

nav = 1.0
state = "out"
prev_w = 0.0
events = []
for i, d in enumerate(dates):
    if i > 0:
        d_prev = dates[i - 1]
        if state == "out" and entry_raw.loc[d_prev]:
            state = "in"
        elif state == "in" and exit_raw.loc[d_prev]:
            state = "out"
    w = 1.0 if state == "in" else 0.0
    r = float(ew_trad.loc[d]) if state == "in" else float(v8_daily.get(d, 0.0))
    c = abs(w - prev_w) * COST
    nav *= (1 + r - c)
    if w != prev_w:
        events.append(f"{d} {'进' if w==1 else '出'} (nav={nav:.4f})")
    prev_w = w
for e in events[:30]:
    print(f"  {e}")
print(f"  共 {len(events)} 次切换, 最终NAV={nav:.4f}")

print("\n[3] 2019-02 的信号明细 (确认 s123 是否>=3)")
import importlib
from sector_rotation_traditional import TRADITIONAL_ETFS as _T
_panel = {c: s for c, s in load_industry_daily().items() if c in {c2 for _, c2 in _T}}
_hv = load_hv_daily()
_v8_m = hv_monthly_ret(_hv)
_mnav = {}
for code, s in _panel.items():
    _mnav[code] = (1 + s).cumprod().groupby(s.index.str[:6]).last()
_navp = pd.DataFrame(_mnav).sort_index()
_sig = build_signals4(list(_navp.index), _navp, [c for _, c in _T])
print(_sig.loc["2019-02":"2019-06", ["s1", "s2", "s3", "s4", "s123"]])

print("\n[4] T7 日频 vs 月频 月度收益对比 (2019-01 ~ 2023-12)")
# 日频T7
sig_d = sig_daily.copy()
def run_t7_daily():
    entry = pd.Series(False, index=dates)
    for d in month_end_dates:
        if d in sig_d.index and sig_d.loc[d] >= 3:
            entry.loc[d] = True
    exit_r = (sig_d <= 1).astype(bool).shift(1).fillna(False)
    n = 1.0; st = "out"; pw = 0.0; recs = []
    for i, d in enumerate(dates):
        if i > 0:
            p = dates[i - 1]
            if st == "out" and entry.loc[p]:
                st = "in"
            elif st == "in" and exit_r.loc[p]:
                st = "out"
        w = 1.0 if st == "in" else 0.0
        r = float(ew_trad.loc[d]) if st == "in" else float(v8_daily.get(d, 0.0))
        c = abs(w - pw) * COST
        n *= (1 + r - c)
        recs.append({"d": d, "nav": n, "w": w})
        pw = w
    o = pd.DataFrame(recs).set_index("d")
    o["ym"] = o.index.str[:6]
    return o.groupby("ym").nav.last()

daily_m = run_t7_daily()
# 月频T7
monthly_m = run_graded(_navp, _sig, monthly_from_daily(build_series(_panel)),
                       _v8_m, mode="strict", entry_sig=3, exit_sig=1, sig_col="s123")
# 对齐
common = daily_m.index.intersection(monthly_m.index)
cmp_df = pd.DataFrame({
    "日频nav": daily_m.reindex(common),
    "月频nav": monthly_m["nav"].reindex(common),
})
cmp_df["日频ret"] = cmp_df["日频nav"].pct_change()
cmp_df["月频ret"] = cmp_df["月频nav"].pct_change()
cmp_df["diff"] = (cmp_df["日频ret"] - cmp_df["月频ret"]).abs()
big = cmp_df[cmp_df["diff"] > 0.002]
print(f"  差异>0.2%的月份: {len(big)} 个 / 共 {len(common)} 个月")
print(big[["日频ret", "月频ret", "diff"]].head(20).to_string())
print(f"\n  日频T7最终NAV={daily_m.iloc[-1]:.4f}  月频T7最终NAV={monthly_m['nav'].iloc[-1]:.4f}")
