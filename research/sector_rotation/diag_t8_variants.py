# -*- coding: utf-8 -*-
"""T8 变体: 补漏底部的进场规则

背景: T7(S1S2S3>=3进/<=1出) 漏掉 2016-02/2018-12/2020-03 三个底部(均 s123=2)。
但三个底部 S4(池内哀嚎) 都满足。设计"任意 3 条信号满足"的进场组合:

  T7  进 = s123>=3                                  (基准)
  T8a 进 = s123>=3 OR (s2&s3&s4)                    补缺S1型底部(2016/2020疫情底)
  T8b 进 = s123>=3 OR (s1&s3&s4)                    补缺S2型底部(2018-12)
  T8c 进 = s123>=3 OR (s1&s2&s4)                    补缺S3型底部
统一离场 = s123<=1, 空仓期V8避险, 30bps, 无前视
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
    INDUSTRY_ETFS, load_industry_daily, load_hv_daily, build_series,
    hv_monthly_ret, monthly_from_daily, calc_stats, COST, OUT_DIR,
)
from sector_rotation_traditional import TRADITIONAL_ETFS, build_signals4  # noqa: E402

panel = load_industry_daily()
trad_codes = [c for _, c in TRADITIONAL_ETFS]
trad_panel = {c: s for c, s in panel.items() if c in set(trad_codes)}
ew_trad_daily = build_series(trad_panel)
plain_trad_m = monthly_from_daily(ew_trad_daily)

monthly_nav = {}
for code, s in panel.items():
    nav_s = (1 + s).cumprod()
    monthly_nav[code] = nav_s.groupby(s.index.str[:6]).last()
nav_panel = pd.DataFrame(monthly_nav).sort_index()

hv = load_hv_daily()
v8_m = hv_monthly_ret(hv)
sig = build_signals4(list(nav_panel.index), nav_panel, trad_codes)
yms = list(nav_panel.index)


def run_state(entry_fn, exit_fn):
    """状态机回测: entry_fn(row)->bool 进场, exit_fn(row)->bool 离场"""
    records = []
    nav = 1.0
    prev_w = None
    holding = False
    for i in range(len(yms) - 1):
        ym_sig, ym_ret = yms[i], yms[i + 1]
        row = sig.loc[ym_sig]
        if not holding and entry_fn(row):
            holding = True
        elif holding and exit_fn(row):
            holding = False
        w = 1.0 if holding else 0.0
        port = float(plain_trad_m.get(ym_ret, 0.0)) if holding else float(v8_m.get(ym_ret, 0.0))
        c = COST * w if prev_w is None else abs(w - prev_w) * COST
        nav *= (1 + port - c)
        records.append({"ym": ym_ret, "nav": nav, "w": w, "ret": port - c})
        prev_w = w
    return pd.DataFrame(records).set_index("ym")


# 进场/离场条件
def make_entry(s1=None, s2=None, s3=None, s4=None):
    """组合条件: 指定信号全满足 OR (s123>=3)。不指定任何信号时 = 仅 s123>=3"""
    has_combo = any(v is not None for v in (s1, s2, s3, s4))

    def f(r):
        combo = False
        if has_combo:
            combo = (s1 is None or r["s1"] == 1) and (s2 is None or r["s2"] == 1) and \
                    (s3 is None or r["s3"] == 1) and (s4 is None or r["s4"] == 1)
        return (r["s123"] >= 3) or combo
    return f


def exit_1(r):
    return r["s123"] <= 1


VARIANTS = [
    ("T7  S1S2S3>=3进", make_entry()),
    ("T8a 补缺S1(2&3&4)", make_entry(s2=1, s3=1, s4=1)),
    ("T8b 补缺S2(1&3&4)", make_entry(s1=1, s3=1, s4=1)),
    ("T8c 补缺S3(1&2&4)", make_entry(s1=1, s2=1, s4=1)),
]

print("=" * 112)
print(f"{'版本':<20} {'NAV':>6} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'仓位':>6} {'进出场数':>7}")
print("-" * 112)
all_navs = {}
events_map = {}
for name, entry_fn in VARIANTS:
    nv = run_state(entry_fn, exit_1)
    all_navs[name] = nv
    st = calc_stats(nv)
    # 进出场统计
    holding = False
    events = []
    for i in range(len(yms) - 1):
        y, r = yms[i], sig.loc[yms[i]]
        if not holding and entry_fn(r):
            holding = True
            events.append((y, "进"))
        elif holding and exit_1(r):
            holding = False
            events.append((y, "出"))
    events_map[name] = events
    n_events = len([e for e in events if e[1] == "进"])
    print(f"{name:<20} {st['FinalNAV']:>6.2f} {st['CAGR']:>7.2%} {st['MaxDD']:>7.2%} "
          f"{st['Sharpe']:>6.2f} {st['Calmar']:>6.2f} {st['avg_w']:>5.0%} {n_events:>5}次进")

print("\n=== 分期间 ===")
for label, start in [("2021-01起", "2021-01"), ("2024-01起", "2024-01")]:
    print(f"\n{label}:")
    for name, _ in VARIANTS:
        nv = all_navs[name]
        sub = nv[nv.index >= start]
        st = calc_stats(sub)
        print(f"  {name:<20} NAV={st['FinalNAV']:>5.2f} CAGR={st['CAGR']:>7.2%} "
              f"MaxDD={st['MaxDD']:>7.2%} Calmar={st['Calmar']:>5.2f} 仓位={st['avg_w']:.0%}")

# 底部捕捉检验
BOTTOMS = [("2016-02", "201602"), ("2018-12", "201812"), ("2020-03", "202003"),
           ("2022-10", "202210"), ("2024-08", "202408"), ("2025-04", "202504")]
print("\n=== 重要底部捕捉 (底部后6个月内是否在场) ===")
print(f"{'底部':<12}" + "".join(f"{n.split()[0]:>13}" for n, _ in VARIANTS))
for label, bm in BOTTOMS:
    b_idx = yms.index(bm)
    row = "  "
    for name, _ in VARIANTS:
        nv = all_navs[name]
        # 底部后6个月窗口内任意月持仓=1
        window = nv.iloc[b_idx:b_idx + 6]
        caught = bool((window["w"] >= 0.99).any())
        row += f"{'捕捉✓' if caught else '漏掉✗':>13}"
    print(f"{label:<12}{row}")

print("\n=== 进出场明细 ===")
for name, evs in events_map.items():
    print(f"\n{name}:")
    print("  " + " -> ".join(f"{y}{a}" for y, a in evs))

# 存盘
rows = []
for name, entry_fn in VARIANTS:
    nv = all_navs[name]
    st = calc_stats(nv)
    rows.append({"版本": name, **st})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "traditional_t8_variants.csv"),
                          index=False, encoding="utf-8-sig")
print(f"\n[saved] traditional_t8_variants.csv")
