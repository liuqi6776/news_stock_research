# -*- coding: utf-8 -*-
"""T7 信号可靠性审计 + 杠杆模拟

审计内容:
1. 进场信号质量: 每次进场后 1/3/6/12 个月传统行业等权收益(是否真买在低位)
2. 离场信号质量: 每次离场后 1/3/6/12 个月传统行业等权收益(是否真逃了顶)
3. 重大底部捕捉率: 传统行业等权月频局部最低点, 检查 s123>=3 覆盖情况
4. 持有/空仓月度收益分布
5. 杠杆模拟: 1x/1.5x/2x/3x (含6%年化融资成本), 空仓期不加杠杆(V8)
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
    hv_monthly_ret, monthly_from_daily, calc_stats, OUT_DIR,
)
from sector_rotation_traditional import TRADITIONAL_ETFS, build_signals4, run_graded  # noqa: E402

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

# ---- 1/2. 进出场信号质量 ----
yms = list(nav_panel.index)
hold = False
events = []
for i in range(len(yms) - 1):
    y = yms[i]
    n = int(sig.loc[y, "s123"])
    if not hold and n >= 3:
        hold = True
        events.append((y, "进"))
    elif hold and n <= 1:
        hold = False
        events.append((y, "出"))

trad_m = plain_trad_m.reindex(yms)  # 月收益对齐
trad_nav = (1 + plain_trad_m).cumprod().reindex(yms)

print("=" * 92)
print("【1】进场信号质量: 进场后 N 个月传统行业等权收益 (正=买对位置)")
print("-" * 92)
print(f"{'进场':<8} {'1月':>8} {'3月':>8} {'6月':>8} {'12月':>8} {'进场时等权回撤':>12}")
for y, act in events:
    if act != "进":
        continue
    i = yms.index(y)
    vals = {}
    for h in (1, 3, 6, 12):
        if i + h < len(yms):
            vals[h] = trad_nav.iloc[i + h] / trad_nav.iloc[i] - 1
        else:
            vals[h] = np.nan
    dd_now = trad_nav.iloc[i] / trad_nav.iloc[:i + 1].max() - 1
    print(f"{y:<8} {vals[1]:>7.1%} {vals[3]:>7.1%} {vals[6]:>7.1%} {vals[12]:>7.1%} {dd_now:>11.1%}")

print()
print("【2】离场信号质量: 离场后 N 个月传统行业等权收益 (负=逃顶成功)")
print("-" * 92)
print(f"{'离场':<8} {'1月':>8} {'3月':>8} {'6月':>8} {'12月':>8} {'离场时等权回撤':>12}")
for y, act in events:
    if act != "出":
        continue
    i = yms.index(y)
    vals = {}
    for h in (1, 3, 6, 12):
        if i + h < len(yms):
            vals[h] = trad_nav.iloc[i + h] / trad_nav.iloc[i] - 1
        else:
            vals[h] = np.nan
    dd_now = trad_nav.iloc[i] / trad_nav.iloc[:i + 1].max() - 1
    print(f"{y:<8} {vals[1]:>7.1%} {vals[3]:>7.1%} {vals[6]:>7.1%} {vals[12]:>7.1%} {dd_now:>11.1%}")

# ---- 3. 重大底部捕捉率 ----
print()
print("【3】传统行业等权重要底部 vs 信号覆盖 (s123>=3 才在低吸状态)")
print("-" * 92)
# 局部最低点: 该月为前后6个月的窗口最低
window = 6
low_idxs = []
for i in range(window, len(trad_nav) - window):
    if trad_nav.iloc[i] <= trad_nav.iloc[i - window:i].min() and \
       trad_nav.iloc[i] <= trad_nav.iloc[i + 1:i + window + 1].min():
        low_idxs.append(i)
# 合并相邻低点(6个月内) 取最深
bottoms = []
for i in low_idxs:
    if bottoms and i - bottoms[-1][1] <= 6:
        if trad_nav.iloc[i] < trad_nav.iloc[bottoms[-1][1]]:
            bottoms[-1] = (trad_nav.index[i], i)
    else:
        bottoms.append((trad_nav.index[i], i))

print(f"{'底部月份':<10} {'s123':>6} {'在场':>6} {'此后6月':>8} {'此后12月':>8}")
for ym, i in bottoms:
    n = int(sig.loc[ym, "s123"])
    in_market = n >= 3
    h6 = trad_nav.iloc[min(i + 6, len(trad_nav) - 1)] / trad_nav.iloc[i] - 1
    h12 = trad_nav.iloc[min(i + 12, len(trad_nav) - 1)] / trad_nav.iloc[i] - 1
    print(f"{ym:<10} {n:<6} {'是' if in_market else '否':<6} {h6:>7.1%} {h12:>7.1%}")

# ---- 4. 持有/空仓月度收益分布 ----
nv = run_graded(nav_panel, sig, plain_trad_m, v8_m, use_v8=True, mode="strict",
                entry_sig=3, exit_sig=1, sig_col="s123")
hold_mask = nv["w"] >= 0.99
h_ret = nv["ret"][hold_mask]
f_ret = nv["ret"][~hold_mask]
print()
print("【4】月度收益分布: 持有期(传统行业) vs 空仓期(V8避险)")
print(f"  持有期 {len(h_ret)}月: 均值={h_ret.mean():.2%} 胜率={(h_ret > 0).mean():.1%} "
      f"累计={(1 + h_ret).prod() - 1:.1%}")
print(f"  空仓期 {len(f_ret)}月: 均值={f_ret.mean():.2%} 胜率={(f_ret > 0).mean():.1%} "
      f"累计={(1 + f_ret).prod() - 1:.1%}")

# ---- 5. 杠杆模拟 ----
FIN_M = 0.06 / 12  # 6% 年化融资成本


def run_lever(L, nv, trad_m, v8_m, yms):
    """T7 状态机持仓 w 驱动: w>0 加杠杆持有, w==0 空仓V8(不加杠杆)"""
    records = []
    nav = 1.0
    prev_w = 0.0
    for i in range(len(yms) - 1):
        ym_ret = yms[i + 1]
        w = float(nv.loc[ym_ret, "w"])
        if w > 0:
            r = L * float(trad_m.get(ym_ret, 0.0)) - (L - 1) * FIN_M
        else:
            r = float(v8_m.get(ym_ret, 0.0))
        c = abs(w - prev_w) * 0.003
        nav *= (1 + r - c)
        records.append({"ym": ym_ret, "nav": nav, "w": w, "ret": r - c})
        prev_w = w
    return pd.DataFrame(records).set_index("ym")


print()
print("【5】杠杆模拟 (融资成本 6%/年, 只在持有期加杠杆, 空仓期V8不加杠杆)")
print("-" * 92)
print(f"{'杠杆':<8} {'NAV':>7} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'持有期':>7}")
lev_curves = {}
for L in (1.0, 1.5, 2.0, 3.0):
    lv = run_lever(L, nv, trad_m, v8_m, yms)
    lev_curves[L] = lv
    st = calc_stats(lv)
    note = " (仅理论)" if L >= 3 else ""
    print(f"{L:g}x{note:<7} {st['FinalNAV']:>6.2f} {st['CAGR']:>7.2%} {st['MaxDD']:>7.2%} "
          f"{st['Sharpe']:>6.2f} {st['Calmar']:>6.2f} {(lv['w'] > 0).mean():>6.1%}")

# 杠杆下最坏年度
print()
print("【6】2x 杠杆 vs 1x 年度收益对比")
print("-" * 92)
for L in (1.0, 2.0):
    lv = lev_curves[L]
    years = pd.Series(lv["ret"].values, index=pd.to_datetime(lv.index, format="%Y%m"))
    yr = years.groupby(years.index.year).apply(lambda s: (1 + s).prod() - 1)
    print(f"  {L:g}x: " + "  ".join(f"{y}:{v:+.1%}" for y, v in yr.items()))

print()
print("【结论数据】写入 traditional_leverage_stats.csv")
leverage_rows = []
for L in (1.0, 1.5, 2.0, 3.0):
    lv = lev_curves[L]
    st = calc_stats(lv)
    leverage_rows.append({"杠杆": L, **st})
pd.DataFrame(leverage_rows).to_csv(os.path.join(OUT_DIR, "traditional_leverage_stats.csv"),
                                   index=False, encoding="utf-8-sig")
