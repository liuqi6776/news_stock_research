# -*- coding: utf-8 -*-
"""回撤口径对比分析: 股票方案(日频) vs T7(月频) 统一口径

从 stock_gbdt_s123_results.pkl 读取 (主脚本已保存 nav_dated + t7)
对比:
  1. 日频口径: 股票 MaxDD(逐日) vs T7 MaxDD(逐日, 用月频 NAV 内插/月末)
  2. 月频口径: 股票 MaxDD(月末重采样) vs T7 MaxDD(月末) — 与交易同频
  3. 关键结论: 口径差异有多大, 对比是否公平
"""
import os, pickle
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
PKL = os.path.join(ROOT, "research", "sector_rotation", "results", "stock_gbdt_s123_results.pkl")

with open(PKL, "rb") as f:
    data = pickle.load(f)
results, t7 = data["results"], data["t7"]

# T7 月频 nav (index=ym int) → 月末日期 int
t7_m = t7["nav"].copy()
t7_m.index = [int(i) for i in t7_m.index]
t7_daily = {}
for ym in t7_m.index:
    y, m = ym // 100, ym % 100
    last_day = (pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)).day
    t7_daily[y * 10000 + m * 100 + last_day] = t7_m[ym]
t7_daily = pd.Series(t7_daily).sort_index()

def maxdd(ser):
    return ((ser - ser.cummax()) / ser.cummax()).min()

def month_end(ser):
    """日频 → 月末重采样 (交易同频口径)"""
    idx = pd.to_datetime([str(int(d)) for d in ser.index], format="%Y%m%d")
    s = pd.Series(ser.values, index=idx)
    return s.resample("M").last()

KEYS = [
    ("ENS_T40_S123_ONLY_S123", "ENS_T40_S123"),
    ("ENS_T60_S123_ONLY_S123", "ENS_T60_S123"),
    ("ENS_T60_S123_TV12",      "ENS_T60_TV12"),
]

print("=" * 90)
print(f"{'方案':<18} {'日频MaxDD':>10} {'月频MaxDD':>10} {'月/日比':>8} | {'T7日频':>8} {'T7月频':>8}")
print("-" * 90)

t7_dd_daily = maxdd(t7_daily)          # T7 日频(月末点连接, 近似)
t7_dd_month = maxdd(t7_m)              # T7 月频(原口径)
for tag, label in KEYS:
    nav = results[tag]["nav_dated"]
    dd_daily = maxdd(nav)              # 日频 (原口径)
    nav_m = month_end(nav)
    dd_month = maxdd(nav_m)            # 月频 (交易同频)
    ratio = dd_month / dd_daily
    print(f"{label:<18} {dd_daily:>10.2%} {dd_month:>10.2%} {ratio:>8.2f} | "
          f"{t7_dd_daily:>8.2%} {t7_dd_month:>8.2%}")

print("-" * 90)
print(f"T7 原口径(月频): {t7_dd_month:.2%} | T7 月末连接近似日频: {t7_dd_daily:.2%}")

# 统一口径后的公平对比 (月频 = 交易同频)
print("\n=== 统一月频口径后 vs T7 (交易同频) ===")
for tag, label in KEYS:
    nav = results[tag]["nav_dated"]
    nav_m = month_end(nav)
    dd_m = maxdd(nav_m)
    print(f"  {label:<16} 月频MaxDD={dd_m:>7.2%}  vs T7月频 {t7_dd_month:>7.2%} "
          f"→ 差距 {dd_m - t7_dd_month:+.2f}pp")

# 统一日频口径后 vs T7 (T7 用月末点连接, 是月频点, 日频口径需逐日数据, 无法完全对齐)
# 但可对比"股票日频 vs T7月末点"(保守)
print("\n=== 股票日频 vs T7月末连接点 (T7无日内数据, 此对比对T7有利) ===")
for tag, label in KEYS:
    dd_d = maxdd(results[tag]["nav_dated"])
    print(f"  {label:<16} 日频MaxDD={dd_d:>7.2%}  vs T7月末 {t7_dd_daily:>7.2%} "
          f"→ 差距 {dd_d - t7_dd_daily:+.2f}pp")
