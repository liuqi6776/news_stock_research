# -*- coding: utf-8 -*-
"""绘制 2023-01 起 OOS 收益曲线 (当前最优组合对比)

复用 stock_gbdt_s123_backtest 全量引擎 (import 触发重跑, 产出 bt.results)
画:
  - ENS_T40_S123          (进取版: CAGR 最高 11.50%)
  - ENS_T60_S123          (无TV: 11.40%)
  - ENS_T60_S123_TV12     (均衡版: Sharpe 0.84 最高, 推荐)
  - ETF原版 T7            (基准 6.15%)

产出: results/ens_best_oos_2023.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 复用主引擎 (重跑一次, 产出所有组合的 nav 与 T7 月频 nav)
import stock_gbdt_s123_backtest as bt

OUT = os.path.join(ROOT, "research", "sector_rotation", "results", "ens_best_oos_2023.png")

# T7 月频 NAV → 转日频对齐 (用其月末值重采样)
t7_nav = bt.t7["nav"]
t7_nav.index = [str(i) for i in t7_nav.index]
# 月频 ym (如 202301) → 月末日期 int
t7_daily = {}
for ym in t7_nav.index:
    y, m = int(ym[:4]), int(ym[4:6])
    last_day = (pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)).day
    t7_daily[y * 10000 + m * 100 + last_day] = t7_nav[ym]
t7_daily = pd.Series(t7_daily).sort_index()

KEYS = [
    ("ENS_T40_S123_ONLY_S123", "ENS_T40_S123 (进取 11.50%)", "darkorange", "-"),
    ("ENS_T60_S123_ONLY_S123", "ENS_T60_S123 (无TV 11.40%)", "purple", "-"),
    ("ENS_T60_S123_TV12",      "ENS_T60_S123_TV12 (均衡 8.21%, Sharpe 0.84)", "crimson", "-"),
]

# 2023-01 起切片
START = 20230101
fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1.2]})
ax = axes[0]

# 主引擎 nav 无日期 index → 从 log 的 date 字段重建
def nav_with_date(tag):
    log = bt.results[tag]["log"]
    return pd.Series(log["nav"].values, index=log["date"].values).sort_index()

lines = []
for tag, label, color, ls in KEYS:
    nav = nav_with_date(tag)
    nav = nav[nav.index >= START]
    nav = nav / nav.iloc[0]
    l, = ax.plot(nav.index, nav.values, label=label, color=color,
                 lw=1.5, ls=ls)
    lines.append(l)
    # 标注最终累计收益
    tot = nav.iloc[-1] - 1
    ax.annotate(f"{tot:+.0%}", xy=(nav.index[-1], nav.iloc[-1]),
                xytext=(8, 4), textcoords="offset points", fontsize=9, color=color)

# T7 基准 (2023-01 后, 月末点, x=日期int 与日频曲线同一数值坐标系)
t7 = t7_daily[t7_daily.index >= START]
t7 = t7 / t7.iloc[0]
l7, = ax.plot(t7.index, t7.values, label="ETF原版 T7 (6.15%, 月频)",
              color="darkgreen", lw=2.0, ls="--")
tot = t7.iloc[-1] - 1
ax.annotate(f"{tot:+.0%}", xy=(t7.index[-1], t7.iloc[-1]),
            xytext=(8, 4), textcoords="offset points", fontsize=9, color="darkgreen")

# 中证1000指数基准 (日频)
ix = bt.load_index_ret("000852.SH")
ix.index = ix.index.astype(int)
ix = ix[ix.index >= START].cumprod() + 1
ix = ix / ix.iloc[0]
lix, = ax.plot(ix.index, ix.values, label="中证1000 指数 (买入持有)",
               color="gray", lw=1.0, ls=":")
tot = ix.iloc[-1] - 1
ax.annotate(f"{tot:+.0%}", xy=(ix.index[-1], ix.iloc[-1]),
            xytext=(8, 4), textcoords="offset points", fontsize=9, color="gray")

ax.set_title("方案B 股票选股 OOS 收益曲线 (2023-01 ~ 2026-08, 回测)", fontsize=13)
ax.set_ylabel("累计净值 (起点=1)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)

# 回撤图
ax2 = axes[1]
for tag, label, color, ls in KEYS:
    nav = nav_with_date(tag)
    nav = nav[nav.index >= START]
    dd = nav / nav.cummax() - 1
    ax2.plot(nav.index, dd.values, color=color, lw=1.2, ls=ls)
ax2.plot(t7.index, (t7 / t7.cummax() - 1).values, color="darkgreen", lw=1.6, ls="--")
ax2.set_title("回撤对比 (2023-01 起)", fontsize=11)
ax2.set_ylabel("Drawdown")
ax2.grid(alpha=0.3)

# 精简 x 轴刻度 (每年一个, x=日期int数值)
nav0 = nav_with_date(KEYS[0][0])
nav0 = nav0[nav0.index >= START]
year_pos, year_labels = {}, {}
for d in nav0.index:
    y = d // 10000
    if y not in year_pos:
        year_pos[y] = d
        year_labels[y] = str(y)
plt.xticks(list(year_pos.values()), list(year_labels.values()), rotation=45)

plt.tight_layout()
plt.savefig(OUT, dpi=130)
print(f"[图] 已保存: {OUT}")

# 终端打印 2023-01 起各组合指标
print("\n=== 2023-01 起 OOS 指标 ===")
def oos_stats(tag):
    nav = nav_with_date(tag)
    nav = nav[nav.index >= START]
    tot = nav.iloc[-1] / nav.iloc[0] - 1
    yrs = len(nav) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    sh = nav.pct_change().fillna(0).mean() / (nav.pct_change().fillna(0).std() + 1e-8) * np.sqrt(242)
    return ann, dd, sh
for tag, label, _, _ in KEYS:
    a, d, s = oos_stats(tag)
    print(f"  {label:<36} CAGR={a:>7.2%} MaxDD={d:>7.2%} Sharpe={s:>5.2f}")

# T7 月频 OOS 统计
t7s = t7_daily[t7_daily.index >= START]
t7s = t7s / t7s.iloc[0]
tot = t7s.iloc[-1] - 1
yrs = len(t7s) / 12.0
a = (1 + tot) ** (1 / yrs) - 1
d = (t7s / t7s.cummax() - 1).min()
s = t7s.pct_change().fillna(0).mean() / (t7s.pct_change().fillna(0).std() + 1e-8) * np.sqrt(12)
print(f"  {'ETF原版 T7':<36} CAGR={a:>7.2%} MaxDD={d:>7.2%} Sharpe={s:>5.2f}")
