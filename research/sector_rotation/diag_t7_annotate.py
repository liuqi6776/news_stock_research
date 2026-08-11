# -*- coding: utf-8 -*-
"""T7 策略净值图 + 进出场标注"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fund_research", "studies", "rotation_dingtou"))

from etf_optimize_backtest2 import (  # noqa: E402
    INDUSTRY_ETFS, load_industry_daily, load_hv_daily, build_series,
    hv_monthly_ret, monthly_from_daily, calc_stats, COST, OUT_DIR,
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

nv = run_graded(nav_panel, sig, plain_trad_m, v8_m, use_v8=True, mode="strict",
                entry_sig=3, exit_sig=1, sig_col="s123")

# 进出场点 (信号月 -> 下月生效)
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

fig, ax = plt.subplots(figsize=(15, 7))
x = np.arange(len(nv))
ax.plot(x, nv["nav"], lw=2.0, color="#2ca02c", label=f"T7 策略净值 ({nv['nav'].iloc[-1]:.2f})")
base = (1 + plain_trad_m).cumprod().reindex(nv.index).ffill()
ax.plot(x, base, lw=1.2, color="#7f7f7f", ls="--", label=f"传统行业等权基准 ({base.iloc[-1]:.2f})")

y_min, y_max = min(nv["nav"].min(), base.min()), max(nv["nav"].max(), base.max())
for ym, act in events:
    idx = list(nv.index).index(ym)
    color = "#E8463A" if act == "进" else "#1f77b4"
    ax.axvline(idx, color=color, alpha=0.55, lw=1.4, ls="--")
    ax.text(idx, y_max * 1.02, f"{ym[0:4]}-{ym[4:6]}\n{act}",
            ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

st = calc_stats(nv)
ax.set_title(f"T7 传统行业+4信号低吸高抛: 全期 CAGR={st['CAGR']:.2%} MaxDD={st['MaxDD']:.2%} "
             f"Calmar={st['Calmar']:.2f} (2015-2026, 红=进/蓝=出)", fontsize=13)
ax.set_ylabel("NAV")
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.3)
year_ticks = [i for i, ym in enumerate(nv.index) if ym.endswith("01")]
year_labels = [ym[:4] for ym in nv.index if ym.endswith("01")]
ax.set_xticks(year_ticks, year_labels)
fig.tight_layout()
png = os.path.join(OUT_DIR, "traditional_t7_curve_annotated.png")
fig.savefig(png, dpi=150, bbox_inches="tight")
print(f"[saved] {png}")
