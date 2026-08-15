# -*- coding: utf-8 -*-
"""诊断 panel 内已算好但 GBDT 未直接使用的因子的 Rank IC + ICIR。

口径对齐 backtest_undervalued_sector_stock.py：
  - 月度横截面 Rank IC = spearman(factor_t, fwd_20_t)
  - fwd_20 为未来20日收益标签(已无前视)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = r"c:\Users\liuqi\quant_system_v2"
panel = pd.read_parquet(os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet"))

# GBDT 已用(对照组)
USED = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012"]
# panel 里已算好但 GBDT 未直接使用(候选扩充源)
CANDIDATES = [
    "alpha_009", "alpha_023",
    "momentum_5", "momentum_10", "momentum_60",
    "volatility_5", "volatility_10",
    "roe", "or_yoy", "netprofit_yoy",
    "vwap_20", "float_pnl_20", "chip_shift_5",
    "prof_pct_20", "chip_conc_20", "pos_vol_20",
    "f_rev", "f_ivol",
]

rows = []
months = sorted(panel["trade_date"].unique())
for m in months:
    g = panel[panel["trade_date"] == m]
    if len(g) < 50:
        continue
    y = g["fwd_20"]
    for f in USED + CANDIDATES:
        x = g[f]
        mask = x.notna() & y.notna()
        if mask.sum() < 30:
            continue
        ic, _ = spearmanr(x[mask], y[mask])
        rows.append({"month": m, "factor": f, "ic": ic})

ic_df = pd.DataFrame(rows)
g = ic_df.groupby("factor")["ic"]
summary = pd.DataFrame({
    "n": g.count(),
    "IC_mean": g.mean(),
    "IC_std": g.std(),
    "IC>0占比": g.apply(lambda s: (s > 0).mean()),
    "|IC|>0.03占比": g.apply(lambda s: (s.abs() > 0.03).mean()),
})
summary["ICIR"] = summary["IC_mean"] / summary["IC_std"]
summary = summary.round(4)

print("=" * 78)
print(f"{'因子':<16} {'N':>5} {'IC均值':>8} {'ICIR':>7} {'IC>0':>6} {'|IC|>0.03':>9}  类型")
print("-" * 78)
type_map = {f: "已用" for f in USED}
for f in CANDIDATES:
    if f.startswith("alpha_"):
        type_map[f] = "Alpha101"
    elif f.startswith("momentum") or f.startswith("volatility"):
        type_map[f] = "价量"
    elif f in ("roe", "or_yoy", "netprofit_yoy"):
        type_map[f] = "财务"
    elif f in ("f_rev", "f_ivol"):
        type_map[f] = "未知"
    else:
        type_map[f] = "筹码"
for f in USED + CANDIDATES:
    r = summary.loc[f]
    print(f"{f:<16} {int(r['n']):>5} {r['IC_mean']:>8.4f} {r['ICIR']:>7.2f} "
          f"{r['IC>0占比']:>6.2f} {r['|IC|>0.03占比']:>9.2f}  {type_map[f]}")
print("=" * 78)
print("ICIR 排序 (候选因子, 仅列 |ICIR|>=0.5):")
cand_sum = summary.loc[[f for f in CANDIDATES if f in summary.index]].copy()
cand_sum["abs_icir"] = cand_sum["ICIR"].abs()
cand_sum = cand_sum.sort_values("abs_icir", ascending=False)
for f, r in cand_sum.iterrows():
    if abs(r["ICIR"]) >= 0.5:
        print(f"  {f:<16} ICIR={r['ICIR']:>7.2f}  IC均值={r['IC_mean']:>8.4f}")
