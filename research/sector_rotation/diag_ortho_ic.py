# -*- coding: utf-8 -*-
"""正交化 IC 诊断: 候选因子对现有 GBDT 价量特征的残差, 再算对 fwd_20 的 Rank IC。

判断候选因子是否有"独立于现有特征"的增量信息(避免双重暴露)。
口径: 逐月横截面 OLS 残差化, 再 spearman(残差, fwd_20)。
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

ROOT = r"c:\Users\liuqi\quant_system_v2"
panel = pd.read_parquet(os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet"))

# 现有 GBDT 价量基础特征(标准化前, 正交化基准)
BASE = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012"]
# 候选因子(算其残差 IC)
CANDIDATES = ["chip_conc_20", "prof_pct_20", "pos_vol_20", "f_rev", "f_ivol",
              "alpha_009", "alpha_023", "momentum_60", "volatility_10"]

def winsorize(s):
    s = s.astype(float)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)

def zscore(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)

rows = []
for m in sorted(panel["trade_date"].unique()):
    g = panel[panel["trade_date"] == m].copy()
    if len(g) < 50:
        continue
    y = g["fwd_20"].values
    # 标准化基准特征
    Xb = np.column_stack([zscore(winsorize(g[c])) for c in BASE])
    Xb = np.nan_to_num(Xb, nan=0.0)
    for c in CANDIDATES:
        x = winsorize(g[c]).values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 50:
            continue
        Xm = np.column_stack([np.ones(mask.sum()), Xb[mask]])
        lr = LinearRegression().fit(Xm, x[mask])
        resid = x[mask] - lr.predict(Xm)
        ic, _ = spearmanr(resid, y[mask])
        rows.append({"month": m, "factor": c, "resid_ic": ic})

r = pd.DataFrame(rows)
print(f"{'因子':<16} {'N':>4} {'残差IC均值':>10} {'残差ICIR':>9} {'IC>0':>6}")
print("-" * 52)
for c in CANDIDATES:
    s = r[r["factor"] == c]["resid_ic"]
    if len(s) == 0:
        continue
    icir = s.mean() / (s.std(ddof=1) + 1e-12)
    print(f"{c:<16} {len(s):>4} {s.mean():>10.4f} {icir:>9.2f} {(s>0).mean():>6.2f}")
