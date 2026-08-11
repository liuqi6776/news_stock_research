# -*- coding: utf-8 -*-
"""C8 + 估值正交因子 + 行业/资金流 的 WFO 对比 (sklearn MLP, 复现 C8_MLP_deep IC=+0.1098)

检验: 估值因子(基本面正交) 是否给 C8_MLP 带来增量
组合:
  C8 (10)                    —— baseline (应复现 +0.1098)
  C8+VAL (20)                —— 估值因子 (pe/pb/市值/换手率)
  C8+IND+VAL (23)            —— 行业+估值
  C8+IND+VAL+MF (27)         —— 全部正交
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

ROOT = r"c:\Users\liuqi\quant_system_v2"
t0 = time.time()

ORTHO_MF = ["net_mf_ratio_5","lg_net_ratio_5","net_mf_ratio_20","lg_net_ratio_20"]
ORTHO_IND = ["ind_mom_20","ind_crowd_20","ind_mf_20"]
VAL_FEATS = ["pe_ep","ln_circ_mv","pe_rank","pb_rank","ln_mv_rank",
             "turn_rank","volratio_rank","pe_pct_3y","pb_pct_3y","turn_pct_3y"]
C8_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012",
           "enh4_score","vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]

# === 面板 ===
panel_ortho = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_ortho2_72m.parquet"))
feats_all = ORTHO_MF + ORTHO_IND + VAL_FEATS
ortho_part = panel_ortho[["trade_date","ts_code","fwd_20"] + feats_all].copy()
for f in feats_all:
    if f in ortho_part.columns:
        ortho_part[f] = ortho_part.groupby("trade_date")[f].transform(
            lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))

orig = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_72m.parquet"))
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)
for c in ["roe","or_yoy"]:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
orig["roe"] = orig["roe"].fillna(-99.0); orig["or_yoy"] = orig["or_yoy"].fillna(-99.0)
gg = orig.groupby("trade_date")
orig["enh4_score"] = (-0.40*gg["ivol"].rank(pct=True) -0.35*gg["ret_1m"].rank(pct=True)
                      +0.15*gg["roe"].rank(pct=True) +0.05*gg["or_yoy"].rank(pct=True))
C7_RAW = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
CHIP_RAW = ["vwap_20","float_pnl_20","chip_shift_5"]
for c in C7_RAW + CHIP_RAW:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    orig[c] = orig.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))
orig["fwd_20"] = orig.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
orig = orig.dropna(subset=["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012",
                           "enh4_score","vwap_20","float_pnl_20","chip_shift_5","fwd_20"])
for c in CHIP_RAW: orig[f"{c}_resid"] = np.nan
for dt, grp in orig.groupby("trade_date"):
    if len(grp) < 50: continue
    Xb = grp[C7_RAW].values
    for c in CHIP_RAW:
        y = grp[c].values
        mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
        if mask.sum() < 50: continue
        lr = LinearRegression().fit(Xb[mask], y[mask])
        orig.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
for c in ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    orig[c] = orig.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))
c8_part = orig[["trade_date","ts_code"] + C8_COLS].dropna(subset=C8_COLS)

panel = ortho_part.merge(c8_part, on=["trade_date","ts_code"], how="inner")
for f in feats_all:
    if f in panel.columns: panel[f] = panel[f].fillna(0.0)
panel = panel.dropna(subset=C8_COLS + ["fwd_20"])
oos_months = sorted([m for m in panel["trade_date"].unique() if m >= 20230101])
print(f"[面板] {len(panel):,} 行, OOS {len(oos_months)} 月, {time.time()-t0:.0f}s", flush=True)

# === WFO ===
def mic(df):
    out = {}
    for dt, gg in df.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg["pred"].rank().corr(gg["fwd_20"].rank())
    s = pd.Series(out).dropna()
    return s.mean(), s.std(ddof=1), (s>0).mean()

def wfo(feats, factory, need_scale=False, label=""):
    t1 = time.time(); preds = []
    for m_idx, m in enumerate(oos_months):
        tr = panel[panel["trade_date"] < m]
        if tr["trade_date"].nunique() < 24: continue
        vm = tr["trade_date"].isin(sorted(tr["trade_date"].unique())[-3:]).values
        Xa, ya = tr[feats].values, tr["fwd_20"].values
        Xtr, ytr, Xv, yv = Xa[~vm], ya[~vm], Xa[vm], ya[vm]
        sc = StandardScaler() if need_scale else None
        if need_scale: Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv)
        mdl = factory(Xtr, ytr, Xv, yv)
        tt = panel[panel["trade_date"] == m]
        Xt = tt[feats].values
        if need_scale: Xt = sc.transform(Xt)
        preds.append(pd.DataFrame({"trade_date":m,"pred":mdl.predict(Xt),
                                   "fwd_20":tt["fwd_20"].values}))
        if (m_idx+1) % 6 == 0: print(f"    {label} WFO {m_idx+1}/{len(oos_months)}, {time.time()-t1:.0f}s", flush=True)
    df = pd.concat(preds, ignore_index=True)
    ic, std, pos = mic(df)
    icir = ic / (std + 1e-12) * np.sqrt(12)
    df["q"] = df.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df.groupby("q", observed=True)["fwd_20"].mean()
    return {"ic": ic, "icir": icir, "pos_rate": pos,
            "q1q5": piv["Q1"]-piv["Q5"], "sec": int(time.time()-t1), "nfeat": len(feats)}

def mlp_deep(Xtr, ytr, Xv, yv):
    m = MLPRegressor((256,128,64,32,16), activation="relu", solver="adam", alpha=0.5,
        batch_size=512, learning_rate_init=0.001, max_iter=200,
        early_stopping=True, n_iter_no_change=12, validation_fraction=0.1, random_state=42)
    m.fit(Xtr, ytr)
    return m

COMBO = {
    "C8(baseline)":   C8_COLS,
    "C8+VAL":         C8_COLS + VAL_FEATS,
    "C8+IND+VAL":     C8_COLS + ORTHO_IND + VAL_FEATS,
    "C8+IND+VAL+MF":  C8_COLS + ORTHO_IND + VAL_FEATS + ORTHO_MF,
}
rows = []
for label, feats in COMBO.items():
    feats = [c for c in feats if c in panel.columns]
    print(f"\n>> {label} / MLP_deep ({len(feats)}因子)...", flush=True)
    r = wfo(feats, mlp_deep, need_scale=True, label=f"{label[:6]}")
    print(f"  IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s", flush=True)
    rows.append({"label": f"{label}/MLP_deep", **r})

print(f"\n{'='*86}", flush=True)
print(f"{'模型':>34} {'因子数':>5} {'IC':>8} {'ICIR':>7} {'Q1-Q5':>7} {'正率':>5} {'耗时':>5}")
print(f"{'-'*86}")
for r in sorted(rows, key=lambda x: x["ic"], reverse=True):
    print(f"{r['label']:>34} {r['nfeat']:>5} {r['ic']:+.4f}  {r['icir']:+.2f}   {r['q1q5']:+.2f}   {r['pos_rate']*100:.0f}%  {r['sec']}s")
print(f"{'='*86}", flush=True)

OUT = os.path.join(ROOT, "research/sector_rotation/results/c8_val_ortho_ic.csv")
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"[保存] {OUT}\n[总耗时] {time.time()-t0:.0f}s", flush=True)
