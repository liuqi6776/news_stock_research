# -*- coding: utf-8 -*-
"""多seed验证 C8+IND+VAL (23因子) vs C8 baseline (10因子) 的稳定性

用 sklearn MLP_deep 跑 3 个 seed (42/0/2024), 检验 IC 提升是否稳健
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
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

# === 面板 (与 diag_c8_val_ortho.py 相同构建) ===
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

def mic(df):
    out = {}
    for dt, gg in df.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg["pred"].rank().corr(gg["fwd_20"].rank())
    s = pd.Series(out).dropna()
    return s.mean(), s.std(ddof=1), (s>0).mean()

def wfo(feats, seed, label=""):
    t1 = time.time(); preds = []
    for m in oos_months:
        tr = panel[panel["trade_date"] < m]
        if tr["trade_date"].nunique() < 24: continue
        vm = tr["trade_date"].isin(sorted(tr["trade_date"].unique())[-3:]).values
        Xa, ya = tr[feats].values, tr["fwd_20"].values
        Xtr, ytr, Xv, yv = Xa[~vm], ya[~vm], Xa[vm], ya[vm]
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv)
        m_ = MLPRegressor((256,128,64,32,16), activation="relu", solver="adam", alpha=0.5,
            batch_size=512, learning_rate_init=0.001, max_iter=200,
            early_stopping=True, n_iter_no_change=12, validation_fraction=0.1, random_state=seed)
        m_.fit(Xtr, ytr)
        tt = panel[panel["trade_date"] == m]
        preds.append(pd.DataFrame({"trade_date":m,"pred":m_.predict(sc.transform(tt[feats].values)),
                                   "fwd_20":tt["fwd_20"].values}))
    df = pd.concat(preds, ignore_index=True)
    ic, std, pos = mic(df)
    icir = ic / (std + 1e-12) * np.sqrt(12)
    print(f"  {label} seed={seed}: IC={ic:+.4f} ICIR={icir:+.2f} 正率={pos*100:.0f}% {int(time.time()-t1)}s", flush=True)
    return ic, icir

SEEDS = [42, 0, 2024]
rows = []
for label, feats in [("C8(baseline)", C8_COLS), ("C8+IND+VAL", C8_COLS+ORTHO_IND+VAL_FEATS)]:
    feats = [c for c in feats if c in panel.columns]
    ics = []
    for sd in SEEDS:
        ic, icir = wfo(feats, sd, label=label)
        ics.append(ic)
    rows.append({"label": label, "nfeat": len(feats),
                 "seed42": ics[0], "seed0": ics[1], "seed2024": ics[2],
                 "mean": np.mean(ics), "std": np.std(ics), "min": np.min(ics)})
    print(f"  [{label}] mean IC={np.mean(ics):+.4f} std={np.std(ics):.4f} min={np.min(ics):+.4f}", flush=True)

print(f"\n{'='*72}", flush=True)
print(f"{'模型':>20} {'因子':>4} {'seed42':>8} {'seed0':>8} {'seed2024':>8} {'mean':>8} {'min':>8}")
print(f"{'-'*72}")
for r in rows:
    print(f"{r['label']:>20} {r['nfeat']:>4} {r['seed42']:+.4f}  {r['seed0']:+.4f}  {r['seed2024']:+.4f}   {r['mean']:+.4f}  {r['min']:+.4f}")
print(f"{'='*72}", flush=True)

OUT = os.path.join(ROOT, "research/sector_rotation/results/c8_indval_stability.csv")
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"[保存] {OUT}\n[总耗时] {time.time()-t0:.0f}s", flush=True)
