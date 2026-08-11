# -*- coding: utf-8 -*-
"""快速验证: C8 vs C6 GBDT OOS IC（只跑 GBDT，不用 MLP/DL，2分钟内出结果）"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "results", "c8_vs_c6_gbdt_ic.csv")

C7_COLS   = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]
CHIP_COLS = ["vwap_20","float_pnl_20","chip_shift_5"]
CHIP_RESID = ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
BASE_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
C8_COLS = C7_COLS + CHIP_RESID
C6_COLS = [c for c in C8_COLS if c not in ("momentum_20","volatility_20")]
# 极端版 C4（只留 VIF<5 的安全因子 + 2个核心 alpha）
VIF_SAFE = ["alpha_006","alpha_012","float_pnl_20_resid","chip_shift_5_resid","vwap_20_resid"]
# + enh4_score（VIF 6.03 轻度）+ ivol（强信号 IC -0.11，虽 VIF=16 但和 enh4_score -0.83 互有增量）
C4_COLS = VIF_SAFE + ["enh4_score","ivol","ret_1m"]

def zscore(s): return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

t0 = time.time()
panel = pd.read_parquet(PANEL)
for c in ["roe","or_yoy"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["roe"] = panel["roe"].fillna(-99.0)
panel["or_yoy"] = panel["or_yoy"].fillna(-99.0)
g = panel.groupby("trade_date")
panel["enh4_score"] = (-0.40 * g["ivol"].rank(pct=True) -0.35 * g["ret_1m"].rank(pct=True)
                       +0.15 * g["roe"].rank(pct=True) +0.05 * g["or_yoy"].rank(pct=True))
ALL_RAW = list(set(C7_COLS + CHIP_COLS) - {"enh4_score"})
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=C7_COLS + CHIP_COLS + ["fwd_20"])
for c in CHIP_COLS: panel[f"{c}_resid"] = np.nan
for dt, grp in panel.groupby("trade_date"):
    if len(grp) < 50: continue
    Xb = grp[BASE_COLS].values
    for c in CHIP_COLS:
        y = grp[c].values
        mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
        if mask.sum() < 50: continue
        lr = LinearRegression(fit_intercept=True)
        lr.fit(Xb[mask], y[mask])
        panel.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
for c in CHIP_RESID:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel = panel.dropna(subset=C8_COLS + ["fwd_20"])
months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]
print(f"[数据] {len(panel):,} 行, {len(months)} 月, OOS {len(oos_months)} 月, {time.time()-t0:.0f}s")

def mic(d, f, ret="fwd_20"):
    out = {}
    for dt, gg in d.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg[f].rank().corr(gg[ret].rank())
    return pd.Series(out).dropna()

def wfo(feats, depth=3, nl=7, mc=80, rl=2.0):
    t1 = time.time()
    preds = []
    for m in oos_months:
        tr = panel[panel["trade_date"] < m]
        if tr["trade_date"].nunique() < 24: continue
        vm = tr["trade_date"].isin(sorted(tr["trade_date"].unique())[-3:]).values
        Xa, ya = tr[feats].values, tr["fwd_20"].values
        Xtr, ytr, Xv, yv = Xa[~vm], ya[~vm], Xa[vm], ya[vm]
        mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                num_leaves=nl, max_depth=depth,
                                min_child_samples=mc, reg_lambda=rl, reg_alpha=0.1,
                                subsample=0.9, colsample_bytree=0.9,
                                random_state=42, verbose=-1)
        mdl.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        tt = panel[panel["trade_date"] == m]
        preds.append(pd.DataFrame({"trade_date":m,"ts_code":tt["ts_code"],
                                   "pred":mdl.predict(tt[feats].values),
                                   "fwd_20":tt["fwd_20"].values}))
    df = pd.concat(preds, ignore_index=True)
    ics = mic(df, "pred")
    icir = ics.mean() / (ics.std(ddof=1)+1e-12) * np.sqrt(12)
    df["q"] = df.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"),5,labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df.groupby("q", observed=True)["fwd_20"].mean()
    imp = pd.Series(mdl.feature_importances_, index=feats).sort_values(ascending=False)
    return {"ic":ics.mean(), "icir":icir, "pos":(ics>0).mean(),
            "q1q5":piv["Q1"]-piv["Q5"], "sec":int(time.time()-t1),
            "top_feats":" | ".join(f"{k}={v}" for k,v in imp.head(5).items()),
            "nfeat":len(feats)}

print("\n=== C8 vs C6 vs C4 GBDT d3 OOS WFO ===")
rows = []
for label, feats, dp, nl, mc, rl in [
    ("C8_d3 (baseline)",          C8_COLS, 3, 7,   80, 2.0),
    ("C6_d3 (去冗余: -mom20-vol20)", C6_COLS, 3, 7,   80, 2.0),
    ("C6_d5 (去冗余+稍深树)",       C6_COLS, 5, 16, 100, 2.5),
    ("C4_d3 (VIF安全+核心)",        C4_COLS, 3, 7,   80, 2.0),
]:
    r = wfo(feats, depth=dp, nl=nl, mc=mc, rl=rl)
    print(f"  {label:>20} n={r['nfeat']} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} "
          f"Q1-Q5={r['q1q5']:+.2f} 正率={r['pos']*100:.0f}% {r['sec']}s")
    print(f"      TopFeats: {r['top_feats']}")
    rows.append({"label": label, **r})

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
base = rows[0]
best = max(rows, key=lambda x:x["ic"])
print(f"\n🏆 最优: {best['label']} IC={best['ic']:+.4f}")
print(f"   vs C8 baseline: ΔIC={best['ic']-base['ic']:+.4f} (正=去冗余有效)")
if best["label"] != base["label"] and (best["ic"]-base["ic"]) >= 0.002:
    print(f"   ✅ 建议: 采用 {best['label']} 替代 C8，更新回测引擎")
elif (best["ic"]-base["ic"]) >= -0.003:
    print(f"   ✅ 持平: 排除冗余因子后 IC 不掉（Δ<0.003）→ 可以用简化版替代")
else:
    print(f"   ⚠️ 去冗余反而掉 IC {best['ic']-base['ic']:+.4f} → 保留 C8")
print(f"\n[保存] {OUT} | 总耗时 {time.time()-t0:.0f}s")
