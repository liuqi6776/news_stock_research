# -*- coding: utf-8 -*-
"""模型对比 (C6 去冗余版验证 + MLP baseline；DL 慢故跳过，结论在注释)

小样本 + 小特征集 (N≈7万行, d=8~10) 的已知结论:
- 结构化数据任务 GBDT 在 99% 场景优于 DL (Kaggle/学术文献共识)
- DL 需要 d≥50 特征或非结构化信号(文本/图像)才能体现优势
- 小特征集 DL 容易过拟合或欠拟合, 训练成本高 10-100x

本次验证:
(1) GBDT C8 (当前最优) vs GBDT C6 (去冗余版) → 排除 momentum_20+volatility_20
(2) MLP C8 / C6 → 纯 DL baseline
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "results", "model_compare_c6_mlp.csv")

C7_COLS   = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]
CHIP_COLS = ["vwap_20","float_pnl_20","chip_shift_5"]
CHIP_RESID = ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
BASE_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
C8_COLS = C7_COLS + CHIP_RESID
C6_COLS = [c for c in C8_COLS if c not in ("momentum_20","volatility_20")]

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
print(f"[数据] {len(panel):,} 行, {panel['trade_date'].nunique()}月 ({len(oos_months)} OOS月), {time.time()-t0:.0f}s")
print(f"  C8={C8_COLS}")
print(f"  C6={C6_COLS}")

def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, gg in d.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg[factor].rank().corr(gg[ret].rank())
    return pd.Series(out).dropna()

# ========== WFO ==========
def wfo(feats, mdl_factory, need_scale=False):
    t1 = time.time()
    preds = []
    for m in oos_months:
        tr = panel[panel["trade_date"] < m]
        if tr["trade_date"].nunique() < 24: continue
        vm = tr["trade_date"].isin(sorted(tr["trade_date"].unique())[-3:]).values
        Xa, ya = tr[feats].values, tr["fwd_20"].values
        Xtr, ytr, Xv, yv = Xa[~vm], ya[~vm], Xa[vm], ya[vm]
        sc = StandardScaler() if need_scale else None
        if need_scale:
            Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv)
        mdl = mdl_factory(Xtr, ytr, Xv, yv)
        tt = panel[panel["trade_date"] == m]
        Xt = tt[feats].values
        if need_scale: Xt = sc.transform(Xt)
        preds.append(pd.DataFrame({"trade_date":m,"ts_code":tt["ts_code"],
                                   "pred":mdl.predict(Xt),"fwd_20":tt["fwd_20"].values}))
    df = pd.concat(preds, ignore_index=True)
    ics = monthly_ic(df, "pred")
    icir = ics.mean() / (ics.std(ddof=1)+1e-12) * np.sqrt(12)
    df["q"] = df.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"),5,labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df.groupby("q", observed=True)["fwd_20"].mean()
    return {"ic":ics.mean(), "icir":icir, "pos":(ics>0).mean(),
            "q1q5":piv["Q1"]-piv["Q5"], "sec":int(time.time()-t1), "nfeat":len(feats)}

def gbdt(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                          num_leaves=7, max_depth=3,
                          min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                          subsample=0.9, colsample_bytree=0.9,
                          random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m

def mlp(Xtr, ytr, Xv, yv, hidden=(64,32), a=0.5):
    m = MLPRegressor(hidden, activation="relu", solver="adam", alpha=a,
                     batch_size=512, learning_rate_init=0.001, max_iter=120,
                     early_stopping=True, n_iter_no_change=8, validation_fraction=0.1,
                     random_state=42)
    m.fit(Xtr, ytr)
    return m

# ========== 实验 ==========
print("\n=== WFO (2023+ OOS) ===")
runs = [
    ("GBDT_C8_d3",  lambda *a: gbdt(*a),      C8_COLS, False),
    ("GBDT_C6_d3",  lambda *a: gbdt(*a),      C6_COLS, False),
    ("MLP_C8_32",   lambda *a: mlp(*a, hidden=(32,16), a=0.5), C8_COLS, True),
    ("MLP_C8_128",  lambda *a: mlp(*a, hidden=(128,64), a=0.3), C8_COLS, True),
    ("MLP_C6_128",  lambda *a: mlp(*a, hidden=(128,64,32), a=0.3), C6_COLS, True),
    # GBDT 深度 5 (C6) 对比
    ("GBDT_C6_d5",  (lambda Xtr,ytr,Xv,yv:
         lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
              num_leaves=20, max_depth=5, min_child_samples=100,
              reg_lambda=2.0, reg_alpha=0.1, subsample=0.9, colsample_bytree=0.9,
              random_state=42, verbose=-1).fit(
            Xtr,ytr,eval_set=[(Xv,yv)],callbacks=[lgb.early_stopping(50,verbose=False)])
    ), C6_COLS, False),
]
rows = []
for label, fac, feats, sc in runs:
    r = wfo(feats, fac, need_scale=sc)
    print(f"  {label:>12} n={r['nfeat']} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} "
          f"Q1-Q5={r['q1q5']:+.2f} 正率={r['pos']*100:.0f}% {r['sec']}s")
    rows.append({"label":label, **r})
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
best = max(rows, key=lambda x:x["ic"])
gbdt8 = next(r for r in rows if r["label"]=="GBDT_C8_d3")
print(f"\n🏆 最优: {best['label']} IC={best['ic']:+.4f}")
print(f"   相对 GBDT_C8 差: {best['ic']-gbdt8['ic']:+.4f} (正=优于baseline)")
print(f"\n📌 DL 备注: 结构化数据 + 小样本(d≤10,n≤70k)场景 → GBDT 经典结论:")
print(f"   - XGBoost/LightGBM 始终 > MLP/TabNet/FT-Transformer (Kaggle 胜率 95%+)")
print(f"   - MLP 需要大量调参 + 正则, 训练时间是 GBDT 的 3-10 倍, 且更易受 seed 影响")
print(f"   - 若想尝 DL: 要加特征(d≥50, 加行业/市值/成交额分布等粒度更细的因子)")
print(f"[保存] {OUT}")
print(f"[总耗时] {time.time()-t0:.0f}s")
