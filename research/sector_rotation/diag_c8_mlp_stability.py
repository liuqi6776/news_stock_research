# -*- coding: utf-8 -*-
"""C8_MLP_deep 稳定性验证: 多 seed + 多模型配置 WFO IC"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

ROOT = r"c:\Users\liuqi\quant_system_v2"
C8_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012",
           "enh4_score","vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]

def zscore(s): return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

t0 = time.time()
orig = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_72m.parquet"))
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
    orig[c] = orig.groupby("trade_date")[c].transform(zscore)
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
    orig[c] = orig.groupby("trade_date")[c].transform(zscore)
orig = orig.dropna(subset=C8_COLS + ["fwd_20"])
oos_months = sorted([m for m in orig["trade_date"].unique() if m >= 20230101])
print(f"[数据] {len(orig):,} 行, OOS {len(oos_months)} 月, {time.time()-t0:.0f}s")

def mic(df):
    out = {}
    for dt, gg in df.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg["pred"].rank().corr(gg["fwd_20"].rank())
    s = pd.Series(out).dropna()
    return s.mean(), s.std(ddof=1), (s>0).mean()

def wfo_seed(factory_fn, label):
    """factory_fn(Xtr,ytr,Xv,yv,seed) -> model"""
    t1 = time.time(); preds = []
    for m in oos_months:
        tr = orig[orig["trade_date"] < m]
        if tr["trade_date"].nunique() < 24: continue
        vm = tr["trade_date"].isin(sorted(tr["trade_date"].unique())[-3:]).values
        Xa, ya = tr[C8_COLS].values, tr["fwd_20"].values
        Xtr, ytr, Xv, yv = Xa[~vm], ya[~vm], Xa[vm], ya[vm]
        mdl = factory_fn(Xtr, ytr, Xv, yv)
        tt = orig[orig["trade_date"] == m]
        preds.append(pd.DataFrame({"trade_date":m,"pred":mdl.predict(tt[C8_COLS].values),
                                   "fwd_20":tt["fwd_20"].values}))
    df = pd.concat(preds, ignore_index=True)
    ic, std, pos = mic(df)
    icir = ic / (std + 1e-12) * np.sqrt(12)
    return ic, icir, pos, int(time.time()-t1), df

def mlp_fac(hidden, alpha, max_iter, seed):
    def _f(Xtr, ytr, Xv, yv):
        m = MLPRegressor(hidden, activation="relu", solver="adam", alpha=alpha,
            batch_size=512, learning_rate_init=0.001, max_iter=max_iter,
            early_stopping=True, n_iter_no_change=12, validation_fraction=0.1,
            random_state=seed)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xv_s = sc.transform(Xv)
        m.fit(Xtr_s, ytr)
        class _P:
            def __init__(self, m, sc): self.m, self.sc = m, sc
            def predict(self, X): return self.m.predict(self.sc.transform(X))
        return _P(m, sc)
    return _f

print(f"\n=== C8 MLP 稳定性验证 (multi-seed) ===")
rows = []

# 1. deep5 多 seed
for seed in [42, 0, 2024]:
    ic, icir, pos, sec, _ = wfo_seed(mlp_fac((256,128,64,32,16), 0.5, 200, seed), f"seed{seed}")
    print(f"  MLP_deep5 seed={seed:>5} | IC={ic:+.4f} ICIR={icir:+.2f} 正率={pos*100:.0f}% {sec}s")
    rows.append({"model":f"MLP_deep5_s{seed}", "ic":ic, "icir":icir, "pos":pos})

# 2. mid 多 seed
for seed in [42, 0]:
    ic, icir, pos, sec, _ = wfo_seed(mlp_fac((128,64,32), 0.3, 200, seed), f"seed{seed}")
    print(f"  MLP_mid   seed={seed:>5} | IC={ic:+.4f} ICIR={icir:+.2f} 正率={pos*100:.0f}% {sec}s")
    rows.append({"model":f"MLP_mid_s{seed}", "ic":ic, "icir":icir, "pos":pos})

# 3. 更深的 6层
ic, icir, pos, sec, _ = wfo_seed(mlp_fac((512,256,128,64,32,16), 0.8, 200, 42), "s42")
print(f"  MLP_6layer seed=  42 | IC={ic:+.4f} ICIR={icir:+.2f} 正率={pos*100:.0f}% {sec}s")
rows.append({"model":"MLP_6layer", "ic":ic, "icir":icir, "pos":pos})

# 4. 小 alpha 更少正则
ic, icir, pos, sec, _ = wfo_seed(mlp_fac((256,128,64,32,16), 0.1, 200, 42), "s42")
print(f"  MLP_deep5_a01 seed=42 | IC={ic:+.4f} ICIR={icir:+.2f} 正率={pos*100:.0f}% {sec}s")
rows.append({"model":"MLP_deep5_a01", "ic":ic, "icir":icir, "pos":pos})

# 5. GBDT baseline 同口径
def gbdt_d3(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
        max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
        subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m
ic, icir, pos, sec, dfg = wfo_seed(gbdt_d3, "gbdt")
print(f"  GBDT_d3 (baseline) | IC={ic:+.4f} ICIR={icir:+.2f} 正率={pos*100:.0f}% {sec}s")
rows.append({"model":"GBDT_d3", "ic":ic, "icir":icir, "pos":pos})

# 6. 最优 MLP 与 GBDT 的月IC序列相关性 (看是否互补)
ic42, _, _, _, dfm = wfo_seed(mlp_fac((256,128,64,32,16), 0.5, 200, 42), "s42")
m_ic = dfm.groupby("trade_date").apply(lambda x: x["pred"].rank().corr(x["fwd_20"].rank()), include_groups=False)
g_ic = dfg.groupby("trade_date").apply(lambda x: x["pred"].rank().corr(x["fwd_20"].rank()), include_groups=False)
aligned = pd.concat([m_ic.rename("mlp"), g_ic.rename("gbdt")], axis=1).dropna()
print(f"\n  月IC序列: MLP vs GBDT 相关 = {aligned['mlp'].corr(aligned['gbdt']):+.3f}")
print(f"  月IC: MLP 均值{aligned['mlp'].mean():+.4f} GBDT 均值{aligned['gbdt'].mean():+.4f}")
print(f"  MLP 胜出月份: {(aligned['mlp']>aligned['gbdt']).mean()*100:.0f}%")

print(f"\n{'='*70}")
for r in sorted(rows, key=lambda x: x["ic"], reverse=True):
    print(f"  {r['model']:>20} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} 正率={r['pos']*100:.0f}%")
print(f"{'='*70}")

OUT = os.path.join(ROOT, "research/sector_rotation/results/c8_mlp_stability.csv")
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n[保存] {OUT}, 总耗时 {time.time()-t0:.0f}s")
