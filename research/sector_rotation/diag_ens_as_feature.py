# -*- coding: utf-8 -*-
"""诊断: ENS 能否作为 GBDT 特征 (两阶段 stacking, 无泄漏)

Phase 1: WFO C8 GBDT_v1 滚动预测所有月份 → gbdt1_pred
         ens_v1(t) = 0.5*ENH4_rank(t) + 0.5*rank(gbdt1_pred(t))
         其中 gbdt1_pred(t) 由训练于 t 之前的模型产生 → 无未来泄漏
Phase 2: 特征集对比 (OOS 2023+):
         C8 (baseline) vs C9 = C8 + ens_v1
         同时计算最终混合 ENS2 = 0.5*ENH4_rank + 0.5*rank(GBDT_v2_pred) 的 IC

输出: results/gbdt_ens_feature_diag.csv
"""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "results", "gbdt_ens_feature_diag.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C7_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]
CHIP_COLS = ["vwap_20","float_pnl_20","chip_shift_5"]
CHIP_RESID = ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
BASE_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
C8_COLS = C7_COLS + CHIP_RESID

t0 = time.time()
panel = pd.read_parquet(PANEL)

def zscore(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

# ENH4 打分
for c in ["roe","or_yoy"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["roe"] = panel["roe"].fillna(-99.0)
panel["or_yoy"] = panel["or_yoy"].fillna(-99.0)
g = panel.groupby("trade_date")
panel["enh4_score"] = (-0.40 * g["ivol"].rank(pct=True)
                       -0.35 * g["ret_1m"].rank(pct=True)
                       +0.15 * g["roe"].rank(pct=True)
                       +0.05 * g["or_yoy"].rank(pct=True))

# 标准化
ALL_RAW = list(set(C7_COLS + CHIP_COLS) - {"enh4_score"})
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=C7_COLS + CHIP_COLS + ["fwd_20"])

# 筹码残差化 (方向对齐取负)
for c in CHIP_COLS:
    panel[f"{c}_resid"] = np.nan
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
print(f"[数据] {len(panel):,} 行, {panel['trade_date'].nunique()}月, {time.time()-t0:.0f}s")

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]

def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, gg in d.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg[factor].rank().corr(gg[ret].rank())
    return pd.Series(out).dropna()

# ========== Phase 1: GBDT_v1 WFO 预测全部月份 → ens_v1 ==========
print("\n=== Phase 1: GBDT_v1 滚动预测全部月份 ===")
GBDT_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=7,
                   max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                   subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)

def wfo_predict_all(feat_cols, min_train=24):
    """expand-window WFO 预测所有月份 (每月 m 只用 m 之前数据训练)"""
    pred_list = []
    for i, m in enumerate(months):
        tr = panel[panel["trade_date"] < m]
        if tr["trade_date"].nunique() < min_train:
            pred_list.append(None); continue
        val_months = sorted(tr["trade_date"].unique())[-3:]
        vm = tr["trade_date"].isin(val_months).values
        X = tr[feat_cols].values; y = tr["fwd_20"].values
        mdl = lgb.LGBMRegressor(**GBDT_PARAMS)
        mdl.fit(X[~vm], y[~vm], eval_set=[(X[vm], y[vm])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        t_m = panel[panel["trade_date"] == m]
        pred_list.append(pd.DataFrame({"trade_date": m, "ts_code": t_m["ts_code"],
                                       "pred": mdl.predict(t_m[feat_cols])}))
    df = pd.concat([p for p in pred_list if p is not None], ignore_index=True)
    return df

t1 = time.time()
g1 = wfo_predict_all(C8_COLS)
print(f"  GBDT_v1 预测覆盖 {g1['trade_date'].nunique()} 个月, 耗时 {time.time()-t1:.0f}s")

# ens_v1 = 0.5*ENH4_rank + 0.5*rank(gbdt1_pred)   (逐月截面 rank)
g1["gbdt1_rank"] = g1.groupby("trade_date")["pred"].rank(pct=True)
panel = panel.merge(g1[["trade_date", "ts_code", "gbdt1_rank"]], on=["trade_date", "ts_code"], how="left")
panel["ens_v1"] = 0.5 * panel["enh4_score"] + 0.5 * panel["gbdt1_rank"]
# 标准化
panel["ens_v1"] = panel.groupby("trade_date")["ens_v1"].transform(lambda s: winsorize(s))
panel["ens_v1"] = panel.groupby("trade_date")["ens_v1"].transform(zscore)
panel = panel.dropna(subset=["ens_v1"])
print(f"  ens_v1 特征构建完成, {len(panel):,} 行")

# 验证 ens_v1 无泄漏: 它只用 t 之前模型 → 检查 OOS IC 合理性
p_oos = panel[panel["trade_date"].isin(oos_months)]
ics_ens = monthly_ic(p_oos, "ens_v1")
print(f"  [校验] ens_v1 单因子 OOS IC={ics_ens.mean():+.4f} (应明显>0, 若≈ENH4说明有效)")

# ========== Phase 2: C8 vs C9=C8+ens_v1, OOS WFO ==========
C9_COLS = C8_COLS + ["ens_v1"]

print("\n=== Phase 2: OOS WFO 对比 (2023+) ===")
def run_wfo_feat(feat_cols, depth, nl, mc, rl, label):
    t1 = time.time()
    pred_list = []
    for i, m in enumerate(oos_months):
        tr = panel[panel["trade_date"] < m]
        val_months = sorted(tr["trade_date"].unique())[-3:]
        vm = tr["trade_date"].isin(val_months).values
        X = tr[feat_cols].values; y = tr["fwd_20"].values
        mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                num_leaves=nl, max_depth=depth,
                                min_child_samples=mc, reg_lambda=rl, reg_alpha=0.1,
                                subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
        mdl.fit(X[~vm], y[~vm], eval_set=[(X[vm], y[vm])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        t_m = panel[panel["trade_date"] == m]
        pred_list.append(pd.DataFrame({"trade_date": m, "ts_code": t_m["ts_code"],
                                       "pred": mdl.predict(t_m[feat_cols]),
                                       "fwd_20": t_m["fwd_20"].values}))
    df = pd.concat(pred_list, ignore_index=True)
    ics = monthly_ic(df, "pred")
    icir = ics.mean() / (ics.std(ddof=1) + 1e-12) * np.sqrt(12)
    df["q"] = df.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df.groupby("q", observed=True)["fwd_20"].mean()
    imp = pd.Series(mdl.feature_importances_, index=feat_cols).sort_values(ascending=False)
    return {"label": label, "n_feat": len(feat_cols), "ic": ics.mean(), "icir": icir,
            "pos_rate": (ics>0).mean(), "q1_q5": piv["Q1"]-piv["Q5"],
            "top_feats": " | ".join(f"{k}={v}" for k,v in imp.head(6).items()),
            "df": df, "sec": int(time.time()-t1)}

rows = []
dfs = []
for fc, lbl in [(C8_COLS, "C8_d3"), (C9_COLS, "C9_ENSfeat_d3")]:
    r = run_wfo_feat(fc, 3, 7, 80, 2.0, lbl)
    print(f"  {lbl:>14} n={r['n_feat']:>2} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} "
          f"Q1-Q5={r['q1_q5']:+.2f} 正率={r['pos_rate']*100:.0f}% 耗时{r['sec']}s")
    print(f"    TopFeats: {r['top_feats']}")
    rows.append({k: v for k, v in r.items() if k != "df"})
    dfs.append(r["df"])

# ENS2 混合 (0.5 ENH4 + 0.5 GBDT_v2) 的 IC 对比
df8, df9 = dfs[0], dfs[1]
p_all = panel[["trade_date", "ts_code", "enh4_score"]].copy()
df9 = df9.merge(p_all, on=["trade_date", "ts_code"], how="left")
df9["ens2"] = 0.5 * df9["enh4_score"] + 0.5 * df9.groupby("trade_date")["pred"].rank(pct=True)
ics2 = monthly_ic(df9, "ens2")
print(f"\n  ENS2 (ENH4+GBDT_v2) OOS IC={ics2.mean():+.4f} "
      f"ICIR={ics2.mean()/(ics2.std(ddof=1)+1e-12)*np.sqrt(12):+.2f}")
# 对照: 用 C8 GBDT 预测做同样混合 (即现有 ENS)
df8 = df8.merge(p_all, on=["trade_date", "ts_code"], how="left")
df8["ens1"] = 0.5 * df8["enh4_score"] + 0.5 * df8.groupby("trade_date")["pred"].rank(pct=True)
ics1 = monthly_ic(df8, "ens1")
print(f"  对照 ENS1 (ENH4+GBDT_C8) OOS IC={ics1.mean():+.4f} "
      f"ICIR={ics1.mean()/(ics1.std(ddof=1)+1e-12)*np.sqrt(12):+.2f}")
# 差异检验: 两个 ENS 混合在 OOS 月的 IC 序列相关性
common = ics1.index.intersection(ics2.index)
diff = (ics2[common] - ics1[common]).mean()
print(f"  ENS2 - ENS1 平均 IC 差 = {diff:+.4f} (正=ENS作特征有帮助)")

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n[保存] {OUT} 总耗时 {time.time()-t0:.0f}s")
