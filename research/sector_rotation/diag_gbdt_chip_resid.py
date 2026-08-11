# -*- coding: utf-8 -*-
"""诊断: 筹码因子残差化 + 深度梯度实验

Step 1: 对6个筹码因子逐月截面回归取残差 (对C7基础因子正交化)
  - 基变量: ivol, ret_1m, momentum_20, volatility_20, alpha_006, alpha_012
  - 残差 = chip_factor - OLS(chip_factor ~ C7_base)
  - 残差方向再取负对齐 (原IC为负)

Step 2: 残差化后单因子IC诊断

Step 3: C10_RES = C7 + 6残差筹码, depth梯度 [3,5,7,9,10,12] WFO
  - 同时测 C7 各深度作为对照
  - 重点看 depth=10

输出: results/gbdt_chip_resid_depth.csv
"""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "results", "gbdt_chip_resid_depth.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C7_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]
CHIP_COLS = ["vwap_20","float_pnl_20","prof_pct_20","chip_conc_20","chip_shift_5","pos_vol_20"]
BASE_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]  # 回归基(不含enh4_score)

t0 = time.time()
panel = pd.read_parquet(PANEL)

def zscore(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

# --- ENH4 打分 ---
for c in ["roe","or_yoy"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["roe"] = panel["roe"].fillna(-99.0)
panel["or_yoy"] = panel["or_yoy"].fillna(-99.0)
g = panel.groupby("trade_date")
panel["enh4_score"] = (-0.40 * g["ivol"].rank(pct=True)
                       -0.35 * g["ret_1m"].rank(pct=True)
                       +0.15 * g["roe"].rank(pct=True)
                       +0.05 * g["or_yoy"].rank(pct=True))

# --- 标准化所有特征 ---
ALL_RAW = list(set(C7_COLS + CHIP_COLS) - {"enh4_score"})
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=C7_COLS + CHIP_COLS + ["fwd_20"])
print(f"[数据] 面板 {len(panel):,} 行, 月数 {panel['trade_date'].nunique()}, 耗时 {time.time()-t0:.0f}s")

# ========== Step 1: 筹码因子残差化 ==========
# 逐月截面 OLS: chip_i = a + b1*ivol + b2*ret_1m + ... + b6*alpha_012 + eps
# 残差 = eps (正交于C7基础因子的部分)
print("\n=== Step 1: 筹码因子残差化 (逐月截面OLS) ===")
chip_resid_cols = []
for c in CHIP_COLS:
    rc = f"{c}_resid"
    chip_resid_cols.append(rc)
    panel[rc] = np.nan

for dt, grp in panel.groupby("trade_date"):
    if len(grp) < 50:
        continue
    X_base = grp[BASE_COLS].values
    for c in CHIP_COLS:
        y = grp[c].values
        mask = np.isfinite(y) & np.all(np.isfinite(X_base), axis=1)
        if mask.sum() < 50:
            continue
        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_base[mask], y[mask])
        resid = y - lr.predict(X_base)
        resid = pd.Series(resid, index=grp.index[mask])
        # 残差方向对齐: 原IC为负 → 取负使"大→涨"
        panel.loc[resid.index, f"{c}_resid"] = -resid.values

# 残差标准化
for c in chip_resid_cols:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)

print(f"  残差化完成, 耗时 {time.time()-t0:.0f}s")

# ========== Step 2: 残差化后单因子IC ==========
months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]
panel_oos = panel[panel["trade_date"].isin(oos_months)].copy()

def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, gg in d.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg[factor].rank().corr(gg[ret].rank())
    return pd.Series(out).dropna()

print("\n=== Step 2: 残差化筹码因子单因子 IC (OOS 2023+) ===")
single_rows = []
for c in chip_resid_cols + CHIP_COLS:  # 残差 vs 原始 对比
    ics = monthly_ic(panel_oos, c)
    icir = ics.mean()/(ics.std(ddof=1)+1e-12)*np.sqrt(12)
    tag = "RESID" if c.endswith("_resid") else "RAW"
    print(f"  {c:>20} [{tag}]: IC={ics.mean():+.4f} ICIR={icir:+.2f} 正率={(ics>0).mean()*100:.0f}%")
    single_rows.append({"factor": c, "type": tag, "ic": ics.mean(), "icir": icir,
                        "pos_rate": (ics>0).mean(), "n": len(ics)})

# ========== Step 3: WFO depth梯度 ==========
C10_RES_COLS = C7_COLS + chip_resid_cols  # 7+6=13

print(f"\n=== Step 3: WFO depth梯度 (C7对照 vs C10_RES) ===")
print(f"  C10_RES = C7 + 6残差筹码 = {C10_RES_COLS}")

def run_wfo_feat(feat_cols, depth, nl, mc, rl, label):
    t1 = time.time()
    pred_list = []
    for i, m in enumerate(oos_months):
        tr = panel[panel["trade_date"] < m]
        val_months = sorted(tr["trade_date"].unique())[-3:]
        val_mask = tr["trade_date"].isin(val_months).values
        X = tr[feat_cols].values; y = tr["fwd_20"].values
        X_fit, X_val = X[~val_mask], X[val_mask]
        y_fit, y_val = y[~val_mask], y[val_mask]
        mdl = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05,
            num_leaves=nl, max_depth=depth,
            min_child_samples=mc, reg_lambda=rl, reg_alpha=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
        mdl.fit(X_fit, y_fit,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        t_m = panel[panel["trade_date"] == m]
        pred = mdl.predict(t_m[feat_cols])
        pred_list.append(pd.DataFrame({"trade_date": m, "ts_code": t_m["ts_code"],
                                       "pred": pred, "fwd_20": t_m["fwd_20"].values}))
    df_oos = pd.concat(pred_list, ignore_index=True)
    ics = monthly_ic(df_oos, "pred")
    icir = ics.mean() / (ics.std(ddof=1) + 1e-12) * np.sqrt(12)
    df_oos["q"] = df_oos.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df_oos.groupby("q", observed=True)["fwd_20"].mean()
    r = {"label": label, "n_feat": len(feat_cols), "depth": depth,
         "num_leaves": nl, "min_child": mc, "reg_lambda": rl,
         "ic": ics.mean(), "icir": icir, "pos_rate": (ics>0).mean(),
         "q1_q5": piv["Q1"] - piv["Q5"], "sec": int(time.time()-t1)}
    imp = pd.Series(mdl.feature_importances_, index=feat_cols).sort_values(ascending=False)
    r["top_feats"] = " | ".join(f"{k}={v}" for k,v in imp.head(8).items())
    return r

# depth梯度: 3,5,7,9,10,12 — C7和C10_RES都测
# num_leaves 随depth调整: d3→7, d5→15, d7→31, d9→63, d10→127, d12→255
# min_child 随depth上调
DEPTH_GRID = [
    # (depth, num_leaves, min_child, reg_lambda)
    (3,  7,   80,  2.0),
    (5,  15,  60,  1.5),
    (7,  31,  50,  1.0),
    (9,  63,  80,  1.5),
    (10, 127, 100, 2.0),
    (12, 255, 150, 3.0),
]

wfo_rows = []
for d, nl, mc, rl in DEPTH_GRID:
    # C7 对照
    r7 = run_wfo_feat(C7_COLS, d, nl, mc, rl, f"C7_d{d}")
    wfo_rows.append(r7)
    print(f"  C7_d{d:>2}  n={r7['n_feat']:>2} | IC={r7['ic']:+.4f} ICIR={r7['icir']:+.2f} "
          f"Q1-Q5={r7['q1_q5']:+.2f} 耗时{r7['sec']}s")

    # C10_RES
    r10 = run_wfo_feat(C10_RES_COLS, d, nl, mc, rl, f"C10RES_d{d}")
    wfo_rows.append(r10)
    print(f"  C10RES_d{d:>2} n={r10['n_feat']:>2} | IC={r10['ic']:+.4f} ICIR={r10['icir']:+.2f} "
          f"Q1-Q5={r10['q1_q5']:+.2f} 耗时{r10['sec']}s")
    print(f"    TopFeats: {r10['top_feats']}")

# 输出
s_df = pd.DataFrame(single_rows)
w_df = pd.DataFrame(wfo_rows)
s_df["part"] = "single_factor"
w_df["part"] = "wfo_model"
all_out = pd.concat([s_df, w_df], ignore_index=True)
all_out.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"\n=== 汇总: depth梯度 IC ===")
pivot = w_df.pivot_table(index="depth", columns="label", values="ic")
print(pivot.to_string())
print(f"\n=== 汇总: depth梯度 ICIR ===")
pivot2 = w_df.pivot_table(index="depth", columns="label", values="icir")
print(pivot2.to_string())
print(f"\n[保存] {OUT} 总耗时 {time.time()-t0:.0f}s")
