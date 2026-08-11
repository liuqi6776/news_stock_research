# -*- coding: utf-8 -*-
"""诊断: C8 = C7 + 精选残差筹码(chip_shift_5_resid, vwap_20_resid, float_pnl_20_resid)

残差化后6个筹码中只有3个IC>0.02:
  chip_shift_5_resid  IC=+0.046 ICIR=1.96 ← 最强
  vwap_20_resid       IC=+0.030
  float_pnl_20_resid  IC=+0.027

测 C8(=C7+3精选) vs C7, depth=3, WFO + 回测验证
"""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "results", "gbdt_c8_chip_resid.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C7_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]
CHIP_COLS = ["vwap_20","float_pnl_20","chip_shift_5"]
CHIP_KEEP = ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
BASE_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
C8_COLS = C7_COLS + CHIP_KEEP  # 7+3=10

t0 = time.time()
panel = pd.read_parquet(PANEL)

def zscore(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

# --- ENH4 ---
for c in ["roe","or_yoy"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["roe"] = panel["roe"].fillna(-99.0)
panel["or_yoy"] = panel["or_yoy"].fillna(-99.0)
g = panel.groupby("trade_date")
panel["enh4_score"] = (-0.40 * g["ivol"].rank(pct=True)
                       -0.35 * g["ret_1m"].rank(pct=True)
                       +0.15 * g["roe"].rank(pct=True)
                       +0.05 * g["or_yoy"].rank(pct=True))

ALL_RAW = list(set(C7_COLS + CHIP_COLS) - {"enh4_score"})
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=C7_COLS + CHIP_COLS + ["fwd_20"])
print(f"[数据] {len(panel):,} 行, {panel['trade_date'].nunique()}月, {time.time()-t0:.0f}s")

# --- 残差化3个筹码 ---
for c in CHIP_COLS:
    rc = f"{c}_resid"
    panel[rc] = np.nan
for dt, grp in panel.groupby("trade_date"):
    if len(grp) < 50: continue
    X_base = grp[BASE_COLS].values
    for c in CHIP_COLS:
        y = grp[c].values
        mask = np.isfinite(y) & np.all(np.isfinite(X_base), axis=1)
        if mask.sum() < 50: continue
        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_base[mask], y[mask])
        resid = y - lr.predict(X_base)
        panel.loc[grp.index[mask], f"{c}_resid"] = -resid  # 方向对齐
for c in CHIP_KEEP:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel = panel.dropna(subset=C8_COLS + ["fwd_20"])
print(f"[残差化] C8面板 {len(panel):,} 行, {time.time()-t0:.0f}s")

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]

def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, gg in d.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg[factor].rank().corr(gg[ret].rank())
    return pd.Series(out).dropna()

def run_wfo(feat_cols, depth, nl, mc, rl, label):
    t1 = time.time()
    pred_list = []
    for i, m in enumerate(oos_months):
        tr = panel[panel["trade_date"] < m]
        val_months = sorted(tr["trade_date"].unique())[-3:]
        vm = tr["trade_date"].isin(val_months).values
        X = tr[feat_cols].values; y = tr["fwd_20"].values
        mdl = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05,
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
    icir = ics.mean()/(ics.std(ddof=1)+1e-12)*np.sqrt(12)
    df["q"] = df.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df.groupby("q", observed=True)["fwd_20"].mean()
    r = {"label": label, "n_feat": len(feat_cols), "depth": depth,
         "ic": ics.mean(), "icir": icir, "pos_rate": (ics>0).mean(),
         "q1_q5": piv["Q1"]-piv["Q5"], "sec": int(time.time()-t1)}
    imp = pd.Series(mdl.feature_importances_, index=feat_cols).sort_values(ascending=False)
    r["top_feats"] = " | ".join(f"{k}={v}" for k,v in imp.head(8).items())
    return r

# --- 测试网格 ---
cfgs = [
    ("C7_d3",       C7_COLS,  3, 7,  80, 2.0),
    ("C8_d3",       C8_COLS,  3, 7,  80, 2.0),
    ("C8_d5",       C8_COLS,  5, 15, 60, 1.5),
    ("C8_d7",       C8_COLS,  7, 31, 50, 1.0),
    ("C8_d10",      C8_COLS,  10, 127, 100, 2.0),
    # 额外: C8 + 更强正则的d3变体
    ("C8_d3_strong", C8_COLS, 3, 7,  120, 3.0),
]
rows = []
for lbl, fc, d, nl, mc, rl in cfgs:
    r = run_wfo(fc, d, nl, mc, rl, lbl)
    rows.append(r)
    print(f"  {lbl:>15} n={r['n_feat']:>2} d={d:>2} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} "
          f"Q1-Q5={r['q1_q5']:+.2f} 正率={r['pos_rate']*100:.0f}% 耗时{r['sec']}s")
    print(f"    TopFeats: {r['top_feats']}")

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n[保存] {OUT} 总耗时 {time.time()-t0:.0f}s")
