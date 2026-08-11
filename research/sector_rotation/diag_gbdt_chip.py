# -*- coding: utf-8 -*-
"""诊断: 筹码因子单因子 IC + C7 vs C13_CHIP(C7+6筹码) 深度3 WFO 对比

输出: results/gbdt_chip_diag.csv
     - 6 筹码因子单因子 Rank IC (截面Spearman, 月频)
     - C7 (baseline) WFO IC vs C13_CHIP (depth=3, num_leaves=7, mc=80, λ=2.0)
"""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "results", "gbdt_chip_diag.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C7_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]
CHIP_COLS = ["vwap_20","float_pnl_20","prof_pct_20","chip_conc_20","chip_shift_5","pos_vol_20"]
C13_COLS = C7_COLS + CHIP_COLS

t0 = time.time()
panel = pd.read_parquet(PANEL)

def zscore(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

# --- 计算 ENH4 打分 (对齐 C7 的 enh4_score) ---
for c in ["roe","or_yoy"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["roe"] = panel["roe"].fillna(-99.0)
panel["or_yoy"] = panel["or_yoy"].fillna(-99.0)
g = panel.groupby("trade_date")
panel["enh4_score"] = (-0.40 * g["ivol"].rank(pct=True)
                       -0.35 * g["ret_1m"].rank(pct=True)
                       +0.15 * g["roe"].rank(pct=True)
                       +0.05 * g["or_yoy"].rank(pct=True))

# --- 标准化 ---
ALL_STD_COLS = list(set(C13_COLS) - {"enh4_score"})  # enh4 已是秩分数不用 zscore
for c in ALL_STD_COLS:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
for c in ALL_STD_COLS:
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=C13_COLS + ["fwd_20"])
print(f"[数据] 面板 {len(panel):,} 行, 月数 {panel['trade_date'].nunique()}, 耗时 {time.time()-t0:.0f}s")

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]
print(f"[设置] OOS 月数 {len(oos_months)} ({oos_months[0]}~{oos_months[-1]})")

def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, gg in d.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg[factor].rank().corr(gg[ret].rank())
    return pd.Series(out).dropna()

# ---------- Part 1: 6 个筹码因子单因子 IC ----------
print("\n=== Part 1: 单因子 Rank IC (OOS 2023+) ===")
single_rows = []
panel_oos = panel[panel["trade_date"].isin(oos_months)].copy()
for c in CHIP_COLS + ["ivol","ret_1m","momentum_20","volatility_20","enh4_score"]:
    ics = monthly_ic(panel_oos, c)
    icir = ics.mean()/(ics.std(ddof=1)+1e-12)*np.sqrt(12)
    print(f"  {c:>14}: IC={ics.mean():+.4f} ICIR={icir:+.2f} 正率={(ics>0).mean()*100:.0f}% N={len(ics)}")
    single_rows.append({"factor": c, "type": "CHIP" if c in CHIP_COLS else "BASE",
                        "ic": ics.mean(), "icir": icir, "pos_rate": (ics>0).mean(), "n": len(ics)})

# ---------- Part 2: WFO C7(baseline,d6) vs C7(d3) vs C13(d3) ----------
print("\n=== Part 2: WFO 对比 ===")
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
         "ic": ics.mean(), "icir": icir, "pos_rate": (ics>0).mean(),
         "q1_q5": piv["Q1"] - piv["Q5"], "sec": int(time.time()-t1)}
    # 特征重要性 (末次重训 top6)
    imp = pd.Series(mdl.feature_importances_, index=feat_cols).sort_values(ascending=False)
    r["top_feats"] = " | ".join(f"{k}={v}" for k,v in imp.head(6).items())
    return r, imp

cfgs = [
    ("C7_baseline_d6",  C7_COLS,  6, 31,  50,  1.0),
    ("C7_depth3_opt",   C7_COLS,  3, 7,   80,  2.0),
    ("C13_CHIP_d3",     C13_COLS, 3, 7,   80,  2.0),
    ("C13_CHIP_d5",     C13_COLS, 5, 15,  60,  1.5),
]

wfo_rows = []
for cfg in cfgs:
    lbl, fc, d, nl, mc, rl = cfg
    r, imp = run_wfo_feat(fc, d, nl, mc, rl, lbl)
    wfo_rows.append(r)
    print(f"  {lbl:>20} n={r['n_feat']} d={d} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} "
          f"Q1-Q5={r['q1_q5']:+.2f} 耗时{r['sec']}s")
    print(f"    TopFeats: {r['top_feats']}")

# ---------- Part 2b: 筹码方向修正 + 筛选 对齐后加入 ----------
# 单因子 IC 全为负 → 对 chip_conc_20, vwap_20, prof_pct_20 三个最强取负, 对齐"大→涨"
for c, orig in [("chip_conc_neg", "chip_conc_20"), ("vwap_neg", "vwap_20"), ("prof_neg", "prof_pct_20")]:
    panel[c] = -panel[orig]
CHIP3_ALIGNED = ["chip_conc_neg","vwap_neg","prof_neg"]
C10_COLS = C7_COLS + CHIP3_ALIGNED
for c in CHIP3_ALIGNED:
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel = panel.dropna(subset=C10_COLS + ["fwd_20"])
print(f"\n[2b] 方向对齐后 C10 面板: {len(panel):,} 行")
cfgs2 = [
    ("C7_depth3_opt_again",  C7_COLS,    3, 7,  80, 2.0),
    ("C10_CHIP3_d3",         C10_COLS,   3, 7,  80, 2.0),
    ("C10_CHIP3_d5",         C10_COLS,   5, 15, 60, 1.5),
    ("C10_CHIP3_d7",         C10_COLS,   7, 63, 80, 1.5),
]
for cfg in cfgs2:
    lbl, fc, d, nl, mc, rl = cfg
    r, imp = run_wfo_feat(fc, d, nl, mc, rl, lbl)
    wfo_rows.append(r)
    print(f"  {lbl:>20} n={r['n_feat']} d={d} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} "
          f"Q1-Q5={r['q1_q5']:+.2f} 耗时{r['sec']}s")
    print(f"    TopFeats: {r['top_feats']}")

# 输出综合
s_df = pd.DataFrame(single_rows)
w_df = pd.DataFrame(wfo_rows)
s_df["part"] = "1_single_factor"
w_df["part"] = "2_wfo_model"
all_out = pd.concat([s_df, w_df], ignore_index=True)
all_out.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n[保存] {OUT} 总耗时 {time.time()-t0:.0f}s")
