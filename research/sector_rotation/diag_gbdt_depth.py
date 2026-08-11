# -*- coding: utf-8 -*-
"""诊断: GBDT 深度调参 (C7 特征集, LGBMRegressor Walk-Forward OOS IC)

当前 baseline: max_depth=6, num_leaves=31, min_child_samples=50 (reg模式)
测试网格:
  depth: 3, 5, 7, 9, 12
  num_leaves: min(2^depth - 1, 31/31/63/127/255) 避免过深无意义
  min_child_samples: 随深度增大上调 (浅树小样本/深树要正则)
输出: results/gbdt_depth_grid.csv
"""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "results", "gbdt_depth_grid.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C7_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]

t0 = time.time()
panel = pd.read_parquet(PANEL)

# --- 计算 ENH4 打分 (对齐 C7 的 enh4_score) ---
def zscore(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

for c in ["roe","or_yoy"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["roe"] = panel["roe"].fillna(-99.0)
panel["or_yoy"] = panel["or_yoy"].fillna(-99.0)
panel["has_fin"] = (panel["roe"] > -90).astype(int)
g = panel.groupby("trade_date")
panel["enh4_score"] = (-0.40 * g["ivol"].rank(pct=True)
                       -0.35 * g["ret_1m"].rank(pct=True)
                       +0.15 * g["roe"].rank(pct=True)
                       +0.05 * g["or_yoy"].rank(pct=True))

# --- C7 winsorize + zscore (截面标准化, 同 diag_gbdt_features) ---
for c in C7_COLS:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
for c in C7_COLS:
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=C7_COLS + ["fwd_20"])
print(f"[数据] 面板 {len(panel):,} 行, 月数 {panel['trade_date'].nunique()}, 耗时 {time.time()-t0:.0f}s")

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]
print(f"[设置] OOS 月数 {len(oos_months)} ({oos_months[0]}~{oos_months[-1]})")

def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, g in d.groupby("trade_date"):
        if len(g) < 50: continue
        out[dt] = g[factor].rank().corr(g[ret].rank())
    return pd.Series(out).dropna()

def run_wfo(depth, num_leaves, min_child, reg_lambda=1.0):
    t1 = time.time()
    pred_list = []
    for i, m in enumerate(oos_months):
        tr = panel[panel["trade_date"] < m]
        val_months = sorted(tr["trade_date"].unique())[-3:]
        val_mask = tr["trade_date"].isin(val_months).values
        X = tr[C7_COLS].values; y = tr["fwd_20"].values
        X_fit, X_val = X[~val_mask], X[val_mask]
        y_fit, y_val = y[~val_mask], y[val_mask]
        mdl = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05,
            num_leaves=num_leaves, max_depth=depth,
            min_child_samples=min_child, reg_lambda=reg_lambda, reg_alpha=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
        mdl.fit(X_fit, y_fit,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        t_m = panel[panel["trade_date"] == m]
        pred = mdl.predict(t_m[C7_COLS])
        pred_list.append(pd.DataFrame({"trade_date": m, "ts_code": t_m["ts_code"],
                                       "pred": pred, "fwd_20": t_m["fwd_20"].values}))
    df_oos = pd.concat(pred_list, ignore_index=True)
    ics = monthly_ic(df_oos, "pred")
    icir = ics.mean() / (ics.std(ddof=1) + 1e-12) * np.sqrt(12)
    df_oos["q"] = df_oos.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df_oos.groupby("q", observed=True)["fwd_20"].mean()
    return {"depth": depth, "num_leaves": num_leaves, "min_child": min_child,
            "reg_lambda": reg_lambda,
            "ic": ics.mean(), "icir": icir, "pos_rate": (ics>0).mean(),
            "q1_q5": piv["Q1"] - piv["Q5"],
            "n_months": len(ics), "sec": int(time.time()-t1)}

# --- 测试网格: 深度 + 匹配 num_leaves + 匹配 min_child ---
GRID = [
    # (depth, num_leaves, min_child_samples, reg_lambda)
    (3,  7,    80,  2.0),   # 更浅 → 更宽正则, 样本数要求高
    (5,  15,   60,  1.5),   # 中等浅
    (6,  31,   50,  1.0),   # baseline
    (7,  63,   80,  1.5),   # 略深 → 加正则加 min_child
    (9,  127,  120, 2.0),   # 较深
    (12, 255,  200, 3.0),   # 深树, 强正则
]
rows = []
for i, cfg in enumerate(GRID):
    d, nl, mc, rl = cfg
    r = run_wfo(d, nl, mc, rl)
    rows.append(r)
    print(f"[{i+1}/{len(GRID)}] depth={d:>2} leaves={nl:>3} mc={mc:>3} lam={rl} | "
          f"IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1_q5']:+.2f} 耗时{r['sec']}s",
          flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n=== 深度调参汇总 ===")
print(res[["depth","num_leaves","min_child","reg_lambda","ic","icir","pos_rate","q1_q5"]].to_string(index=False))
print(f"\n[保存] {OUT} 总耗时 {time.time()-t0:.0f}s")
