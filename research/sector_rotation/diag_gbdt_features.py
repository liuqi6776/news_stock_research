# -*- coding: utf-8 -*-
"""B3 v3 诊断: 特征集梯度实验 — GBDT 合理去冗余因子后能否反超 ENH4

假设: 当前 GBDT 17 特征冗余+财务漂移导致过拟合。做特征集梯度:
  A5 : ENH4 对齐 5 因子 (ivol,ret_1m,roe,or_yoy,netprofit_yoy)
  E5v: 纯价量 5 因子  (ivol,ret_1m,momentum_20,volatility_20,alpha_006) 全期无财务漂移
  B6 : 核心价量 6 因子 (+alpha_012 量价反转)
  C7 : B6 + ENH4_score (GBDT 学 ENH4 残差)
  D17: 当前全部 17 特征 (对照)

模型: 滚动重训 LGBMRegressor (expand-window, 早停) + LGBMRanker 对照
评估: OOS Rank IC / 五分位 Q1-Q5 / 训练IC
"""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
t0 = time.time()
panel = pd.read_parquet(PANEL)

PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
ALL17 = PRICE_COLS + FIN_COLS + ["has_fin"]

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

p = panel.copy()
for c in PRICE_COLS + FIN_COLS:
    p[c] = p.groupby("trade_date")[c].transform(lambda s: winsorize(s))
p["has_fin"] = p["roe"].notna().astype(int)
for c in PRICE_COLS + FIN_COLS:
    p[c] = p.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
p[FIN_COLS] = p[FIN_COLS].fillna(-99.0)
p["fwd_20"] = p.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
p = p.dropna(subset=["fwd_20"])

# ENH4 score (截面 rank 加权) — 与 B4 回测完全一致
p["enh4_score"] = (-0.40*p["ivol"].rank(pct=True) - 0.35*p["ret_1m"].rank(pct=True)
                   + 0.15*p["roe"].rank(pct=True) + 0.05*p["or_yoy"].rank(pct=True)
                   + 0.05*p["netprofit_yoy"].rank(pct=True))

FEAT_SETS = {
    "A5_enh4对齐": ["ivol", "ret_1m", "roe", "or_yoy", "netprofit_yoy"],
    "E5v_纯价量": ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006"],
    "B6_核心价量": ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012"],
    "C7_学残差": ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012", "enh4_score"],
    "D17_全部": ALL17,
}

months = sorted(p["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]

def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, g in d.groupby("trade_date"):
        if len(g) < 50:
            continue
        out[dt] = g[factor].rank().corr(g[ret].rank())
    return pd.Series(out).dropna()

def rolling_eval(feat_cols, mode="reg"):
    pred_list = []
    for i, m in enumerate(oos_months):
        tr = p[p["trade_date"] < m].sort_values("trade_date")
        X, y = tr[feat_cols].values, tr["fwd_20"].values
        val_months = sorted(tr["trade_date"].unique())[-3:]
        vm = tr["trade_date"].isin(val_months).values
        X_fit, y_fit = X[~vm], y[~vm]
        X_val, y_val = X[vm], y[vm]

        if mode == "reg":
            mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                    max_depth=6, min_child_samples=50, reg_lambda=1.0,
                                    subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
            mdl.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
        else:  # rank
            y5 = tr.groupby("trade_date", sort=False)["fwd_20"].transform(
                lambda s: pd.qcut(s.rank(method="first"), 5, labels=[0,1,2,3,4]).astype(int)).values
            g = tr.groupby("trade_date", sort=False).size().values
            g_fit = tr.loc[~vm].groupby("trade_date", sort=False).size().values
            g_val = tr.loc[vm].groupby("trade_date", sort=False).size().values
            mdl = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                                 num_leaves=31, max_depth=6, min_child_samples=50, reg_lambda=1.0,
                                 subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
            mdl.fit(X_fit, y5[~vm], group=g_fit, eval_set=[(X_val, y5[vm])], eval_group=[g_val],
                    eval_metric="ndcg", eval_at=[5],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
            y = y5

        om = p[p["trade_date"] == m].sort_values("trade_date")
        pred = mdl.predict(om[feat_cols])
        tr_ic = monthly_ic(tr.assign(_p=mdl.predict(X)), factor="_p").mean()
        pred_list.append(pd.DataFrame({"trade_date": m, "fwd_20": om["fwd_20"].values,
                                       "pred": pred, "tr_ic": tr_ic}))
    return pd.concat(pred_list, ignore_index=True)

# ENH4 基准
df_enh = p[p["trade_date"] >= 20230101]
enh_ic = monthly_ic(df_enh, "enh4_score").mean()
print(f"\n{'特征集':<12} {'模式':<5} {'训练IC':>7} {'OOS IC':>8} {'ICIR':>6} {'正率':>6}  {'Q1-Q5':>7}  耗时")
print("-" * 90)
summary = {}
for name, cols in FEAT_SETS.items():
    for mode in ("reg", "rank"):
        df = rolling_eval(cols, mode)
        ics = monthly_ic(df, "pred")
        icir = ics.mean() / (ics.std(ddof=1) + 1e-12) * np.sqrt(12)
        df["q"] = df.groupby("trade_date")["pred"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=[0,1,2,3,4]))
        q15 = df.groupby("q", observed=True)["fwd_20"].mean()
        ls = q15.iloc[0] - q15.iloc[-1]
        tr_ic = df.groupby("trade_date")["tr_ic"].first().mean()
        print(f"{name:<12} {mode:<5} {tr_ic:>7.4f} {ics.mean():>8.4f} {icir:>6.2f} "
              f"{(ics>0).mean()*100:>5.0f}% {ls:>7.2f}  {time.time()-t0:>5.0f}s", flush=True)
        summary[(name, mode)] = {"ic": ics.mean(), "tr_ic": tr_ic, "icir": icir,
                                 "q15": ls, "q1": q15.iloc[0], "q5": q15.iloc[-1]}

print(f"\n=== ENH4 线性基准: OOS IC = {enh_ic:+.4f} ===")
print("\n=== 汇总 (按 OOS IC 排序) ===")
for k in sorted(summary, key=lambda k: -summary[k]["ic"]):
    v = summary[k]
    print(f"  {k[0]:<12} {k[1]:<5} IC={v['ic']:+.4f} 训练IC={v['tr_ic']:+.4f} "
          f"ICIR={v['icir']:+.2f} Q1-Q5={v['q15']:+.2f} | ENH4差值={v['ic']-enh_ic:+.4f}")

pd.DataFrame([{"特征集": k[0], "模式": k[1], **v} for k, v in summary.items()]).to_csv(
    os.path.join(ROOT, "research", "sector_rotation", "results", "gbdt_features_grid.csv"),
    index=False, encoding="utf-8-sig")
print(f"\n[完成] 总耗时 {time.time()-t0:.0f}s")
