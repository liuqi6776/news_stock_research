# -*- coding: utf-8 -*-
"""B3 v2: 股票日频 GBDT (LGBMRanker) 滚动重训 Walk-Forward 训练与 OOS 评估 (Plan B)

v2 改进 (修复 v1 静态切分过拟合):
  - expand-window 滚动重训: 预测月 m 只用 <= m-1 的数据重训模型
  - 早停: 训练集最后 3 个月做验证 (ndcg@5)
  - 更强正则: min_child_samples=50, num_leaves=31
  - 2023-04 后财务因子自然进入训练集

产出: artifacts/stock_gbdt_lgbmranker.joblib + reports/gbdt_stock_oos_diagnosis.png
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
ART = os.path.join(ROOT, "artifacts", "stock_gbdt_lgbmranker.joblib")
REPORT = os.path.join(ROOT, "research", "sector_rotation", "reports", "gbdt_stock_oos_diagnosis.png")
os.makedirs(os.path.dirname(REPORT), exist_ok=True)

t0 = time.time()
panel = pd.read_parquet(PANEL)

PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
FEAT_COLS = PRICE_COLS + FIN_COLS + ["has_fin"]

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

for c in PRICE_COLS + FIN_COLS:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["has_fin"] = panel["roe"].notna().astype(int)
for c in PRICE_COLS + FIN_COLS:
    panel[c] = panel.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
panel[FIN_COLS] = panel[FIN_COLS].fillna(-99.0)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=FEAT_COLS + ["fwd_20"])

def make_rank_data(df):
    df = df.sort_values("trade_date")
    X = df[FEAT_COLS].values
    y = df.groupby("trade_date", sort=False)["fwd_20"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=[0, 1, 2, 3, 4]).astype(int)).values
    group = df.groupby("trade_date", sort=False).size().values
    return df, X, y, group

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]

# ---------- 滚动重训: 每月重训, 预测下月 ----------
pred_list = []
last_model = None
for i, m in enumerate(oos_months):
    train_panel = panel[panel["trade_date"] < m]
    df_tr, X_tr, y_tr, g_tr = make_rank_data(train_panel)
    # 早停验证: 训练集最后 3 个月
    val_months = sorted(df_tr["trade_date"].unique())[-3:]
    val_mask = df_tr["trade_date"].isin(val_months).values
    X_fit, y_fit = X_tr[~val_mask], y_tr[~val_mask]
    X_val, y_val = X_tr[val_mask], y_tr[val_mask]
    g_fit = df_tr.loc[~val_mask].groupby("trade_date", sort=False).size().values
    g_val = df_tr.loc[val_mask].groupby("trade_date", sort=False).size().values

    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
        min_child_samples=50, reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1,
    )
    model.fit(X_fit, y_fit, group=g_fit,
              eval_set=[(X_val, y_val)], eval_group=[g_val],
              eval_metric="ndcg", eval_at=[5],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    last_model = model

    df_oos_m = panel[panel["trade_date"] == m].sort_values("trade_date")
    pred = model.predict(df_oos_m[FEAT_COLS])
    pred_list.append(pd.DataFrame({"trade_date": m, "ts_code": df_oos_m["ts_code"],
                                   "pred": pred, "fwd_20": df_oos_m["fwd_20"].values,
                                   "ivol": df_oos_m["ivol"].values, "ret_1m": df_oos_m["ret_1m"].values,
                                   "roe": df_oos_m["roe"].values}))
    if (i + 1) % 6 == 0 or i == len(oos_months) - 1:
        print(f"  重训 {i+1}/{len(oos_months)}: 预测 {m}, 训练 {train_panel['trade_date'].nunique()}月, "
              f"耗时{time.time()-t0:.0f}s", flush=True)

df_oos = pd.concat(pred_list, ignore_index=True)
print(f"[滚动重训] OOS 预测完成: {df_oos['trade_date'].nunique()}月/{len(df_oos):,}行, 耗时{time.time()-t0:.0f}s")

# ---------- OOS 评估 ----------
def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, g in d.groupby("trade_date"):
        if len(g) < 50:
            continue
        out[dt] = g[factor].rank().corr(g[ret].rank())
    return pd.Series(out).dropna()

ics = monthly_ic(df_oos, "pred")
icir = ics.mean() / (ics.std(ddof=1) + 1e-12) * np.sqrt(12)
print("\n=== OOS (2023-2025) 滚动重训 GBDT Rank IC ===")
print(f"  IC={ics.mean():+.4f}, ICIR={icir:+.2f}, 正率={(ics>0).mean()*100:.0f}%, 月数={len(ics)}")

df_oos["enh4"] = (-0.40*df_oos["ivol"].rank(pct=True) - 0.35*df_oos["ret_1m"].rank(pct=True)
                  + 0.15*df_oos["roe"].rank(pct=True) + 0.05*df_oos["or_yoy"].rank(pct=True))
enh_ics = monthly_ic(df_oos, "enh4")
print(f"\n=== 对比: ENH4 OOS IC = {enh_ics.mean():+.4f} (GBDT {ics.mean():+.4f}) ===")

df_oos["q"] = df_oos.groupby("trade_date")["pred"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
piv = df_oos.groupby("q", observed=True)["fwd_20"].mean()
print("\n=== 五分位月均fwd20 (%) ===")
print("  " + ", ".join(f"{k}={v:+.2f}" for k, v in piv.items()) + f"  Q1-Q5={piv['Q1']-piv['Q5']:+.2f}")

# 逐年
print("\n=== 逐年 IC ===")
for y in (2023, 2024, 2025):
    sub = ics[ics.index // 100 == y]
    print(f"  {y}: IC={sub.mean():+.4f}, 月数={len(sub)}")

# ---------- 图 ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
axes[0][0].plot(ics.index.astype(str), ics.values, marker="o", ms=3)
axes[0][0].axhline(0, color="r", lw=0.8)
axes[0][0].axhline(ics.mean(), color="g", lw=0.8, ls="--", label=f"GBDT IC={ics.mean():+.4f}")
axes[0][0].axhline(enh_ics.mean(), color="b", lw=0.8, ls=":", label=f"ENH4 IC={enh_ics.mean():+.4f}")
axes[0][0].set_title(f"OOS 月度 Rank IC (ICIR={icir:+.2f})")
axes[0][0].legend(fontsize=8)
axes[0][1].bar(piv.index.astype(str), piv.values, color="steelblue")
axes[0][1].set_title("OOS 五分位月均 fwd_20 (%)")
imp = pd.Series(last_model.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
axes[1][0].barh(imp.index[:12], imp.values[:12][::-1], color="darkorange")
axes[1][0].set_title("Feature Importance Top12 (末次重训)")
ls = {}
for dt, g in df_oos.groupby("trade_date"):
    ls[dt] = g.loc[g["q"]=="Q1","fwd_20"].mean() - g.loc[g["q"]=="Q5","fwd_20"].mean()
ls = pd.Series(ls)
axes[1][1].plot(ls.index.astype(str), (1 + ls/100).cumprod(), marker="o", ms=3)
axes[1][1].set_title("Q1-Q5 多空累计净值")
plt.tight_layout()
plt.savefig(REPORT, dpi=120)
print(f"\n[图] 已保存: {REPORT}")

joblib.dump({"model": last_model, "feat_cols": FEAT_COLS, "rank_ic": ics.mean(),
             "enh4_ic": enh_ics.mean()}, ART)
print(f"[模型] 已保存: {ART} (OOS GBDT IC={ics.mean():+.4f} vs ENH4 {enh_ics.mean():+.4f}), "
      f"总耗时{time.time()-t0:.0f}s")
