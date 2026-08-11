# -*- coding: utf-8 -*-
"""B3 诊断2: 滚动重训 LGBMRegressor vs LGBMRanker vs ENH4 的 OOS IC 对照"""
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

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]

def run_rolling(mode):
    """mode: 'reg' 回归 fwd_20 | 'rank' lambdarank 5档"""
    pred_list = []
    for i, m in enumerate(oos_months):
        tr = panel[panel["trade_date"] < m].sort_values("trade_date")
        X, y = tr[FEAT_COLS].values, tr["fwd_20"].values
        group = tr.groupby("trade_date", sort=False).size().values
        val_months = sorted(tr["trade_date"].unique())[-3:]
        vm = tr["trade_date"].isin(val_months).values
        X_fit, y_fit = X[~vm], y[~vm]
        X_val, y_val = X[vm], y[vm]
        g_fit = tr.loc[~vm].groupby("trade_date", sort=False).size().values
        g_val = tr.loc[vm].groupby("trade_date", sort=False).size().values

        if mode == "reg":
            mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                    max_depth=6, min_child_samples=50, reg_lambda=1.0,
                                    subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
            mdl.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
            # 训练IC(全部训练集) 用于诊断
            tr_ic = _ic(tr.assign(_p=mdl.predict(X)), factor="_p")
        else:
            y5 = tr.groupby("trade_date", sort=False)["fwd_20"].transform(
                lambda s: pd.qcut(s.rank(method="first"), 5, labels=[0,1,2,3,4]).astype(int)).values
            mdl = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                                 num_leaves=31, max_depth=6, min_child_samples=50,
                                 reg_lambda=1.0, subsample=0.9, colsample_bytree=0.9,
                                 random_state=42, verbose=-1)
            mdl.fit(X_fit, y5[~vm], group=g_fit, eval_set=[(X_val, y5[vm])], eval_group=[g_val],
                    eval_metric="ndcg", eval_at=[5],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
            tr_ic = _ic(tr.assign(_p=mdl.predict(X)), factor="_p")

        om = panel[panel["trade_date"] == m].sort_values("trade_date")
        p = mdl.predict(om[FEAT_COLS])
        pred_list.append(pd.DataFrame({"trade_date": m, "fwd_20": om["fwd_20"].values,
                                       "pred": p, "tr_ic": tr_ic,
                                       "ivol": om["ivol"].values, "ret_1m": om["ret_1m"].values,
                                       "roe": om["roe"].values}))
    return pd.concat(pred_list, ignore_index=True)

def _ic(d, factor="pred", ret="fwd_20"):
    out = {}
    for dt, g in d.groupby("trade_date"):
        if len(g) < 50:
            continue
        out[dt] = g[factor].rank().corr(g[ret].rank())
    return pd.Series(out).dropna().mean()

for mode in ("reg", "rank"):
    df = run_rolling(mode)
    ics = _ic(df)
    tr_ics = df.groupby("trade_date")["tr_ic"].first()
    print(f"\n=== 滚动重训 {mode} ===")
    print(f"  OOS IC={ics:+.4f}, 训练IC均值={tr_ics.mean():+.4f}, 耗时{time.time()-t0:.0f}s")

# ENH4 对照
df = panel[panel["trade_date"] >= 20230101].copy()
df["enh4"] = (-0.40*df["ivol"].rank(pct=True) - 0.35*df["ret_1m"].rank(pct=True)
              + 0.15*df["roe"].rank(pct=True) + 0.05*df["or_yoy"].rank(pct=True)
              + 0.05*df["netprofit_yoy"].rank(pct=True))
print(f"\n=== ENH4 OOS IC = {_ic(df, 'enh4'):+.4f} ===")
