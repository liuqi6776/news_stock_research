# -*- coding: utf-8 -*-
"""B3 诊断: 训练期 vs OOS 因子 IC + 回归模型对照"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
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

df_tr = panel[panel["trade_date"] <= 20221130]
df_oos = panel[panel["trade_date"] >= 20230101]

def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, g in d.groupby("trade_date"):
        if len(g) < 50:
            continue
        out[dt] = g[factor].rank().corr(g[ret].rank())
    return pd.Series(out).dropna()

print("=== 单因子 IC: 训练期(2020-22) vs OOS(2023-25) ===")
for c in PRICE_COLS[:6] + ["f_vol_iv"]:
    pass
for c in ["ret_1m", "ivol", "momentum_5", "momentum_20", "volatility_20", "alpha_006"]:
    ic_tr = monthly_ic(df_tr, c).mean()
    ic_oos = monthly_ic(df_oos, c).mean()
    print(f"  {c:<12} 训练IC={ic_tr:+.4f}  OOS IC={ic_oos:+.4f}")

# ENH4 两期
def enh4(d):
    return (-0.40*d["ivol"].rank(pct=True) - 0.35*d["ret_1m"].rank(pct=True)
            + 0.15*d["roe"].rank(pct=True) + 0.05*d["or_yoy"].rank(pct=True)
            + 0.05*d["netprofit_yoy"].rank(pct=True))
print(f"\n  ENH4:        训练IC={monthly_ic(df_tr.assign(x=enh4(df_tr)), 'x').mean():+.4f}  "
      f"OOS IC={monthly_ic(df_oos.assign(x=enh4(df_oos)), 'x').mean():+.4f}")

# 回归模型对照
df_tr = df_tr.sort_values("trade_date"); df_oos = df_oos.sort_values("trade_date")
reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=6,
                        min_child_samples=30, reg_lambda=1.0, random_state=42, verbose=-1)
reg.fit(df_tr[FEAT_COLS], df_tr["fwd_20"])
pred = reg.predict(df_oos[FEAT_COLS])
df_oos = df_oos.copy(); df_oos["pred"] = pred
ic_reg = monthly_ic(df_oos, "pred").mean()
ic_reg_tr = monthly_ic(df_tr.assign(pred=reg.predict(df_tr[FEAT_COLS])), "pred").mean()
print(f"\n=== LGBMRegressor(回归 fwd_20) ===")
print(f"  训练IC={ic_reg_tr:+.4f}  OOS IC={ic_reg:+.4f}")
print(f"  feature importance top8: {list(pd.Series(reg.feature_importances_, index=FEAT_COLS).nlargest(8).index)}")
