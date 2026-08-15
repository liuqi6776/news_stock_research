# -*- coding: utf-8 -*-
"""Alpha101 因子进 GBDT 的滚动 WFO 对比诊断 (三臂)

臂1: C8 基线 (现有)
臂2: C8 + 10 个原始 Alpha101 因子
臂3: C8 + 10 个残差化 Alpha101 因子 (对 CHIP_BASE 逐月截面 OLS 正交, 与筹码因子同法)

目的: 判断 Alpha101 因子是否对 GBDT 提供独立增量 (公平对比: 残差化后入模)。
"""
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression as _LR

ROOT = r"c:\Users\liuqi\quant_system_v2"
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()
panel = pd.read_parquet(os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet"))
a101 = pd.read_parquet(os.path.join(ROOT, "research", "sector_rotation", "alpha101_factor_panel.parquet"))

A101_SEL = ["alpha_045", "alpha_018", "alpha_038", "alpha_040", "alpha_025",
            "alpha_047", "alpha_015", "alpha_046", "alpha_010", "alpha_033"]

a101_sub = a101[["trade_date", "ts_code"] + A101_SEL].copy()
panel = panel.merge(a101_sub, on=["trade_date", "ts_code"], how="left")
print(f"[1] 面板合并: {len(panel):,} 行", flush=True)

PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "chip_shift_5"]
CHIP_BASE = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012"]
CHIP_RESID_COLS = ["vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]

GBDT_FEATS_BASE = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012",
                   "enh4_score", "vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]
GBDT_FEATS_RAW = GBDT_FEATS_BASE + A101_SEL
A101_RESID_COLS = [c + "_resid" for c in A101_SEL]
GBDT_FEATS_RESID = GBDT_FEATS_BASE + A101_RESID_COLS

PROC_COLS = PRICE_COLS + FIN_COLS + CHIP_COLS + A101_SEL


def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


def _ortho_resid(df, cols, base):
    """逐月截面 OLS: cols 对 base 正交化, 返回残差 (保留原符号)"""
    for c in cols:
        df[f"{c}_resid"] = np.nan
    for dt, grp in df.groupby("trade_date"):
        if len(grp) < 50:
            continue
        Xb = grp[base].values
        for c in cols:
            y = grp[c].values
            mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
            if mask.sum() < 50:
                continue
            lr = _LR(fit_intercept=True)
            lr.fit(Xb[mask], y[mask])
            resid = y - lr.predict(Xb)
            df.loc[grp.index[mask], f"{c}_resid"] = resid


def prep_feats(df):
    df = df.copy()
    df["has_fin"] = df["roe"].notna().astype(int)
    for c in PROC_COLS:
        df[c] = df.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    for c in PROC_COLS:
        df[c] = df.groupby("trade_date")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
    df[FIN_COLS] = df[FIN_COLS].fillna(-99.0)
    df[A101_SEL] = df[A101_SEL].fillna(0.0)
    df["enh4_score"] = (-0.40 * df["ivol"].rank(pct=True) - 0.35 * df["ret_1m"].rank(pct=True)
                        + 0.15 * df["roe"].rank(pct=True) + 0.05 * df["or_yoy"].rank(pct=True)
                        + 0.05 * df["netprofit_yoy"].rank(pct=True))
    # 筹码因子残差化 (负对齐方向, 与现有 pipeline 一致)
    _ortho_resid(df, CHIP_COLS, CHIP_BASE)
    for c in CHIP_COLS:
        df[f"{c}_resid"] = -df[f"{c}_resid"]
    # A101 因子残差化 (保留原符号)
    _ortho_resid(df, A101_SEL, CHIP_BASE)
    for c in CHIP_RESID_COLS + A101_RESID_COLS:
        df[c] = df.groupby("trade_date")[c].transform(lambda s: winsorize(s))
        df[c] = df.groupby("trade_date")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
    return df


def train_month_model(tr, feats, om):
    X, y = tr[feats].values, tr["fwd_20"].values
    val_months = sorted(tr["trade_date"].unique())[-3:]
    vm = tr["trade_date"].isin(val_months).values
    mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                            max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
    mdl.fit(X[~vm], y[~vm], eval_set=[(X[vm], y[vm])],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return pd.Series(mdl.predict(om[feats].values), index=om["ts_code"])


oos_months = [d for d in sorted(panel["trade_date"].unique()) if d >= 20230101]
arms = {"C8基线": GBDT_FEATS_BASE, "C8+原始A101": GBDT_FEATS_RAW, "C8+残差A101": GBDT_FEATS_RESID}
preds = {k: [] for k in arms}

for i, m in enumerate(oos_months):
    tr = prep_feats(panel[panel["trade_date"] < m]).sort_values("trade_date")
    om = prep_feats(panel[panel["trade_date"] == m])
    for k, feats in arms.items():
        p = train_month_model(tr, feats, om)
        preds[k].append(pd.DataFrame({"trade_date": m, "ts_code": om["ts_code"].values,
                                      "pred": p.reindex(om["ts_code"]).values,
                                      "fwd_20": om["fwd_20"].values}))
    if (i + 1) % 6 == 0 or i == len(oos_months) - 1:
        print(f"  WFO {i+1}/{len(oos_months)} 月, 耗时{time.time()-t0:.0f}s", flush=True)


def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, g in d.groupby("trade_date"):
        if len(g) < 50:
            continue
        out[dt] = g[factor].rank().corr(g[ret].rank())
    return pd.Series(out).dropna()


def qspread(d):
    d = d.copy()
    d["q"] = d.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]))
    piv = d.groupby("q", observed=True)["fwd_20"].mean()
    return piv.get("Q1", np.nan) - piv.get("Q5", np.nan)


rows = []
print("\n=== OOS (2023-2025, 36月) 三臂对比 ===")
for k, lst in preds.items():
    df = pd.concat(lst, ignore_index=True)
    ics = monthly_ic(df, "pred")
    icir = ics.mean() / (ics.std(ddof=1) + 1e-12) * np.sqrt(12)
    qs = qspread(df)
    rows.append({"版本": k, "OOS_IC": ics.mean(), "OOS_ICIR": icir,
                 "IC正率": (ics > 0).mean(), "Q1Q5_spread": qs, "月数": len(ics)})
    print(f"  {k:<14} IC={ics.mean():+.4f}  ICIR={icir:+.2f}  "
          f"正率={(ics>0).mean():.0%}  Q1-Q5={qs:+.3f}%", flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR, "alpha101_gbdt_wfo_compare.csv"), index=False, encoding="utf-8-sig")
print(f"\n[存] {os.path.join(OUT_DIR, 'alpha101_gbdt_wfo_compare.csv')}")
print(f"[完成] 总耗时 {time.time()-t0:.0f}s")
