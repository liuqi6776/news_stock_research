# -*- coding: utf-8 -*-
"""情绪因子(涨停封单)进 GBDT 的滚动 WFO 对比诊断 (三臂)

臂1: C8 基线 (现有)
臂2: C8 + zt_fd_amount_mean (封单金额均值, 预检中唯一正交候选)
臂3: C8 + zt_cnt_1m (月涨停次数, 作对照——预期被 ivol/momentum 覆盖)

复用 diag_alpha101_gbdt_wfo.py 的 C8 预处理 + 滚动 WFO 训练口径 (2023-2025, 36月 OOS)。
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
SENT = os.path.join(ROOT, "research", "sector_rotation", "data", "sentiment")

t0 = time.time()
panel = pd.read_parquet(os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet"))

# ---------- 构建月度情绪特征 (PIT 安全: 只用当月涨停事件) ----------
lim = pd.read_parquet(os.path.join(SENT, "limit_list_d.parquet"))
lim["ym"] = lim["trade_date"].astype(str).str[:6]
zt = lim[lim["limit"] == "U"]
zt_agg = zt.groupby(["ym", "ts_code"]).agg(
    zt_cnt_1m=("trade_date", "count"),
    zt_fd_amount_mean=("fd_amount", "mean"),
    zt_limit_times_max=("limit_times", "max"),
).reset_index()

panel["ym"] = panel["trade_date"].astype(str).str[:6]
panel = panel.merge(zt_agg, on=["ym", "ts_code"], how="left")
SENT_SEL = ["zt_cnt_1m", "zt_fd_amount_mean", "zt_limit_times_max"]
for c in SENT_SEL:
    panel[c] = panel[c].fillna(0.0)
print(f"[1] 面板合并情绪特征: {len(panel):,} 行, 耗时{time.time()-t0:.0f}s", flush=True)

PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "chip_shift_5"]
CHIP_BASE = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012"]
CHIP_RESID_COLS = ["vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]

GBDT_FEATS_BASE = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012",
                   "enh4_score", "vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]
GBDT_FEATS_FD = GBDT_FEATS_BASE + ["zt_fd_amount_mean"]
GBDT_FEATS_CNT = GBDT_FEATS_BASE + ["zt_cnt_1m"]

PROC_COLS = PRICE_COLS + FIN_COLS + CHIP_COLS + SENT_SEL


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
    df[SENT_SEL] = df[SENT_SEL].fillna(0.0)
    df["enh4_score"] = (-0.40 * df["ivol"].rank(pct=True) - 0.35 * df["ret_1m"].rank(pct=True)
                        + 0.15 * df["roe"].rank(pct=True) + 0.05 * df["or_yoy"].rank(pct=True)
                        + 0.05 * df["netprofit_yoy"].rank(pct=True))
    _ortho_resid(df, CHIP_COLS, CHIP_BASE)
    for c in CHIP_COLS:
        df[f"{c}_resid"] = -df[f"{c}_resid"]
    for c in CHIP_RESID_COLS:
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
arms = {"C8基线": GBDT_FEATS_BASE, "C8+封单金额": GBDT_FEATS_FD, "C8+涨停次数": GBDT_FEATS_CNT}
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
res.to_csv(os.path.join(OUT_DIR, "sentiment_gbdt_wfo_compare.csv"), index=False, encoding="utf-8-sig")
print(f"\n[存] {os.path.join(OUT_DIR, 'sentiment_gbdt_wfo_compare.csv')}")
print(f"[完成] 总耗时 {time.time()-t0:.0f}s")
