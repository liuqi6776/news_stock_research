# -*- coding: utf-8 -*-
"""改进版大单净流入因子: 绝对值标准化 + 市值中性化 + 行业中性化 + 动量过滤

因子设计 (按用户建议):
  1. mf_abs_ratio_20:  20日累计大单净流入 / 20日累计|大单净流入|  (方向一致性)
  2. mf_lg_net_20:     20日大单+特大单净流入 (绝对值标准化)
  3. mf_neutral_20:    mf_abs_ratio_20 经市值+行业中性化
  4. mf_mom_filt:      mf_neutral_20 * sign(momentum_20)  (动量过滤)
  5. mf_high_ret:      高收益日大单净流入占比 (仅在涨幅>3%日累计)
  6. mf_5d:            5日版本 (短周期)
"""
import os, sys, time, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
t0 = time.time()

# === 1. 加载 moneyflow1 日频 ===
print("[1] 加载 moneyflow1...", flush=True)
fs = sorted(glob.glob(os.path.join(DATA, "moneyflow1", "*.parquet")))
fs = [f for f in fs if os.path.getsize(f) > 1024]
parts = []
for f in fs:
    d = os.path.basename(f)[:8]
    if d < "20190601": continue
    df = pd.read_parquet(f, columns=["ts_code","trade_date",
        "buy_lg_amount","sell_lg_amount","buy_elg_amount","sell_elg_amount","net_mf_amount"])
    parts.append(df)
mf = pd.concat(parts, ignore_index=True)
mf["trade_date"] = mf["trade_date"].astype(int)
mf = mf.drop_duplicates(subset=["ts_code","trade_date"])
# 大单+特大单净流入 (每日)
mf["lg_net"] = (mf["buy_lg_amount"] - mf["sell_lg_amount"]) + \
               (mf["buy_elg_amount"] - mf["sell_elg_amount"])
mf["lg_abs"] = mf["lg_net"].abs()
mf["lg_total"] = (mf["buy_lg_amount"] + mf["sell_lg_amount"] +
                  mf["buy_elg_amount"] + mf["sell_elg_amount"])
print(f"    {len(mf):,} 行, {mf['trade_date'].min()}~{mf['trade_date'].max()}, {time.time()-t0:.0f}s", flush=True)

# === 2. 加载日频行情 (用于动量过滤和高收益日) ===
print("[2] 加载日频行情...", flush=True)
px_parts = []
for f in sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet"))):
    if os.path.getsize(f) <= 1024: continue
    d = os.path.basename(f)[:8]
    if d < "20190601": continue
    df = pd.read_parquet(f, columns=["ts_code","trade_date","pct_chg","close","amount"])
    px_parts.append(df)
px = pd.concat(px_parts, ignore_index=True)
px["trade_date"] = px["trade_date"].astype(int)
px["r"] = px["pct_chg"] / 100.0
px = px[["ts_code","trade_date","r","close","amount"]]
print(f"    {len(px):,} 行, {time.time()-t0:.0f}s", flush=True)

# === 3. 合并 + 构建因子 ===
print("[3] 构建改进版因子...", flush=True)
df = mf.merge(px[["ts_code","trade_date","r","close","amount"]], on=["ts_code","trade_date"], how="left")
df = df.sort_values(["ts_code","trade_date"]).reset_index(drop=True)

# 3a. 20日累计大单净流入 / 20日累计|大单净流入|  (方向一致性, ∈ [-1, 1])
g = df.groupby("ts_code")
df["mf_abs_ratio_20"] = g["lg_net"].transform(lambda s: s.rolling(20, min_periods=10).sum()) / \
                        (g["lg_abs"].transform(lambda s: s.rolling(20, min_periods=10).sum()) + 1e-6)
df["mf_abs_ratio_5"] = g["lg_net"].transform(lambda s: s.rolling(5, min_periods=3).sum()) / \
                       (g["lg_abs"].transform(lambda s: s.rolling(5, min_periods=3).sum()) + 1e-6)

# 3b. 20日大单净流入 (绝对值标准化, 单位: 万元)
df["mf_lg_net_20"] = g["lg_net"].transform(lambda s: s.rolling(20, min_periods=10).sum())
df["mf_lg_net_5"] = g["lg_net"].transform(lambda s: s.rolling(5, min_periods=3).sum())
# 截面 zscore
for c in ["mf_lg_net_20","mf_lg_net_5"]:
    df[c] = df.groupby("trade_date")[c].transform(lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))

# 3c. 高收益日大单净流入 (仅在涨幅>3%日累计)
df["high_ret_lg_net"] = df["lg_net"] * (df["r"] > 0.03).astype(float)
df["mf_high_ret_20"] = g["high_ret_lg_net"].transform(lambda s: s.rolling(20, min_periods=5).sum())
df["mf_high_ret_20"] = df.groupby("trade_date")["mf_high_ret_20"].transform(
    lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))

# 3d. 动量过滤: mf_abs_ratio_20 * sign(momentum_20)
df["momentum_20"] = g["close"].transform(lambda s: s.pct_change(20))
df["mf_mom_filt"] = df["mf_abs_ratio_20"] * np.sign(df["momentum_20"])

# 3e. 市值 (用 amount 近似, 非严格但够做中性化)
df["ln_amount"] = np.log(df.groupby("trade_date")["amount"].transform(
    lambda s: s.rolling(5, min_periods=3).mean()) + 1e-6)

print(f"    因子构建完成, {time.time()-t0:.0f}s", flush=True)

# === 4. 市值+行业中性化 ===
print("[4] 市值+行业中性化...", flush=True)
im = pd.read_parquet(os.path.join(ROOT, "research", "studies", "study_008_enhancements",
                                  "data", "industry_map.parquet"))
ind_map = dict(zip(im["ts_code"], im["industry"]))
df["industry"] = df["ts_code"].map(ind_map).fillna("其他")

# 对 mf_abs_ratio_20 做市值+行业中性化
def neutralize(df, factor_col, date_col="trade_date"):
    """截面回归: factor ~ ln_amount + industry dummies, 取残差"""
    residuals = pd.Series(index=df.index, dtype=float)
    for dt, grp in df.groupby(date_col):
        if len(grp) < 100: continue
        y = grp[factor_col].values
        mask = np.isfinite(y) & np.isfinite(grp["ln_amount"].values)
        if mask.sum() < 50: continue
        X_list = [grp["ln_amount"].values[mask]]
        # industry dummies
        inds = grp["industry"].values[mask]
        uniq = sorted(set(inds))
        for ind in uniq[1:]:  # drop first
            X_list.append((inds == ind).astype(float))
        X = np.column_stack(X_list)
        lr = LinearRegression().fit(X, y[mask])
        pred = lr.predict(X)
        res = y[mask] - pred
        residuals.loc[grp.index[mask]] = res
    return residuals

df["mf_neutral_20"] = neutralize(df, "mf_abs_ratio_20")
df["mf_neutral_5"] = neutralize(df, "mf_abs_ratio_5")
df["mf_neu_mom"] = df["mf_neutral_20"] * np.sign(df["momentum_20"])
print(f"    中性化完成, {time.time()-t0:.0f}s", flush=True)

# === 5. 月末快照 + IC 检验 ===
print("\n[5] 月末快照 + IC 检验 (2023+ OOS)...", flush=True)
# 月末快照
df["month"] = df["trade_date"] // 100
df_m = df.sort_values(["ts_code","trade_date"]).drop_duplicates(subset=["ts_code","month"], keep="last")

# 加载 fwd_20 (从原始面板)
panel = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_72m.parquet"))
panel["month"] = panel["trade_date"] // 100
fwd = panel[["ts_code","month","fwd_20"]].drop_duplicates(subset=["ts_code","month"], keep="last")

df_m = df_m.merge(fwd, on=["ts_code","month"], how="inner")
oos = sorted([m for m in df_m["month"].unique() if m >= 202301])
print(f"    月末快照: {len(df_m):,} 行, OOS {len(oos)} 月", flush=True)

FACTORS = ["mf_abs_ratio_20","mf_abs_ratio_5","mf_lg_net_20","mf_lg_net_5",
           "mf_high_ret_20","mf_mom_filt","mf_neutral_20","mf_neutral_5","mf_neu_mom"]

print(f"\n{'因子':>20} {'IC':>8} {'ICIR':>8} {'正率':>6}")
print("-"*50)
ic_results = {}
for f in FACTORS:
    ics = []
    for m in oos:
        g = df_m[df_m["month"] == m]
        if len(g) < 50: continue
        ic = g[f].rank().corr(g["fwd_20"].rank())
        if np.isfinite(ic): ics.append(ic)
    s = pd.Series(ics)
    ic, icir, pos = s.mean(), s.mean()/(s.std()+1e-12)*np.sqrt(12), s.gt(0).mean()
    ic_results[f] = {"ic": ic, "icir": icir, "pos": pos}
    print(f"{f:>20} {ic:>+8.4f} {icir:>+8.2f} {pos*100:>5.0f}%")

# === 6. 最佳因子加入 C8 GBDT ===
print(f"\n[6] 最佳大单因子 + C8 GBDT IC...", flush=True)
import lightgbm as lgb
from sklearn.linear_model import LinearRegression as _LR

C8_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012",
           "enh4_score","vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
CHIP_BASE = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
CHIP_RAW = ["vwap_20","float_pnl_20","chip_shift_5"]

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

# 从 orig 面板建 C8
orig = panel.copy()
PRICE_COLS = ["ret_1m","ivol","momentum_5","momentum_10","momentum_20","momentum_60",
              "volatility_5","volatility_10","volatility_20","alpha_006","alpha_009","alpha_012","alpha_023"]
FIN_COLS = ["roe","or_yoy","netprofit_yoy"]
for c in PRICE_COLS + FIN_COLS + CHIP_RAW:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))
orig[FIN_COLS] = orig[FIN_COLS].fillna(-99.0)
gg = orig.groupby("trade_date")
orig["enh4_score"] = (-0.40*gg["ivol"].rank(pct=True) - 0.35*gg["ret_1m"].rank(pct=True)
                      + 0.15*gg["roe"].rank(pct=True) + 0.05*gg["or_yoy"].rank(pct=True)
                      + 0.05*gg["netprofit_yoy"].rank(pct=True))
for c in CHIP_RAW: orig[f"{c}_resid"] = np.nan
for dt, grp in orig.groupby("trade_date"):
    if len(grp) < 50: continue
    Xb = grp[CHIP_BASE].values
    for c in CHIP_RAW:
        y = grp[c].values
        mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
        if mask.sum() < 50: continue
        lr = _LR().fit(Xb[mask], y[mask])
        orig.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
for c in ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))

# merge 大单因子到 orig
mf_keep = df_m[["ts_code","month"] + [f for f in FACTORS if f in df_m.columns]].copy()
mf_keep["month"] = mf_keep["month"].astype("int64")
orig["month"] = orig["trade_date"].astype("int64") // 100
orig = orig.merge(mf_keep, on=["ts_code","month"], how="left")
for f in FACTORS:
    if f in orig.columns:
        orig[f] = orig.groupby("trade_date")[f].transform(lambda s: winsorize(s))
        orig[f] = orig.groupby("trade_date")[f].transform(lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))
        orig[f] = orig[f].fillna(0.0)

orig["fwd_20"] = orig.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
orig = orig.dropna(subset=C8_COLS + ["fwd_20"])

# 选 IC 前3的大单因子
top_mf = sorted(ic_results.items(), key=lambda x: x[1]["ic"], reverse=True)[:3]
top_mf_cols = [x[0] for x in top_mf if x[1]["ic"] > 0]
print(f"    正IC大单因子: {top_mf_cols}", flush=True)

if top_mf_cols:
    COMBO = {"C8": C8_COLS, "C8+MF_best": C8_COLS + top_mf_cols}
    oos_m = sorted([m for m in orig["trade_date"].unique() if m >= 20230101])
    for name, feats in COMBO.items():
        feats_r = [c for c in feats if c in orig.columns]
        preds = []
        for m in oos_m:
            tr = orig[orig["trade_date"] < m].sort_values("trade_date")
            val_months = sorted(tr["trade_date"].unique())[-3:]
            vm = tr["trade_date"].isin(val_months).values
            X, y = tr[feats_r].values, tr["fwd_20"].values
            Xtr, ytr, Xv, yv = X[~vm], y[~vm], X[vm], y[vm]
            mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                                    max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                                    subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
            mdl.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
            om = orig[orig["trade_date"] == m]
            preds.append(pd.DataFrame({"trade_date":m, "pred":mdl.predict(om[feats_r].values),
                                        "fwd_20":om["fwd_20"].values}))
        df_p = pd.concat(preds, ignore_index=True)
        ics = []
        for dt, gg in df_p.groupby("trade_date"):
            if len(gg) < 50: continue
            ics.append(gg["pred"].rank().corr(gg["fwd_20"].rank()))
        s = pd.Series(ics)
        print(f"  {name:>16} ({len(feats_r)}因子): IC={s.mean():+.4f} ICIR={s.mean()/(s.std()+1e-12)*np.sqrt(12):+.2f} 正率={s.gt(0).mean()*100:.0f}%", flush=True)

# 保存改进版因子
df_m_keep = df_m[["ts_code","month"] + FACTORS].copy()
df_m_keep = df_m_keep.rename(columns={"month": "trade_date"})
OUT = os.path.join(ROOT, "research/sector_rotation/stock_mf_improved_72m.parquet")
df_m_keep.to_parquet(OUT, index=False)
print(f"\n[保存] {OUT}, {time.time()-t0:.0f}s", flush=True)
