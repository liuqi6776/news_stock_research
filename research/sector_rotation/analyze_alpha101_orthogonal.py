# -*- coding: utf-8 -*-
"""Alpha101 因子正交化验证

判断新因子对现有 GBDT 特征残差化后，是否仍有独立的预测能力(独立增量 alpha)。

方法:
  - 逐月截面 OLS: factor ~ [1, 现有22个特征], 取残差
  - 残差的月度 Rank IC (vs fwd_20) 与 ICIR
  - 若残差 ICIR 仍显著 (|ICIR|>=0.5 或 |t|>=2), 则该因子提供独立增量
"""
import os
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
A101 = os.path.join(ROOT, "research", "sector_rotation", "alpha101_factor_panel.parquet")
ML = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "alpha101_orthogonal_summary.csv")

# 现有 GBDT 特征 (排除元数据/标签)
# 正交化基线只用零缺失的量价/资金特征; 基本面(roe/or_yoy/netprofit_yoy)缺失率83%, 属另一信号域, 不作为量价因子正交化基线
EXISTING = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
            "volatility_5", "volatility_10", "volatility_20",
            "alpha_006", "alpha_009", "alpha_012", "alpha_023",
            "vwap_20", "float_pnl_20", "prof_pct_20", "chip_conc_20",
            "chip_shift_5", "pos_vol_20"]
OVERLAP = {"alpha_006", "alpha_009", "alpha_012", "alpha_023"}

a = pd.read_parquet(A101)
ml = pd.read_parquet(ML, columns=["trade_date", "ts_code"] + EXISTING)
factor_cols = [c for c in a.columns if c.startswith("alpha_")]

# merge on (trade_date, ts_code), 用 A101 的 fwd_20 作为标签
df = a.merge(ml, on=["trade_date", "ts_code"], how="inner", suffixes=("", "_ml"))
print(f"合并样本: {len(df):,} 行, {df['trade_date'].nunique()} 月, "
      f"{df['ts_code'].nunique()} 股")

def monthly_ic(factor_vals, label_vals, td):
    """逐月截面 Rank IC -> (ic_array)"""
    ics = []
    for d in np.unique(td):
        m = td == d
        f = factor_vals[m]
        l = label_vals[m]
        ok = np.isfinite(f) & np.isfinite(l)
        if ok.sum() < 30:
            continue
        rf = pd.Series(f[ok]).rank().to_numpy(float)
        rl = pd.Series(l[ok]).rank().to_numpy(float)
        rho = np.corrcoef(rf, rl)[0, 1]
        if np.isfinite(rho):
            ics.append(rho)
    return np.asarray(ics)

def ic_stats(ics):
    if len(ics) < 6:
        return (np.nan, np.nan, np.nan, len(ics))
    m = ics.mean(); s = ics.std(ddof=1)
    icir = m / s if s > 1e-12 else np.nan
    t = m / (s / np.sqrt(len(ics))) if s > 1e-12 else np.nan
    return (m, icir, t, len(ics))

td = df["trade_date"].to_numpy()
fwd = df["fwd_20"].to_numpy(float)
# 现有特征矩阵 (含截距), 用于截面 OLS
X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in EXISTING])

rows = []
for f in factor_cols:
    fv = df[f].to_numpy(float)
    # 原始 IC
    ic_orig = monthly_ic(fv, fwd, td)
    m0, icir0, t0, n0 = ic_stats(ic_orig)

    # 截面 OLS 残差
    resid = np.full_like(fv, np.nan)
    for d in np.unique(td):
        m = td == d
        idx = np.where(m)[0]
        y = fv[m]
        Xm = X[m]
        # 仅保留 y 和 X 均无 NaN 的行
        ok = np.isfinite(y) & np.isfinite(Xm).all(axis=1)
        if ok.sum() < len(EXISTING) + 2:
            continue
        yy = y[ok]
        XX = Xm[ok]
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        resid[idx[ok]] = yy - XX @ beta

    ic_resid = monthly_ic(resid, fwd, td)
    m1, icir1, t1, n1 = ic_stats(ic_resid)

    rows.append({
        "factor": f,
        "is_overlap": f in OVERLAP,
        "orig_RankIC": m0, "orig_ICIR": icir0, "orig_t": t0,
        "resid_RankIC": m1, "resid_ICIR": icir1, "resid_t": t1,
        "n_months": n1,
    })

res = pd.DataFrame(rows).sort_values("resid_ICIR", ascending=False).reset_index(drop=True)

print("\n=== 正交化后仍具独立增量的因子 (resid_ICIR 降序, 前 30) ===")
show = ["factor", "is_overlap", "orig_RankIC", "orig_ICIR",
        "resid_RankIC", "resid_ICIR", "resid_t", "n_months"]
print(res[show].head(30).to_string(index=False))

print("\n=== 独立增量统计 (79 因子) ===")
valid = res[res["n_months"] >= 6]
print(f"  |resid_ICIR|>=0.5: {(valid['resid_ICIR'].abs() >= 0.5).sum()} 个")
print(f"  |resid_ICIR|>=0.4: {(valid['resid_ICIR'].abs() >= 0.4).sum()} 个")
print(f"  |resid_t|>=2.0:    {(valid['resid_t'].abs() >= 2.0).sum()} 个")
# IC 衰减
valid2 = valid[np.isfinite(valid["orig_ICIR"]) & (valid["orig_ICIR"].abs() > 0.05)]
if len(valid2):
    decay = (valid2["resid_ICIR"].abs() / valid2["orig_ICIR"].abs()).median()
    print(f"  中位 ICIR 保留率 (resid/orig): {decay:.2%}")

res.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n[保存] {OUT}")
