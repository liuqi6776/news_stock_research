# -*- coding: utf-8 -*-
"""Alpha101 因子有效性验证

1. 月频 Rank IC / ICIR 有效性验证与排序
2. 周频(5日)/月频(20日)/季频(60日) 三档交易级别贡献对比
3. 输出汇总 CSV 供后续正交化验证使用
"""
import os
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "alpha101_factor_panel.parquet")
OUT_CSV = os.path.join(ROOT, "research", "sector_rotation", "alpha101_ic_summary.csv")

df = pd.read_parquet(PANEL)
factor_cols = [c for c in df.columns if c.startswith("alpha_")]
horizons = [("fwd_5", "周频(5日)"), ("fwd_20", "月频(20日)"), ("fwd_60", "季频(60日)")]

print(f"面板: {len(df):,} 行, {df['trade_date'].nunique()} 月, "
      f"{df['ts_code'].nunique()} 股, {len(factor_cols)} 因子")

# 因子空值率 (在有效样本内)
print("\n=== 因子空值率 (Top 15) ===")
na_rate = df[factor_cols].isna().mean().sort_values(ascending=False).head(15)
print(na_rate.to_string())

# 逐因子逐 horizon 逐月截面 Rank IC (Spearman)
rows = []
for f in factor_cols:
    for h, hname in horizons:
        ics = []
        for d, g in df.groupby("trade_date", sort=True):
            sub = g[[f, h]].dropna()
            if len(sub) < 30:
                continue
            rf = sub[f].rank().to_numpy(float)
            rh = sub[h].rank().to_numpy(float)
            rho = np.corrcoef(rf, rh)[0, 1]
            if np.isfinite(rho):
                ics.append(rho)
        ics = np.asarray(ics)
        if len(ics) < 6:
            continue
        mean_ic = ics.mean()
        std_ic = ics.std(ddof=1)
        icir = mean_ic / std_ic if std_ic > 1e-12 else np.nan
        tstat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic > 1e-12 else np.nan
        rows.append({
            "factor": f, "horizon": hname, "n_months": len(ics),
            "RankIC": mean_ic, "ICIR": icir, "t_stat": tstat,
            "pos_rate": float((ics > 0).mean()),
        })
res = pd.DataFrame(rows)

# ---- 月频有效性排序 ----
monthly = res[res["horizon"] == "月频(20日)"].sort_values("ICIR", ascending=False).reset_index(drop=True)
print("\n=== 月频有效性排序 (ICIR 降序, 前 30) ===")
cols_show = ["factor", "n_months", "RankIC", "ICIR", "t_stat", "pos_rate"]
print(monthly[cols_show].head(30).to_string(index=False))

# ---- 全量分布统计 ----
print("\n=== 月频有效性分布 (79 因子) ===")
m = monthly
print(f"  |RankIC|>=0.02: {(m['RankIC'].abs() >= 0.02).sum()} 个")
print(f"  |ICIR|>=0.5:   {(m['ICIR'].abs() >= 0.5).sum()} 个")
print(f"  |ICIR|>=1.0:   {(m['ICIR'].abs() >= 1.0).sum()} 个")
print(f"  |t_stat|>=2.0: {(m['t_stat'].abs() >= 2.0).sum()} 个")

# ---- 三档贡献对比 ----
print("\n=== 周/月/季三档贡献对比 (按月频 ICIR 排序, 前 20) ===")
top_factors = monthly.head(20)["factor"].tolist()
pivot = res[res["factor"].isin(top_factors)].pivot(
    index="factor", columns="horizon", values="RankIC")
pivot_icir = res[res["factor"].isin(top_factors)].pivot(
    index="factor", columns="horizon", values="ICIR")
order = [h[1] for h in horizons]
pivot = pivot[order]
pivot_icir = pivot_icir[order].add_prefix("ICIR_")
pivot = pivot.join(pivot_icir)
pivot = pivot.reindex(top_factors)
print(pivot.round(4).to_string())

# ---- 各 horizon 平均 IC 强度 ----
print("\n=== 三档 horizon 整体预测强度 (均值 |RankIC| / 均值 |ICIR|) ===")
for h, hname in horizons:
    sub = res[res["horizon"] == hname]
    print(f"  {hname}: mean|RankIC|={sub['RankIC'].abs().mean():.4f}, "
          f"median|RankIC|={sub['RankIC'].abs().median():.4f}, "
          f"mean|ICIR|={sub['ICIR'].abs().mean():.3f}")

res.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n[保存] {OUT_CSV}")
