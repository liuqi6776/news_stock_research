# -*- coding: utf-8 -*-
"""C8 因子集截面相关性诊断 + 冗余排除

1. 逐月截面 Spearmann 相关矩阵 → 平均/最大
2. 高相关对（|r|>0.7）逐一列出，给出冗余候选
3. 聚类 dendrogram 可视化
4. VIF（方差膨胀因子）诊断多重共线性
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "c8_corr_diag.csv")
OUT_PNG = os.path.join(ROOT, "research", "sector_rotation", "results", "c8_corr_heatmap.png")

C7_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]
CHIP_COLS = ["vwap_20","float_pnl_20","chip_shift_5"]
CHIP_RESID = ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
BASE_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
C8_COLS = C7_COLS + CHIP_RESID

def zscore(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

t0 = time.time()
panel = pd.read_parquet(PANEL)

# 复刻回测引擎的数据预处理：ENH4打分 → 标准化 → 筹码残差化
for c in ["roe","or_yoy"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["roe"] = panel["roe"].fillna(-99.0)
panel["or_yoy"] = panel["or_yoy"].fillna(-99.0)
g = panel.groupby("trade_date")
panel["enh4_score"] = (-0.40 * g["ivol"].rank(pct=True)
                       -0.35 * g["ret_1m"].rank(pct=True)
                       +0.15 * g["roe"].rank(pct=True)
                       +0.05 * g["or_yoy"].rank(pct=True))

ALL_RAW = list(set(C7_COLS + CHIP_COLS) - {"enh4_score"})
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel = panel.dropna(subset=C7_COLS + CHIP_COLS)

# 筹码残差化
for c in CHIP_COLS:
    panel[f"{c}_resid"] = np.nan
for dt, grp in panel.groupby("trade_date"):
    if len(grp) < 50: continue
    Xb = grp[BASE_COLS].values
    for c in CHIP_COLS:
        y = grp[c].values
        mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
        if mask.sum() < 50: continue
        lr = LinearRegression(fit_intercept=True)
        lr.fit(Xb[mask], y[mask])
        panel.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
for c in CHIP_RESID:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel = panel.dropna(subset=C8_COLS)
print(f"[数据] {len(panel):,} 行, {panel['trade_date'].nunique()} 月, {time.time()-t0:.0f}s")

# ========== 1. 逐月截面 Spearmann 相关矩阵（用 rank 后 Pearson = Spearmann）==========
print("\n=== 1. 因子截面 Spearmann 相关性 (逐月平均) ===")
months = sorted(panel["trade_date"].unique())
mats = []
for dt in months:
    g = panel[panel["trade_date"] == dt][C8_COLS].rank()
    mats.append(g.corr().values)
mat_mean = np.mean(mats, axis=0)
mat_max = np.max(np.abs(mats), axis=0)
df_mean = pd.DataFrame(mat_mean, index=C8_COLS, columns=C8_COLS)
df_max = pd.DataFrame(mat_max, index=C8_COLS, columns=C8_COLS)

print("\n[平均相关矩阵]")
print(df_mean.round(3).to_string())
print("\n[最大绝对相关矩阵]")
print(df_max.round(3).to_string())

# 列出所有高相关对（|平均 r| > 0.5 分级报告）
pairs = []
for i in range(len(C8_COLS)):
    for j in range(i+1, len(C8_COLS)):
        r_avg = mat_mean[i, j]
        r_max = mat_max[i, j]
        pairs.append((C8_COLS[i], C8_COLS[j], r_avg, r_max))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)

rows = []
thresh_labels = [(0.7, "⚠️ 极高度相关"), (0.5, "🔶 高度相关"), (0.3, "🟡 中度相关")]
print("\n=== 2. 高相关因子对（按平均 r 绝对值降序） ===")
for f1, f2, r_avg, r_max in pairs:
    level = next((lbl for th, lbl in thresh_labels if abs(r_avg) >= th), "🟢 低相关")
    flag = "❗ REDUNDANT_CANDIDATE" if abs(r_avg) >= 0.7 else ""
    print(f"  {level:>12} {f1:>20} ↔ {f2:<20} r_avg={r_avg:+.3f} r_max|r|={r_max:.3f} {flag}")
    rows.append({"f1": f1, "f2": f2, "r_avg": r_avg, "r_max_abs": r_max, "level": level})

pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# ========== 3. VIF（方差膨胀因子） ==========
print("\n=== 3. VIF 多重共线性诊断（OOS 2023+ 平均） ===")
oos_months = [m for m in months if m >= 20230101]
vifs_all = {c: [] for c in C8_COLS}
for dt in oos_months:
    X = panel[panel["trade_date"] == dt][C8_COLS].values
    n, k = X.shape
    if k != len(C8_COLS): continue
    for j, c in enumerate(C8_COLS):
        X_j = np.delete(X, j, axis=1)
        y_j = X[:, j]
        lr = LinearRegression()
        lr.fit(X_j, y_j)
        r2 = max(0.0, min(0.9999, lr.score(X_j, y_j)))
        vifs_all[c].append(1.0 / (1.0 - r2))

vif_df = pd.DataFrame({
    "VIF_mean": [np.mean(vifs_all[c]) for c in C8_COLS],
    "VIF_p90": [np.percentile(vifs_all[c], 90) for c in C8_COLS],
    "VIF_max": [np.max(vifs_all[c]) for c in C8_COLS],
}, index=C8_COLS).sort_values("VIF_mean", ascending=False)
print(vif_df.round(2).to_string())
print(f"\n  VIF 解读: <5 安全, 5-10 轻度共线, >10 严重多重共线(需排除)")
vif_violators = vif_df[vif_df["VIF_mean"] > 10].index.tolist()
if vif_violators:
    print(f"  ❌ VIF 超标因子: {vif_violators} → 候选排除")
else:
    print(f"  ✅ 所有因子 VIF 均在安全区间")

# ========== 4. 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
mask = np.triu(np.ones_like(mat_mean, dtype=bool), k=1)
sns.heatmap(df_mean, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-0.8, vmax=0.8, square=True, linewidths=0.5, ax=axes[0])
axes[0].set_title("C8 因子截面 Spearmann 相关（逐月平均）", fontsize=12, fontweight="bold")
sns.heatmap(df_max, mask=mask, annot=True, fmt=".2f", cmap="Reds",
            vmin=0, vmax=0.9, square=True, linewidths=0.5, ax=axes[1])
axes[1].set_title("C8 因子 |r| 历史最大值", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[可视化] {OUT_PNG}")

# ========== 5. 冗余排除建议 ==========
print("\n=== 4. 冗余排除建议（若存在 r_avg>0.7 或 VIF>10） ===")
redundant_pairs = [(r["f1"], r["f2"], r["r_avg"]) for r in rows if abs(r["r_avg"]) >= 0.7]
if redundant_pairs:
    for f1, f2, r in redundant_pairs:
        # 单因子 IC 低的那个是候选排除
        ic_vals = {}
        for f in [f1, f2]:
            ics = []
            for dt in oos_months:
                gg = panel[panel["trade_date"] == dt]
                ics.append(gg[f].rank().corr(gg["fwd_20"].rank()))
            ic_vals[f] = np.nanmean(ics)
        drop = f1 if abs(ic_vals[f1]) < abs(ic_vals[f2]) else f2
        print(f"  ({f1} IC={ic_vals[f1]:+.4f}) ↔ ({f2} IC={ic_vals[f2]:+.4f}), r={r:+.3f}")
        print(f"    → 建议排除: {drop}（IC 较小者）")
else:
    print("  ✅ 无极高度相关（|r|≥0.7），C8 无明显冗余对")

print(f"\n总耗时 {time.time()-t0:.0f}s")
