# -*- coding: utf-8 -*-
"""
验证脚本：检查 WF 数据正确性 + ROE12 在训练期 432 组中的综合排名
"""
import os, pickle as _pkl
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
SR = os.path.join(ROOT, "research", "sector_rotation")
OUT = os.path.join(SR, "results")

# ============================================================
# 1. 加载网格缓存
# ============================================================
cache = os.path.join(OUT, "_v5_wf_grid_cache.pkl")
with open(cache, "rb") as f:
    cfg_rows, cfg_navs = _pkl.load(f)
df = pd.DataFrame(cfg_rows)
print(f"432 组配置加载完成: {len(df)} 组\n")

TRAIN_START = pd.Timestamp("2020-01-01")
R1_TRAIN_END = pd.Timestamp("2024-01-01")  # R1 训练: 2020-2023
R2_TRAIN_END = pd.Timestamp("2025-01-01")  # R2 训练: 2020-2024

# ============================================================
# 2. 检查 WF 拼接的回撤是否正确
# ============================================================
print("=" * 100)
print("【1. WF 拼接回撤验证】")
print("=" * 100)

# R1 选中: ROE10, R2 选中: ROE8
r1_best_label = "K3_ROE10_PEG2.5_CHIP40_MV50_YR3"
r2_best_label = "K3_ROE8_PEG2.5_CHIP40_MV50_YR3"

nv_r1, p_r1 = cfg_navs[r1_best_label]
nv_r2, p_r2 = cfg_navs[r2_best_label]

# R1 执行期 (2024)
r1_exec = nv_r1[(nv_r1.index >= pd.Timestamp("2024-01-01")) & (nv_r1.index < pd.Timestamp("2025-01-01"))]
r1_exec_rb = r1_exec / r1_exec.iloc[0]
r1_mdd = ((r1_exec_rb - r1_exec_rb.cummax()) / r1_exec_rb.cummax()).min()
r1_ret = r1_exec_rb.iloc[-1] - 1

# R2 执行期 (2025)
r2_exec = nv_r2[(nv_r2.index >= pd.Timestamp("2025-01-01")) & (nv_r2.index < pd.Timestamp("2026-01-01"))]
r2_exec_rb = r2_exec / r2_exec.iloc[0]
r2_mdd = ((r2_exec_rb - r2_exec_rb.cummax()) / r2_exec_rb.cummax()).min()
r2_ret = r2_exec_rb.iloc[-1] - 1

# WF 拼接
wf_r1 = r1_exec_rb.copy()
wf_r2 = r2_exec_rb * wf_r1.iloc[-1]
wf_all = pd.concat([wf_r1, wf_r2]).sort_index()
wf_mdd = ((wf_all - wf_all.cummax()) / wf_all.cummax()).min()

print(f"\nR1 (2024, ROE10): 累计{r1_ret:.1%} 回撤{r1_mdd:.1%} 期末{r1_exec_rb.iloc[-1]:.4f}")
print(f"R2 (2025, ROE8):  累计{r2_ret:.1%} 回撤{r2_mdd:.1%} 期末{r2_exec_rb.iloc[-1]:.4f}")
print(f"WF 拼接: 期末{wf_all.iloc[-1]:.4f} 回撤{wf_mdd:.1%}")
print(f"  → WF 回撤 = max(R1内部回撤, 从R1最高点到R2最低点的回撤)")
print(f"  → R1最高点: {wf_all.cummax().max():.4f}, WF最低点: {wf_all.min():.4f}")

# ============================================================
# 3. ROE12 在训练期 432 组中的综合排名
# ============================================================
print(f"\n{'=' * 100}")
print("【2. ROE12 配置在训练期 432 组中的综合排名】")
print("=" * 100)

# ROE12 配置: K3_ROE12_PEG2.0_CHIP50_MV50_YR3
roe12_label = "K3_ROE12_PEG2.0_CHIP50_MV50_YR3"

# R2 训练期 (2020-2024) 排名 — 这是最关键的
print(f"\n--- R2 训练期 (2020-01 ~ 2024-12, 60月) 排名 ---")
df_r2 = df.sort_values("R2train_夏普", ascending=False).reset_index(drop=True)
roe12_r2_rank = df_r2[df_r2["配置"] == roe12_label].index[0] + 1
print(f"ROE12 按 Sharpe 排名: #{roe12_r2_rank} / {len(df_r2)}")
print(f"  训练 Sharpe: {df_r2[df_r2['配置']==roe12_label]['R2train_夏普'].values[0]:.4f}")
print(f"  训练 年化: {df_r2[df_r2['配置']==roe12_label]['R2train_年化'].values[0]:.1%}")
print(f"  训练 回撤: {df_r2[df_r2['配置']==roe12_label]['R2train_回撤'].values[0]:.1%}")

# 按 年化 排名
df_r2_ann = df.sort_values("R2train_年化", ascending=False).reset_index(drop=True)
roe12_ann_rank = df_r2_ann[df_r2_ann["配置"] == roe12_label].index[0] + 1
print(f"ROE12 按 年化 排名: #{roe12_ann_rank} / {len(df_r2_ann)}")

# 按回撤排名（回撤越小越好）
df_r2_mdd = df.sort_values("R2train_回撤", ascending=False).reset_index(drop=True)  # 最大(最接近0)在前
roe12_mdd_rank = df_r2_mdd[df_r2_mdd["配置"] == roe12_label].index[0] + 1
print(f"ROE12 按 回撤 排名: #{roe12_mdd_rank} / {len(df_r2_mdd)} (回撤越小越好)")

# 综合分: Sharpe + 年化排名 + 回撤排名 的平均排名
df["_r2_sharpe_rank"] = df["R2train_夏普"].rank(ascending=False)
df["_r2_ann_rank"] = df["R2train_年化"].rank(ascending=False)
df["_r2_mdd_rank"] = df["R2train_回撤"].rank(ascending=False)  # 越大越好(越接近0)
df["_r2_composite_rank"] = (df["_r2_sharpe_rank"] + df["_r2_ann_rank"] + df["_r2_mdd_rank"]) / 3
df_r2_comp = df.sort_values("_r2_composite_rank").reset_index(drop=True)
roe12_comp_rank = df_r2_comp[df_r2_comp["配置"] == roe12_label].index[0] + 1
print(f"ROE12 综合 排名(Sharpe+年化+回撤平均): #{roe12_comp_rank} / {len(df_r2_comp)}")

# Top 10 综合排名
print(f"\n--- R2 训练期综合排名 Top15 ---")
print(f"{'排名':>4} {'配置':<40} {'Sharpe':>8} {'年化':>8} {'回撤':>8} {'综合排名':>8}")
for i in range(15):
    r = df_r2_comp.iloc[i]
    print(f"{i+1:>4} {r['配置']:<40} {r['R2train_夏普']:>8.2f} {r['R2train_年化']:>7.1%} {r['R2train_回撤']:>7.1%} {r['_r2_composite_rank']:>8.1f}")

# ROE12 附近排名
print(f"\n--- ROE12 附近排名 (#{roe12_comp_rank-2} ~ #{roe12_comp_rank+2}) ---")
for i in range(max(0, roe12_comp_rank-3), min(len(df_r2_comp), roe12_comp_rank+2)):
    r = df_r2_comp.iloc[i]
    print(f"{i+1:>4} {r['配置']:<40} {r['R2train_夏普']:>8.2f} {r['R2train_年化']:>7.1%} {r['R2train_回撤']:>7.1%}")

# ============================================================
# 4. ROE12 在各子区间的表现（不只看 2025）
# ============================================================
print(f"\n{'=' * 100}")
print("【3. ROE12 vs ROE8 逐年表现对比】")
print("=" * 100)

roe8_label = "K3_ROE8_PEG2.5_CHIP40_MV50_YR3"
nv_12, _ = cfg_navs[roe12_label]
nv_8, _ = cfg_navs[roe8_label]

years = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026)]
print(f"\n{'年份':<10} {'ROE12 年化':>12} {'ROE12 回撤':>12} {'ROE12 夏普':>12} | {'ROE8 年化':>12} {'ROE8 回撤':>12} {'ROE8 夏普':>12}")
print("-" * 100)
for ys, ye in years:
    s12 = nv_12[(nv_12.index >= pd.Timestamp(f"{ys}-01-01")) & (nv_12.index < pd.Timestamp(f"{ye}-01-01"))]
    s8 = nv_8[(nv_8.index >= pd.Timestamp(f"{ys}-01-01")) & (nv_8.index < pd.Timestamp(f"{ye}-01-01"))]
    if len(s12) < 10 or len(s8) < 10:
        continue
    s12 = s12 / s12.iloc[0]
    s8 = s8 / s8.iloc[0]
    def m(s):
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        ann = s.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else np.nan
        mdd = ((s - s.cummax()) / s.cummax()).min()
        ret = s.pct_change().dropna()
        shp = ret.mean() / (ret.std(ddof=1)+1e-12) * np.sqrt(252) if len(ret) > 1 else np.nan
        return ann, mdd, shp
    a12, d12, s12_ = m(s12)
    a8, d8, s8_ = m(s8)
    print(f"{ys}{'-'+str(ye) if ye<2026 else '+':<6} {a12:>11.1%} {d12:>11.1%} {s12_:>11.2f} | {a8:>11.1%} {d8:>11.1%} {s8_:>11.2f}")

# ============================================================
# 5. ROE12 在 2024 单年的表现（验证"普通年份也不差"）
# ============================================================
print(f"\n{'=' * 100}")
print("【4. ROE12 vs ROE8 vs ROE10 在 2024（非超级牛市）的表现】")
print("=" * 100)
roe10_label = "K3_ROE10_PEG2.5_CHIP40_MV50_YR3"
nv_10, _ = cfg_navs[roe10_label]

for label, nv in [("ROE12", nv_12), ("ROE10", nv_10), ("ROE8", nv_8)]:
    s = nv[(nv.index >= pd.Timestamp("2024-01-01")) & (nv.index < pd.Timestamp("2025-01-01"))]
    s = s / s.iloc[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    ann = s.iloc[-1] ** (1/yrs) - 1
    mdd = ((s - s.cummax()) / s.cummax()).min()
    ret = s.pct_change().dropna()
    shp = ret.mean() / (ret.std(ddof=1)+1e-12) * np.sqrt(252)
    print(f"  {label}: 2024年化{ann:.1%} 回撤{mdd:.1%} 夏普{shp:.2f} 累计{s.iloc[-1]-1:.1%}")
