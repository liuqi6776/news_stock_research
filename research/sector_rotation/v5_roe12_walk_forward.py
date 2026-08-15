# -*- coding: utf-8 -*-
"""
v5 ROE=12 固定 + 其余参数年度 Walk-Forward
  - ROE 固定 12%，其他参数（K/PEG/CHIP/MV/YR）每年初用训练期Sharpe选最优
  - 复用 _v5_wf_grid_cache.pkl 中的 ROE12 配置（无需重跑432组）
  - 6个年度滚动窗口: 2020~2025 各年
  - 对比: 静态ROE12全期最优、静态ROE8全期最优、中证1000ETF、沪深300ETF、全行业等权
"""
import os, pickle as _pkl
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
SR = os.path.join(ROOT, "research", "sector_rotation")
OUT = os.path.join(SR, "results")
IDX_DIR = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily")

# ============================================================
# 1. 加载432组缓存，筛选 ROE12 配置
# ============================================================
cache_path = os.path.join(OUT, "_v5_wf_grid_cache.pkl")
with open(cache_path, "rb") as f:
    cfg_rows, cfg_navs = _pkl.load(f)

# 筛选 ROE12
roe12_navs = {k: v for k, v in cfg_navs.items() if "_ROE12_" in k}
print(f"[加载] 缓存中 ROE12 配置: {len(roe12_navs)} 组")
for k in sorted(roe12_navs.keys()):
    print(f"  {k}")

TRAIN_START = pd.Timestamp("2020-01-01")

# ============================================================
# 2. 年度 Walk-Forward 窗口
# ============================================================
WF_YEARS = [
    # (年份, 执行起, 执行止, 训练止)
    ("2020", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"), pd.Timestamp("2019-12-31")),
    ("2021", pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01"), pd.Timestamp("2020-12-31")),
    ("2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"), pd.Timestamp("2021-12-31")),
    ("2023", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2022-12-31")),
    ("2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2023-12-31")),
    ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2024-12-31")),
]

def period_metrics(nav_s, start, end):
    mask = (nav_s.index >= start) & (nav_s.index < end)
    sub = nav_s[mask]
    if len(sub) < 10:
        return {"年化": np.nan, "回撤": np.nan, "夏普": np.nan, "累计": np.nan}
    sub = sub / sub.iloc[0]
    yrs = (sub.index[-1] - sub.index[0]).days / 365.25
    cum = sub.iloc[-1] - 1
    ann = (1 + cum) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    mdd = ((sub - sub.cummax()) / sub.cummax()).min()
    ret = sub.pct_change().dropna()
    shp = ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(252) if len(ret) > 1 else np.nan
    return {"年化": ann, "回撤": mdd, "夏普": shp, "累计": cum}

# ============================================================
# 3. 每年选训练期 Sharpe 最高的 ROE12 配置
# ============================================================
print(f"\n{'='*80}")
print("【ROE12 固定: 年度 Walk-Forward 选参】")
print(f"{'='*80}\n")

wf_selections = []
for yr, exec_start, exec_end, train_end in WF_YEARS:
    best_label = None
    best_sharpe = -999
    best_m = None
    for label, (nv, p) in roe12_navs.items():
        m = period_metrics(nv, TRAIN_START, train_end)
        if np.isnan(m["夏普"]):
            continue
        if m["夏普"] > best_sharpe:
            best_sharpe = m["夏普"]
            best_label = label
            best_m = m

    if best_label is None:
        print(f"  {yr}: 无有效配置")
        continue

    _, best_p = roe12_navs[best_label]
    wf_selections.append({
        "year": yr,
        "train_end": train_end,
        "exec_start": exec_start,
        "exec_end": exec_end,
        "label": best_label,
        "train_sharpe": best_sharpe,
        "train_ann": best_m["年化"],
        "train_mdd": best_m["回撤"],
        "params": best_p,
    })
    print(f"  {yr} (训练~{train_end.date()}): {best_label}")
    print(f"    训练: 年化{best_m['年化']:.1%} 夏普{best_sharpe:.2f} 回撤{best_m['回撤']:.1%}")

# ============================================================
# 4. 拼接 Walk-Forward 净值
# ============================================================
print(f"\n{'='*80}")
print("【WF 净值拼接 + 逐年执行表现】")
print(f"{'='*80}\n")

wf_parts = []
for sel in wf_selections:
    nv_full, _ = roe12_navs[sel["label"]]
    mask = (nv_full.index >= sel["exec_start"]) & (nv_full.index < sel["exec_end"])
    sub = nv_full[mask]
    if len(sub) == 0:
        print(f"  {sel['year']}: 无执行期数据")
        continue
    sub_rb = sub / sub.iloc[0]
    if wf_parts:
        sub_rb = sub_rb * wf_parts[-1].iloc[-1]
    wf_parts.append(sub_rb)
    m_ex = period_metrics(nv_full, sel["exec_start"], sel["exec_end"])
    print(f"  {sel['year']} ({sel['exec_start'].date()}~{sel['exec_end'].date()}): "
          f"年化{m_ex['年化']:.1%} 回撤{m_ex['回撤']:.1%} 夏普{m_ex['夏普']:.2f} 累计{m_ex['累计']:.1%}")

wf_nav = pd.concat(wf_parts).sort_index()
wf_nav.name = "WF-ROE12"
print(f"\n  WF 全期: 期末{wf_nav.iloc[-1]:.4f} "
      f"年化{wf_nav.iloc[-1]**(1/((wf_nav.index[-1]-wf_nav.index[0]).days/365.25))-1:.1%} "
      f"回撤{((wf_nav-wf_nav.cummax())/wf_nav.cummax()).min():.1%}")

# ============================================================
# 5. 对比基准
# ============================================================
# 静态 ROE12 全期最优
best_static_label = None
best_static_sharpe = -999
for label, (nv, p) in roe12_navs.items():
    m = period_metrics(nv, TRAIN_START, pd.Timestamp("2026-01-01"))
    if np.isnan(m["夏普"]):
        continue
    if m["夏普"] > best_static_sharpe:
        best_static_sharpe = m["夏普"]
        best_static_label = label
        best_static_m = m
print(f"\n  静态ROE12全期最优: {best_static_label} 年化{best_static_m['年化']:.1%} 夏普{best_static_m['夏普']:.2f}")
nv_static_roe12, _ = roe12_navs[best_static_label]

# 静态 ROE8 全期最优（从全缓存中找）
roe8_navs = {k: v for k, v in cfg_navs.items() if "_ROE8_" in k}
best_r8_label = None
best_r8_sharpe = -999
for label, (nv, p) in roe8_navs.items():
    m = period_metrics(nv, TRAIN_START, pd.Timestamp("2026-01-01"))
    if np.isnan(m["夏普"]):
        continue
    if m["夏普"] > best_r8_sharpe:
        best_r8_sharpe = m["夏普"]
        best_r8_label = label
        best_r8_m = m
print(f"  静态ROE8全期最优: {best_r8_label} 年化{best_r8_m['年化']:.1%} 夏普{best_r8_m['夏普']:.2f}")
nv_static_roe8, _ = roe8_navs[best_r8_label]

# ETF 基准
def etf_daily_nav(code):
    pq = os.path.join(IDX_DIR, f"{code}.parquet")
    edf = pd.read_parquet(pq)
    edf["trade_date"] = edf["trade_date"].astype(str)
    edf = edf.sort_values("trade_date").reset_index(drop=True)
    edf["dt"] = pd.to_datetime(edf["trade_date"], format="%Y%m%d")
    edf = edf[(edf["dt"] >= TRAIN_START) & (edf["dt"] < pd.Timestamp("2026-01-01"))].set_index("dt")
    return (1 + edf["pct_chg"].fillna(0) / 100.0).cumprod()

etf_1000 = etf_daily_nav("512100.SH")
etf_300 = etf_daily_nav("510300.SH")

# 行业等权
ir_df = pd.read_csv(os.path.join(OUT, "industry_ret.csv"), index_col=0)
ir_df.index = pd.to_datetime(ir_df.index)
ew_ret = ir_df.mean(axis=1, skipna=True)
ew_m = (1 + ew_ret).cumprod()
ew_daily = ew_m.resample("M").last().reindex(
    pd.date_range(TRAIN_START, pd.Timestamp("2025-12-31"), freq='B')).ffill()

# ============================================================
# 6. 合并 + 指标
# ============================================================
comb = pd.DataFrame({
    "WF-ROE12(滚动选参)": wf_nav,
    "静态ROE12全期最优": nv_static_roe12,
    "静态ROE8全期最优": nv_static_roe8,
    "中证1000ETF": etf_1000,
    "沪深300ETF": etf_300,
    "行业等权": ew_daily,
})
comb = comb.reindex(pd.date_range(TRAIN_START, pd.Timestamp("2025-12-31"), freq='B')).ffill()

def stats_row(s, start, end):
    s = s[(s.index >= start) & (s.index < end)].dropna()
    if len(s) < 10:
        return None
    s = s / s.iloc[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    ann = s.iloc[-1] ** (1/yrs) - 1
    mdd = ((s - s.cummax()) / s.cummax()).min()
    ret = s.pct_change().dropna()
    shp = ret.mean() / (ret.std(ddof=1)+1e-12) * np.sqrt(252)
    return {"期末": s.iloc[-1], "年化": ann, "回撤": mdd, "夏普": shp, "累计": s.iloc[-1]-1}

def print_table(title, start, end):
    print(f"\n{'='*100}")
    print(f"【{title}】({start.date()} ~ {end.date()})")
    print(f"{'='*100}")
    print(f"{'策略':<24} {'期末':>8} {'年化':>8} {'回撤':>8} {'夏普':>8} {'累计':>8}")
    print("-" * 70)
    for col in comb.columns:
        r = stats_row(comb[col], start, end)
        if r is None:
            continue
        print(f"{col:<24} {r['期末']:>8.3f} {r['年化']:>7.1%} {r['回撤']:>7.1%} {r['夏普']:>8.2f} {r['累计']:>7.1%}")

print_table("全周期 2020-2025", TRAIN_START, pd.Timestamp("2026-01-01"))
print_table("训练期 2020-2024", TRAIN_START, pd.Timestamp("2025-01-01"))
print_table("样本外 2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"))

# 逐年明细
print(f"\n{'='*100}")
print("【逐年明细】")
print(f"{'='*100}")
print(f"{'年份':<6} {'策略':<24} {'年化':>8} {'回撤':>8} {'夏普':>8}")
print("-" * 60)
for yr, exec_start, exec_end, _ in WF_YEARS:
    for col in comb.columns:
        s = comb[col][(comb[col].index >= exec_start) & (comb[col].index < exec_end)].dropna()
        if len(s) < 5:
            continue
        s = s / s.iloc[0]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        ann = s.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else np.nan
        mdd = ((s - s.cummax()) / s.cummax()).min()
        ret = s.pct_change().dropna()
        shp = ret.mean() / (ret.std(ddof=1)+1e-12) * np.sqrt(252) if len(ret) > 1 else np.nan
        print(f"{yr:<6} {col:<24} {ann:>7.1%} {mdd:>7.1%} {shp:>8.2f}")
    print("-" * 60)

# ============================================================
# 7. 保存
# ============================================================
comb.resample("M").last().to_csv(os.path.join(OUT, "v5_roe12_wf_vs_benchmarks_monthly.csv"),
                                  encoding='utf-8-sig')

sel_rows = []
for sel in wf_selections:
    p = sel["params"]
    sel_rows.append({
        "年份": sel["year"], "训练截止": sel["train_end"].date(),
        "所选配置": sel["label"],
        "训练年化": f"{sel['train_ann']:.1%}", "训练夏普": f"{sel['train_sharpe']:.2f}",
        "K": p["global_top_k"], "PEG": p["max_peg"],
        "CHIP": int(p["chip_conc_pctl_threshold"]*100),
        "MV(亿)": p["min_circ_mv_yi"], "上市年限": p["min_list_years"],
    })
pd.DataFrame(sel_rows).to_csv(os.path.join(OUT, "v5_roe12_wf_selection.csv"),
                               index=False, encoding='utf-8-sig')

# ============================================================
# 8. PNG 对比图
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

STYLE = {
    "WF-ROE12(滚动选参)":  dict(color="#c0392b", lw=2.5, ls="-"),
    "静态ROE12全期最优":   dict(color="#2980b9", lw=1.8, ls="--"),
    "静态ROE8全期最优":    dict(color="#8e44ad", lw=1.8, ls="-."),
    "中证1000ETF":         dict(color="#7f8c8d", lw=1.2, ls=":"),
    "沪深300ETF":          dict(color="#95a5a6", lw=1.2, ls=":"),
    "行业等权":            dict(color="#27ae60", lw=1.2, ls=":"),
}

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
# 全周期
ax = axes[0]
for c in comb.columns:
    s = comb[c].dropna()
    if len(s) < 20:
        continue
    s = s / s.iloc[0]
    st = STYLE.get(c, dict(color="gray", lw=1, ls="-"))
    ax.plot(s.index, s.values, lw=st["lw"], ls=st["ls"], color=st["color"], label=c, alpha=0.9)
ax.axhline(1.0, color="gray", lw=0.5, ls=":")
ax.set_title("ROE12固定: 年度Walk-Forward选参 vs 静态最优 vs ETF (2020-2025)", fontsize=13)
ax.set_ylabel("净值(基准=1)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

# 2024-2025特写
ax2 = axes[1]
for c in comb.columns:
    s = comb[c].dropna()
    s = s[s.index >= pd.Timestamp("2024-01-01")]
    if len(s) < 10:
        continue
    s = s / s.iloc[0]
    st = STYLE.get(c, dict(color="gray", lw=1, ls="-"))
    ax2.plot(s.index, s.values, lw=st["lw"], ls=st["ls"], color=st["color"], label=c, alpha=0.9)
ax2.axhline(1.0, color="gray", lw=0.5, ls=":")
ax2.set_title("2024-2025 样本外特写", fontsize=12)
ax2.set_ylabel("净值(2024-01=1)")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "v5_roe12_wf_vs_benchmarks.png"), dpi=150)
plt.close(fig)

print(f"\n[完成] 产物已保存:")
print(f"  - v5_roe12_wf_vs_benchmarks_monthly.csv")
print(f"  - v5_roe12_wf_selection.csv")
print(f"  - v5_roe12_wf_vs_benchmarks.png")
