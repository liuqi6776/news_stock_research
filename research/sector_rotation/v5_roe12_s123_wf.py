# -*- coding: utf-8 -*-
"""
v5 ROE=12 固定 + s123 大盘择时 + 年度 Walk-Forward（回撤约束选参）
P0 改进: (1) 叠加 s123 择时控制熊市回撤 (2) 选参加回撤约束避免激进参数

对比 4 个版本, 隔离每个改进的贡献:
  A. WF-ROE12 无s123 + 纯夏普选参     (基线, 年化7.1%/-50.5%)
  B. WF-ROE12 + s123 + 纯夏普选参     (加择时)
  C. WF-ROE12 + s123 + 回撤约束选参    (加择时 + 回撤约束)  ← 目标
  D. 静态基准 (ROE12全期最优/ROE8全期最优) + ETF + 行业等权
"""
import os, time, itertools, pickle as _pkl
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
SR = os.path.join(ROOT, "research", "sector_rotation")
OUT = os.path.join(SR, "results")
IDX_DIR = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily")

# ============================================================
# 1. 加载 v5 引擎 + s123 信号
# ============================================================
v5_path = os.path.join(SR, 'backtest_stock_picking_v5.py')
v5_code = open(v5_path, encoding='utf-8').read()
marker = '# ============================================================\n# 8. v5'
ns = {}
exec(v5_code.split(marker)[0], ns)
run_strategy_v5 = ns["run_strategy_v5"]
build_s123_v8 = ns["build_s123_v8"]
print(f"[1] 引擎加载, 耗时 {time.time()-t0:.0f}s")

sig_map, v8_daily = build_s123_v8()
print(f"[2] s123 信号: {len(sig_map)} 个月, V8 避险日收益 {len(v8_daily)} 天")

# ============================================================
# 2. 参数网格 (ROE 固定 12)
# ============================================================
FIXED_ROE = 12
GRID = {
    'global_top_k': [3, 5, 8],
    'max_peg': [1.5, 2.0, 2.5],
    'chip_conc_pctl_threshold': [0.40, 0.50, 0.60],
    'min_circ_mv_yi': [50, 100],
    'min_list_years': [2, 3],
}
keys = list(GRID.keys())
vals = list(GRID.values())
total = 1
for v in vals: total *= len(v)
print(f"[3] 网格 {total} 组 (ROE={FIXED_ROE} 固定)")

TRAIN_START = pd.Timestamp("2020-01-01")
MAX_DD_CONSTRAINT = -0.30  # 训练期回撤不得超过 30%

def cfg_label(p):
    return (f"K{p['global_top_k']}_ROE{FIXED_ROE}_PEG{p['max_peg']}"
            f"_CHIP{int(p['chip_conc_pctl_threshold']*100)}"
            f"_MV{p['min_circ_mv_yi']}_YR{p['min_list_years']}")

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
# 3. 跑 108 组 (use_s123=True), 带缓存
# ============================================================
CACHE = os.path.join(OUT, "_v5_roe12_s123_grid_cache.pkl")
if os.path.exists(CACHE):
    print(f"[4] 命中缓存 {CACHE}, 跳过网格回测")
    with open(CACHE, "rb") as f:
        cfg_navs = _pkl.load(f)
else:
    print(f"[4] 跑网格 {total} 组 (use_s123=True), 首次执行...")
    cfg_navs = {}
    for i, combo in enumerate(itertools.product(*vals)):
        p = dict(zip(keys, combo))
        label = cfg_label(p)
        try:
            nv, trs, _ = run_strategy_v5(
                global_top_k=p['global_top_k'], max_same_sector=2, max_pe=60,
                min_turnover_pct=0.5, min_circ_mv_yi=p['min_circ_mv_yi'],
                max_circ_mv_yi=2000, max_peg=p['max_peg'], peg_preferred=1.5,
                min_roe_pct=FIXED_ROE,
                chip_conc_pctl_threshold=p['chip_conc_pctl_threshold'],
                min_list_years=p['min_list_years'], preferred_weight=1.2,
                use_s123=True, sig_map=sig_map, v8_daily=v8_daily, verbose=False)
        except Exception as e:
            print(f"    {label}: ERROR {e}")
            continue
        if len(nv) < 50:
            continue
        cfg_navs[label] = (nv, p)
        if (i + 1) % 24 == 0:
            print(f"    {i+1}/{total}, 耗时 {time.time()-t0:.0f}s")
    with open(CACHE, "wb") as f:
        _pkl.dump(cfg_navs, f)
    print(f"    缓存已保存, 共 {len(cfg_navs)} 组有效")

# ============================================================
# 4. Walk-Forward 选参 (两种准则: 纯夏普 / 回撤约束+夏普)
# ============================================================
WF_YEARS = [
    ("2020", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"), pd.Timestamp("2019-12-31")),
    ("2021", pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01"), pd.Timestamp("2020-12-31")),
    ("2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"), pd.Timestamp("2021-12-31")),
    ("2023", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2022-12-31")),
    ("2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2023-12-31")),
    ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2024-12-31")),
]

def run_wf(use_dd_constraint):
    """年度滚动选参, 返回 (wf_nav, selections)"""
    selections = []
    parts = []
    for yr, exec_start, exec_end, train_end in WF_YEARS:
        best_label, best_sharpe, best_m = None, -999, None
        for label, (nv, p) in cfg_navs.items():
            m = period_metrics(nv, TRAIN_START, train_end)
            if np.isnan(m["夏普"]):
                continue
            if use_dd_constraint and m["回撤"] < MAX_DD_CONSTRAINT:
                continue  # 回撤约束: 训练期回撤超25%直接剔除
            if m["夏普"] > best_sharpe:
                best_sharpe, best_label, best_m = m["夏普"], label, m
        if best_label is None:
            selections.append({"year": yr, "label": None})
            continue
        _, best_p = cfg_navs[best_label]
        selections.append({
            "year": yr, "label": best_label, "train_sharpe": best_sharpe,
            "train_ann": best_m["年化"], "train_mdd": best_m["回撤"],
            "exec_start": exec_start, "exec_end": exec_end,
            "params": best_p,
        })
        nv_full, _ = cfg_navs[best_label]
        mask = (nv_full.index >= exec_start) & (nv_full.index < exec_end)
        sub = nv_full[mask]
        if len(sub) == 0:
            continue
        sub_rb = sub / sub.iloc[0]
        if parts:
            sub_rb = sub_rb * parts[-1].iloc[-1]
        parts.append(sub_rb)
    wf_nav = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    return wf_nav, selections

print(f"\n[5] Walk-Forward 选参")
print("=" * 100)

wf_nav_sharpe, sel_sharpe = run_wf(use_dd_constraint=False)
wf_nav_dd, sel_dd = run_wf(use_dd_constraint=True)

print(f"\n--- B: s123 + 纯夏普选参 ---")
for s in sel_sharpe:
    if s["label"]:
        print(f"  {s['year']}: {s['label']}  训练夏普{s['train_sharpe']:.2f} 回撤{s['train_mdd']:.1%}")
print(f"\n--- C: s123 + 回撤约束(>-25%)选参 ---")
for s in sel_dd:
    if s["label"]:
        print(f"  {s['year']}: {s['label']}  训练夏普{s['train_sharpe']:.2f} 回撤{s['train_mdd']:.1%}")

# ============================================================
# 5. 静态基准 + ETF + 行业等权
# ============================================================
print(f"\n[6] 构建对比基准")
# 静态 ROE12 全期最优 (s123)
best_r12_label, best_r12_sharpe, best_r12_m = None, -999, None
for label, (nv, p) in cfg_navs.items():
    m = period_metrics(nv, TRAIN_START, pd.Timestamp("2026-01-01"))
    if np.isnan(m["夏普"]):
        continue
    if m["夏普"] > best_r12_sharpe:
        best_r12_sharpe, best_r12_label, best_r12_m = m["夏普"], label, m
print(f"  静态ROE12全期最优(s123): {best_r12_label} 年化{best_r12_m['年化']:.1%} 夏普{best_r12_m['夏普']:.2f}")
nv_static_r12, _ = cfg_navs[best_r12_label]

# 静态 ROE8 (无s123) 全期最优 — 从旧缓存读
old_cache = os.path.join(OUT, "_v5_wf_grid_cache.pkl")
with open(old_cache, "rb") as f:
    _, old_navs = _pkl.load(f)
roe8_navs = {k: v for k, v in old_navs.items() if "_ROE8_" in k}
best_r8_label, best_r8_sharpe, best_r8_m = None, -999, None
for label, (nv, p) in roe8_navs.items():
    m = period_metrics(nv, TRAIN_START, pd.Timestamp("2026-01-01"))
    if np.isnan(m["夏普"]):
        continue
    if m["夏普"] > best_r8_sharpe:
        best_r8_sharpe, best_r8_label, best_r8_m = m["夏普"], label, m
print(f"  静态ROE8全期最优(无s123): {best_r8_label} 年化{best_r8_m['年化']:.1%} 夏普{best_r8_m['夏普']:.2f}")
nv_static_r8, _ = roe8_navs[best_r8_label]

# 旧基线 WF (无s123, 纯夏普) — 从旧脚本结果读
wf_old_path = os.path.join(OUT, "v5_roe12_wf_vs_benchmarks_monthly.csv")
wf_old = None
if os.path.exists(wf_old_path):
    _df = pd.read_csv(wf_old_path, index_col=0)
    _df.index = pd.to_datetime(_df.index)
    if "WF-ROE12(滚动选参)" in _df.columns:
        wf_old = _df["WF-ROE12(滚动选参)"].dropna()

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

ir_df = pd.read_csv(os.path.join(OUT, "industry_ret.csv"), index_col=0)
ir_df.index = pd.to_datetime(ir_df.index)
ew_ret = ir_df.mean(axis=1, skipna=True)
ew_daily = (1 + ew_ret).cumprod().resample("M").last().reindex(
    pd.date_range(TRAIN_START, pd.Timestamp("2025-12-31"), freq='B')).ffill()

# ============================================================
# 6. 合并 + 指标
# ============================================================
comb = pd.DataFrame({
    "A.WF无s123+纯夏普": wf_old if wf_old is not None else pd.Series(dtype=float),
    "B.WF+s123+纯夏普": wf_nav_sharpe,
    "C.WF+s123+回撤约束": wf_nav_dd,
    "静态ROE12全期(s123)": nv_static_r12,
    "静态ROE8全期(无s123)": nv_static_r8,
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
    print(f"{'策略':<26} {'期末':>8} {'年化':>8} {'回撤':>8} {'夏普':>8}")
    print("-" * 66)
    for col in comb.columns:
        r = stats_row(comb[col], start, end)
        if r is None:
            continue
        print(f"{col:<26} {r['期末']:>8.3f} {r['年化']:>7.1%} {r['回撤']:>7.1%} {r['夏普']:>8.2f}")

print_table("全周期 2020-2025", TRAIN_START, pd.Timestamp("2026-01-01"))
print_table("训练期 2020-2024", TRAIN_START, pd.Timestamp("2025-01-01"))
print_table("样本外 2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"))

# 逐年明细 (关键年份)
print(f"\n{'='*100}")
print("【逐年明细】")
print(f"{'='*100}")
print(f"{'年份':<6} {'策略':<26} {'年化':>8} {'回撤':>8} {'夏普':>8}")
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
        print(f"{yr:<6} {col:<26} {ann:>7.1%} {mdd:>7.1%} {shp:>8.2f}")
    print("-" * 60)

# ============================================================
# 7. 保存
# ============================================================
comb.resample("M").last().to_csv(os.path.join(OUT, "v5_roe12_s123_wf_vs_benchmarks_monthly.csv"),
                                  encoding='utf-8-sig')
sel_dd_rows = []
for s in sel_dd:
    if s["label"] is None:
        continue
    p = s["params"]
    sel_dd_rows.append({
        "年份": s["year"], "所选配置": s["label"],
        "训练年化": f"{s['train_ann']:.1%}", "训练夏普": f"{s['train_sharpe']:.2f}",
        "训练回撤": f"{s['train_mdd']:.1%}",
        "K": p["global_top_k"], "PEG": p["max_peg"],
        "CHIP": int(p["chip_conc_pctl_threshold"]*100),
        "MV(亿)": p["min_circ_mv_yi"], "上市年限": p["min_list_years"],
    })
pd.DataFrame(sel_dd_rows).to_csv(os.path.join(OUT, "v5_roe12_s123_wf_selection.csv"),
                                 index=False, encoding='utf-8-sig')

# ============================================================
# 8. PNG 图
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

STYLE = {
    "A.WF无s123+纯夏普": dict(color="#e74c3c", lw=1.6, ls=":"),
    "B.WF+s123+纯夏普": dict(color="#2980b9", lw=2.0, ls="--"),
    "C.WF+s123+回撤约束": dict(color="#c0392b", lw=2.6, ls="-"),
    "静态ROE12全期(s123)": dict(color="#27ae60", lw=1.8, ls="-."),
    "静态ROE8全期(无s123)": dict(color="#8e44ad", lw=1.4, ls="-."),
    "中证1000ETF": dict(color="#7f8c8d", lw=1.2, ls=":"),
    "沪深300ETF": dict(color="#95a5a6", lw=1.2, ls=":"),
    "行业等权": dict(color="#f39c12", lw=1.2, ls=":"),
}

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
ax = axes[0]
for c in comb.columns:
    s = comb[c].dropna()
    if len(s) < 20:
        continue
    s = s / s.iloc[0]
    st = STYLE.get(c, dict(color="gray", lw=1, ls="-"))
    ax.plot(s.index, s.values, lw=st["lw"], ls=st["ls"], color=st["color"], label=c, alpha=0.9)
ax.axhline(1.0, color="gray", lw=0.5, ls=":")
ax.set_title("ROE12固定: s123择时 + 回撤约束选参 全周期对比 (2020-2025)", fontsize=13)
ax.set_ylabel("净值(基准=1)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

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
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "v5_roe12_s123_wf_vs_benchmarks.png"), dpi=150)
plt.close(fig)

print(f"\n[完成] 耗时 {time.time()-t0:.0f}s")
print(f"  - v5_roe12_s123_wf_vs_benchmarks_monthly.csv")
print(f"  - v5_roe12_s123_wf_selection.csv")
print(f"  - v5_roe12_s123_wf_vs_benchmarks.png")
