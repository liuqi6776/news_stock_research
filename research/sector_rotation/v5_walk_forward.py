# -*- coding: utf-8 -*-
"""
v5 Walk-Forward（滚动前向选参）— 真正的 OOS 流程
  Roll1: 2020-01~2023-12 (训练, 48月) → 按训练Sharpe选最优配置 → 2024 执行
  Roll2: 2020-01~2024-12 (训练, 60月) → 按训练Sharpe选最优配置 → 2025 执行
  拼接 2024+2025 得到 Walk-Forward 组合（参数永不看当年）
对比：静态ROE8（训练Best）、静态ROE12（稳健分选参，含2025信息）、纯ETF、全行业等权
"""
import os, sys, time, itertools
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
SR = os.path.join(ROOT, "research", "sector_rotation")
OUT = os.path.join(SR, "results")
IDX_DIR = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily")
GRID_CACHE = os.path.join(OUT, "_v5_wf_grid_cache.pkl")

# ============================================================
# 1. 加载 v5 引擎（前视修复版）
# ============================================================
v5_path = os.path.join(SR, 'backtest_stock_picking_v5.py')
v5_code = open(v5_path, encoding='utf-8').read()
marker = '# ============================================================\n# 8. v5'
code_before = v5_code.split(marker)[0]
ns = {}
exec(code_before, ns)
run_strategy_v5 = ns["run_strategy_v5"]
print(f"[引擎加载] 完成, 耗时 {time.time()-t0:.0f}s")

# ============================================================
# 2. 网格参数空间（和 v5_grid_search.py 完全一致）
# ============================================================
GRID = {
    'global_top_k':           [3, 5, 8],
    'min_roe_pct':            [8, 10, 12, 15],
    'max_peg':                [1.5, 2.0, 2.5],
    'chip_conc_pctl_threshold': [0.40, 0.50, 0.60],
    'min_circ_mv_yi':         [50, 100],
    'min_list_years':         [2, 3],
}
keys = list(GRID.keys())
vals = list(GRID.values())
total = 1
for v in vals: total *= len(v)

# Roll 定义
ROLLS = [
    # (roll名, 训练结束日期(不含), 执行起止)
    ("R1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
    ("R2", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
]
TRAIN_START = pd.Timestamp("2020-01-01")

def cfg_label(p):
    return (f"K{p['global_top_k']}_ROE{p['min_roe_pct']}"
            f"_PEG{p['max_peg']}_CHIP{int(p['chip_conc_pctl_threshold']*100)}"
            f"_MV{p['min_circ_mv_yi']}_YR{p['min_list_years']}")

def period_metrics(nav_s, start, end):
    mask = (nav_s.index >= start) & (nav_s.index < end)
    sub = nav_s[mask].copy()
    if len(sub) < 10:
        return {"年化": np.nan, "回撤": np.nan, "夏普": np.nan, "累计": np.nan, "期末": np.nan}
    sub = sub / sub.iloc[0]
    yrs = (sub.index[-1] - sub.index[0]).days / 365.25
    cum = sub.iloc[-1] - 1
    ann = (1 + cum) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    mdd = ((sub - sub.cummax()) / sub.cummax()).min()
    ret = sub.pct_change().dropna()
    shp = ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(252) if len(ret) > 1 else np.nan
    return {"年化": ann, "回撤": mdd, "夏普": shp, "累计": cum, "期末": sub.iloc[-1]}

# ============================================================
# 3. 跑 432 组全区间回测，保存 nav + 训练期 Sharpe
# ============================================================
print(f"\n[网格回测] {total} 组配置, 两个滚动训练窗口")
print("=" * 80)

import pickle as _pkl
if os.path.exists(GRID_CACHE):
    print(f"  命中缓存: {GRID_CACHE}, 跳过网格回测")
    with open(GRID_CACHE, "rb") as _f:
        cfg_rows, _cfg_navs_raw = _pkl.load(_f)
    cfg_navs = _cfg_navs_raw
else:
    cfg_rows = []  # 存每组训练期Sharpe
    cfg_navs = {}  # {label: (nav_series, p_dict)}
    for i, combo in enumerate(itertools.product(*vals)):
        p = dict(zip(keys, combo))
        label = cfg_label(p)
        try:
            nv, trs, _ = run_strategy_v5(
                global_top_k=p['global_top_k'],
                max_same_sector=2,
                max_pe=60,
                min_turnover_pct=0.5,
                min_circ_mv_yi=p['min_circ_mv_yi'],
                max_circ_mv_yi=2000,
                max_peg=p['max_peg'],
                peg_preferred=1.5,
                min_roe_pct=p['min_roe_pct'],
                chip_conc_pctl_threshold=p['chip_conc_pctl_threshold'],
                min_list_years=p['min_list_years'],
                preferred_weight=1.2,
                verbose=False,
            )
        except Exception as e:
            continue
        if len(nv) < 50:
            continue

        cfg_navs[label] = (nv, p)

        # 两个训练窗口的 Sharpe
        m_r1 = period_metrics(nv, TRAIN_START, ROLLS[0][1])
        m_r2 = period_metrics(nv, TRAIN_START, ROLLS[1][1])

        cfg_rows.append({
            "配置": label,
            **{f"R1train_{k}": v for k, v in m_r1.items()},
            **{f"R2train_{k}": v for k, v in m_r2.items()},
        })

        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{total}, 已耗时 {time.time()-t0:.0f}s")

    with open(GRID_CACHE, "wb") as _f:
        _pkl.dump((cfg_rows, cfg_navs), _f)
    print(f"  缓存已保存: {GRID_CACHE}")

df_cfg = pd.DataFrame(cfg_rows)
print(f"\n[完成] {len(df_cfg)} 组有效, 耗时 {time.time()-t0:.0f}s")

# ============================================================
# 4. 每个滚动窗口选训练期 Sharpe 最高的配置（严格 OOS 选参）
# ============================================================
print(f"\n{'='*80}")
print("【Walk-Forward 选参（只看训练 Sharpe）】")
print(f"{'='*80}\n")

wf_selected = []  # [(name, cfg_label, train_sharpe, exec_start, exec_end, params)]
for roll_name, train_end, exec_start, exec_end in ROLLS:
    col = f"{roll_name}train_夏普"
    df_sorted = df_cfg.sort_values(col, ascending=False).reset_index(drop=True)
    best = df_sorted.iloc[0]
    best_label = best["配置"]
    best_sharpe = best[col]
    best_ann = best[f"{roll_name}train_年化"]
    _, params = cfg_navs[best_label]
    print(f"{roll_name} (训练~{train_end.date()}): {best_label}")
    print(f"   训练: 年化{best_ann:.1%} 夏普{best_sharpe:.2f}")
    wf_selected.append((roll_name, best_label, best_sharpe, exec_start, exec_end, params))

# ============================================================
# 5. 拼接 Walk-Forward 组合净值
# ============================================================
wf_nav_parts = []
for roll_name, cfg_label, ts, exec_start, exec_end, params in wf_selected:
    nv_full, _ = cfg_navs[cfg_label]
    mask = (nv_full.index >= exec_start) & (nv_full.index < exec_end)
    sub = nv_full[mask]
    if len(sub) == 0:
        continue
    sub_rb = sub / sub.iloc[0]
    # 衔接: 前一段期末值作为下一段期初乘数
    if len(wf_nav_parts):
        prev_end = wf_nav_parts[-1].iloc[-1]
        sub_rb = sub_rb * prev_end
    wf_nav_parts.append(sub_rb)
    # 执行期指标
    m_ex = period_metrics(nv_full, exec_start, exec_end)
    print(f"   执行({exec_start.date()}~{exec_end.date()}): "
          f"年化{m_ex['年化']:.1%} 回撤{m_ex['回撤']:.1%} 夏普{m_ex['夏普']:.2f} "
          f"累计{m_ex['累计']:.1%}")

wf_nav = pd.concat(wf_nav_parts).sort_index()
wf_nav.name = "Walk-Forward"

# ============================================================
# 6. 对比基准
# ============================================================
print(f"\n{'='*80}")
print("【对比基准：静态配置 + ETF + 全行业等权】")
print(f"{'='*80}\n")

# 静态配置
STATIC = {
    "静态ROE8(训练Best)": dict(global_top_k=3, min_roe_pct=8, max_peg=2.5,
                                chip_conc_pctl_threshold=0.40, min_circ_mv_yi=50,
                                min_list_years=3),
    "静态ROE12(稳健分选)": dict(global_top_k=3, min_roe_pct=12, max_peg=2.0,
                                  chip_conc_pctl_threshold=0.50, min_circ_mv_yi=50,
                                  min_list_years=3),
}
for name, p in STATIC.items():
    nv, trs, _ = run_strategy_v5(
        global_top_k=p['global_top_k'], max_same_sector=2, max_pe=60,
        min_turnover_pct=0.5, min_circ_mv_yi=p['min_circ_mv_yi'],
        max_circ_mv_yi=2000, max_peg=p['max_peg'], peg_preferred=1.5,
        min_roe_pct=p['min_roe_pct'],
        chip_conc_pctl_threshold=p['chip_conc_pctl_threshold'],
        min_list_years=p['min_list_years'], preferred_weight=1.2, verbose=False,
    )
    cfg_navs[name] = (nv, p)

# ETF & 行业等权
def etf_daily_nav(code):
    pq = os.path.join(IDX_DIR, f"{code}.parquet")
    edf = pd.read_parquet(pq)
    edf["trade_date"] = edf["trade_date"].astype(str)
    edf = edf.sort_values("trade_date").reset_index(drop=True)
    edf["dt"] = pd.to_datetime(edf["trade_date"], format="%Y%m%d")
    edf = edf[(edf["dt"] >= TRAIN_START) & (edf["dt"] < pd.Timestamp("2026-01-01"))].set_index("dt")
    nav = (1 + edf["pct_chg"].fillna(0) / 100.0).cumprod()
    return nav
etf_1000 = etf_daily_nav("512100.SH")
etf_300 = etf_daily_nav("510300.SH")

ir_path = os.path.join(OUT, "industry_ret.csv")
ir_df = pd.read_csv(ir_path, index_col=0)
ir_df.index = pd.to_datetime(ir_df.index)
# 日频等权（从月收益用resample展开到日，月初不变）
ew_ret = ir_df.mean(axis=1, skipna=True)
ew_m = (1 + ew_ret).cumprod()
ew_m = ew_m.resample("M").last()
# 月频转日频（前值填充）
ew_daily = ew_m.reindex(pd.date_range(TRAIN_START, pd.Timestamp("2025-12-31"), freq='B')).ffill()

# ============================================================
# 7. 指标汇总（全期/执行期分拆）
# ============================================================
comb = pd.DataFrame()
comb["Walk-Forward"] = wf_nav
for name in STATIC.keys():
    nv, _ = cfg_navs[name]
    comb[name] = nv[~nv.index.duplicated(keep='first')]
comb["中证1000ETF"] = etf_1000
comb["沪深300ETF"] = etf_300
comb = comb.reindex(pd.date_range(TRAIN_START, pd.Timestamp("2025-12-31"), freq='B')).ffill().dropna(how='all')
# WF从2024-01开始, 之前用NaN

# 全期 2020-01 ~ 2025-12 (对WF 执行期只取 2024-2025; 其它全区间)
print(f"\n{'='*100}")
print("【执行期汇总 (2024-01 ~ 2025-12, 2年 = WF真实OOS)】")
print(f"{'='*100}\n")
print(f"{'策略':<20} {'期末':>8} {'年化':>8} {'回撤':>8} {'夏普':>8}")
print("-" * 56)
for col in comb.columns:
    s = comb[col].dropna()
    s = s[s.index >= pd.Timestamp("2024-01-01")]
    if len(s) < 20:
        continue
    s_rb = s / s.iloc[0]
    yrs = (s_rb.index[-1] - s_rb.index[0]).days / 365.25
    ann = s_rb.iloc[-1] ** (1/yrs) - 1
    mdd = ((s_rb - s_rb.cummax()) / s_rb.cummax()).min()
    ret = s_rb.pct_change().dropna()
    shp = ret.mean() / (ret.std(ddof=1)+1e-12) * np.sqrt(252)
    print(f"{col:<20} {s_rb.iloc[-1]:>8.3f} {ann:>7.1%} {mdd:>7.1%} {shp:>8.2f}")

print(f"\n{'='*100}")
print("【2024 单独一年】")
print(f"{'='*100}\n")
print(f"{'策略':<20} {'期末':>8} {'年化':>8} {'回撤':>8} {'夏普':>8}")
print("-" * 56)
for col in comb.columns:
    s = comb[col].dropna()
    s = s[(s.index >= pd.Timestamp("2024-01-01")) & (s.index < pd.Timestamp("2025-01-01"))]
    if len(s) < 10:
        continue
    s_rb = s / s.iloc[0]
    yrs = (s_rb.index[-1] - s_rb.index[0]).days / 365.25
    ann = s_rb.iloc[-1] ** (1/yrs) - 1
    mdd = ((s_rb - s_rb.cummax()) / s_rb.cummax()).min()
    ret = s_rb.pct_change().dropna()
    shp = ret.mean() / (ret.std(ddof=1)+1e-12) * np.sqrt(252)
    print(f"{col:<20} {s_rb.iloc[-1]:>8.3f} {ann:>7.1%} {mdd:>7.1%} {shp:>8.2f}")

print(f"\n{'='*100}")
print("【2025 单独一年】")
print(f"{'='*100}\n")
print(f"{'策略':<20} {'期末':>8} {'年化':>8} {'回撤':>8} {'夏普':>8}")
print("-" * 56)
for col in comb.columns:
    s = comb[col].dropna()
    s = s[(s.index >= pd.Timestamp("2025-01-01")) & (s.index < pd.Timestamp("2026-01-01"))]
    if len(s) < 10:
        continue
    s_rb = s / s.iloc[0]
    yrs = (s_rb.index[-1] - s_rb.index[0]).days / 365.25
    ann = s_rb.iloc[-1] ** (1/yrs) - 1
    mdd = ((s_rb - s_rb.cummax()) / s_rb.cummax()).min()
    ret = s_rb.pct_change().dropna()
    shp = ret.mean() / (ret.std(ddof=1)+1e-12) * np.sqrt(252)
    print(f"{col:<20} {s_rb.iloc[-1]:>8.3f} {ann:>7.1%} {mdd:>7.1%} {shp:>8.2f}")

# 存月度对比 + WF 选参表
comb_m = comb.resample("M").last()
comb_m.to_csv(os.path.join(OUT, "v5_wf_vs_benchmarks_monthly.csv"))
df_cfg.to_csv(os.path.join(OUT, "v5_wf_grid_train_metrics.csv"), index=False, encoding='utf-8-sig')

# 存选参记录
sel_rows = []
for roll_name, cfg_label, ts, exec_start, exec_end, params in wf_selected:
    sel_rows.append({"Roll": roll_name, "训练结束": (exec_start - pd.Timedelta(days=1)).date(),
                     "执行期": f"{exec_start.date()}~{(exec_end-pd.Timedelta(days=1)).date()}",
                     "所选配置": cfg_label, "训练夏普": ts, **{k: v for k, v in params.items()}})
pd.DataFrame(sel_rows).to_csv(os.path.join(OUT, "v5_wf_selection.csv"), index=False, encoding='utf-8-sig')

# ============================================================
# 8. PNG 对比曲线图
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False

STYLE = {
    "Walk-Forward":       dict(color="#c0392b", lw=2.4, ls="-"),
    "静态ROE8(训练Best)":  dict(color="#8e44ad", lw=1.8, ls="-"),
    "静态ROE12(稳健分选)": dict(color="#f39c12", lw=1.8, ls="-."),
    "中证1000ETF":        dict(color="#7f8c8d", lw=1.4, ls="--"),
    "沪深300ETF":         dict(color="#95a5a6", lw=1.4, ls="-."),
}
# 执行期 2024-2025 画图（WF真实区间）
start = pd.Timestamp("2024-01-01")
fig, ax = plt.subplots(figsize=(14, 7))
for c in comb.columns:
    s = comb[c].dropna()
    s = s[s.index >= start]
    if len(s) < 20:
        continue
    s = s / s.iloc[0]
    ax.plot(s.index, s.values, lw=STYLE[c]["lw"], ls=STYLE[c]["ls"],
            color=STYLE[c]["color"], label=c)
ax.axhline(1.0, color="gray", lw=0.6, ls=":")
ax.axvline(pd.Timestamp("2025-01-02"), color="gray", lw=0.6, ls=":", label="R1→R2切换")
ax.set_title("Walk-Forward vs 静态配置 vs ETF (2024-01 ~ 2025-12, WF真实OOS执行期)", fontsize=13)
ax.set_ylabel("净值(基准=1)")
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v5_wf_vs_benchmarks.png"), dpi=150)
plt.close(fig)

print(f"\n[总耗时] {time.time()-t0:.0f}s, 产物: v5_wf_vs_benchmarks_monthly.csv + v5_wf_vs_benchmarks.png")
