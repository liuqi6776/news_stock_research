# -*- coding: utf-8 -*-
"""
分散持仓测试: Top-K = 3/5/8/10/15 对回撤与收益的影响
固定其他参数 (ROE12/PEG1.5/CHIP50/MV50/YR3, 无择时)
目标: 找收益与回撤的最佳平衡点, 降低 -30.4% 回撤
"""
import os, time
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
SR = os.path.join(ROOT, "research", "sector_rotation")
OUT = os.path.join(SR, "results")
IDX_DIR = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily")

v5_path = os.path.join(SR, 'backtest_stock_picking_v5.py')
v5_code = open(v5_path, encoding='utf-8').read()
marker = '# ============================================================\n# 8. v5'
ns = {}
exec(v5_code.split(marker)[0], ns)
run_strategy_v5 = ns["run_strategy_v5"]
print(f"[1] 引擎加载, 耗时 {time.time()-t0:.0f}s")

FIXED = dict(max_same_sector=2, max_pe=60, min_turnover_pct=0.5,
             min_circ_mv_yi=50, max_circ_mv_yi=2000, max_peg=1.5, peg_preferred=1.5,
             min_roe_pct=12, chip_conc_pctl_threshold=0.50,
             min_list_years=3, preferred_weight=1.2)

TOPK_LIST = [3, 5, 8, 10, 15]
results = {}
for k in TOPK_LIST:
    nv, trs, _ = run_strategy_v5(global_top_k=k, use_s123=False, verbose=False, **FIXED)
    results[f"Top{k}"] = nv
    print(f"  Top{k}: {len(nv)}天, 期末{nv.iloc[-1]:.3f}, 换手{len(trs)}笔")

# 基准
TRAIN_START = pd.Timestamp("2020-01-01")
def etf_nav(code):
    pq = os.path.join(IDX_DIR, f"{code}.parquet")
    edf = pd.read_parquet(pq)
    edf["trade_date"] = edf["trade_date"].astype(str)
    edf = edf.sort_values("trade_date").reset_index(drop=True)
    edf["dt"] = pd.to_datetime(edf["trade_date"], format="%Y%m%d")
    edf = edf[(edf["dt"] >= TRAIN_START) & (edf["dt"] < pd.Timestamp("2026-01-01"))].set_index("dt")
    return (1 + edf["pct_chg"].fillna(0) / 100.0).cumprod()
results["中证1000ETF"] = etf_nav("512100.SH")
results["沪深300ETF"] = etf_nav("510300.SH")

ir_df = pd.read_csv(os.path.join(OUT, "industry_ret.csv"), index_col=0)
ir_df.index = pd.to_datetime(ir_df.index)
ew_daily = (1 + ir_df.mean(axis=1, skipna=True)).cumprod().resample("M").last().reindex(
    pd.date_range(TRAIN_START, pd.Timestamp("2025-12-31"), freq='B')).ffill()
results["行业等权"] = ew_daily

comb = pd.DataFrame(results)
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
    calmar = ann / abs(mdd) if mdd != 0 else np.nan
    return {"期末": s.iloc[-1], "年化": ann, "回撤": mdd, "夏普": shp, "卡玛": calmar}

print(f"\n{'='*90}")
print(f"【全周期 2020-2025】(分散持仓 Top-K 扫描)")
print(f"{'='*90}")
print(f"{'策略':<12} {'期末':>7} {'年化':>8} {'回撤':>8} {'夏普':>8} {'卡玛':>8}")
print("-" * 56)
for col in comb.columns:
    r = stats_row(comb[col], TRAIN_START, pd.Timestamp("2026-01-01"))
    if r:
        print(f"{col:<12} {r['期末']:>7.3f} {r['年化']:>7.1%} {r['回撤']:>7.1%} {r['夏普']:>8.2f} {r['卡玛']:>8.2f}")

# 逐年 (关键看熊市年份回撤改善)
print(f"\n{'='*90}")
print(f"【逐年年化】")
print(f"{'='*90}")
years = [("2020","2020-01-01","2021-01-01"),("2021","2021-01-01","2022-01-01"),
         ("2022","2022-01-01","2023-01-01"),("2023","2023-01-01","2024-01-01"),
         ("2024","2024-01-01","2025-01-01"),("2025","2025-01-01","2026-01-01")]
print(f"{'年份':<6}", end="")
for col in comb.columns:
    print(f" {col:<11}", end="")
print()
print("-" * 110)
for yr, s0, s1 in years:
    print(f"{yr:<6}", end="")
    for col in comb.columns:
        s = comb[col][(comb[col].index >= pd.Timestamp(s0)) & (comb[col].index < pd.Timestamp(s1))].dropna()
        if len(s) < 5:
            print(f" {'n/a':<11}", end="")
            continue
        s = s / s.iloc[0]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        ann = s.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else np.nan
        print(f" {ann:>8.1%} ", end="")
    print()

comb.resample("M").last().to_csv(os.path.join(OUT, "v5_topk_scan_monthly.csv"), encoding='utf-8-sig')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
ax = axes[0]
for c in comb.columns:
    s = comb[c].dropna()
    if len(s) == 0:
        continue
    s = s / s.iloc[0]
    ax.plot(s.index, s.values, lw=1.6, label=c, alpha=0.9)
ax.axhline(1.0, color="gray", lw=0.5, ls=":")
ax.set_title("分散持仓 Top-K 扫描 (ROE12/PEG1.5/CHIP50/MV50/YR3, 无择时)", fontsize=13)
ax.set_ylabel("净值(基准=1)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

# 回撤 vs 收益散点 (卡玛分析)
ax2 = axes[1]
xs, ys, labels = [], [], []
for col in comb.columns:
    if col in ("中证1000ETF", "沪深300ETF", "行业等权"):
        continue
    r = stats_row(comb[col], TRAIN_START, pd.Timestamp("2026-01-01"))
    xs.append(abs(r["回撤"]) * 100)
    ys.append(r["年化"] * 100)
    labels.append(col)
ax2.scatter(xs, ys, s=80, c="#c0392b", zorder=3)
for i, lb in enumerate(labels):
    ax2.annotate(lb, (xs[i], ys[i]), textcoords="offset points", xytext=(8, 4), fontsize=9)
ax2.set_xlabel("最大回撤 (%)")
ax2.set_ylabel("年化收益 (%)")
ax2.set_title("回撤-收益权衡 (越靠左上越好)", fontsize=12)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "v5_topk_scan.png"), dpi=150)
plt.close(fig)

print(f"\n[完成] 耗时 {time.time()-t0:.0f}s")
print(f"  - v5_topk_scan_monthly.csv")
print(f"  - v5_topk_scan.png")
