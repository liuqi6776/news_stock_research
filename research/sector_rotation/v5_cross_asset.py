# -*- coding: utf-8 -*-
"""
跨资产配置测试: 股票Top3 + 债券/黄金(V8避险) 的月频再平衡混合
固定股票端参数 (ROE12/K3/PEG1.5/CHIP50/MV50/YR3, 无择时)
目标: 看能否在不牺牲太多收益的情况下把回撤压到 -20% 以内
V8 = 511990(短债) + 511260(信用债) + 518880(黄金) 各1/3

口径统一说明:
- 净值统一 reindex 到工作日(bdays=freq B) 并 ffill, 与 v5_topk_scan.py 完全一致
- 指标统一: 年化=CAGR, 回撤=净值最大回撤, 夏普=日收益 mean/std(ddof=1)*sqrt(252), 卡玛=年化/|回撤|
- 再平衡频率: 月频(每月初重置目标权重, 月内买入持有让权重自然漂移), 而非不现实的日频再平衡
"""
import os, time
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
SR = os.path.join(ROOT, "research", "sector_rotation")
OUT = os.path.join(SR, "results")

v5_path = os.path.join(SR, 'backtest_stock_picking_v5.py')
v5_code = open(v5_path, encoding='utf-8').read()
marker = '# ============================================================\n# 8. v5'
ns = {}
exec(v5_code.split(marker)[0], ns)
run_strategy_v5 = ns["run_strategy_v5"]
build_s123_v8 = ns["build_s123_v8"]
print(f"[1] 引擎加载, 耗时 {time.time()-t0:.0f}s")

FIXED = dict(max_same_sector=2, max_pe=60, min_turnover_pct=0.5,
             min_circ_mv_yi=50, max_circ_mv_yi=2000, max_peg=1.5, peg_preferred=1.5,
             min_roe_pct=12, chip_conc_pctl_threshold=0.50,
             min_list_years=3, preferred_weight=1.2)

# ---------- 1. 股票 Top3 无择时日频 NAV ----------
print("[2] 跑股票 Top3 无择时...", flush=True)
stock_nv, _, _ = run_strategy_v5(global_top_k=3, use_s123=False, verbose=False, **FIXED)
stock_nv = stock_nv[(stock_nv.index >= pd.Timestamp("2020-01-01"))]
print(f"    股票NAV: {len(stock_nv)}天, 期末 {stock_nv.iloc[-1]:.3f}")

# ---------- 2. V8 避险日收益 -> 日频 NAV ----------
print("[3] 构建 V8 避险组合日收益...", flush=True)
_, v8_daily = build_s123_v8()
v8_ret = pd.Series(v8_daily).sort_index()
v8_ret.index = pd.to_datetime(v8_ret.index.astype(str), format="%Y%m%d")
v8_ret = v8_ret[v8_ret.index >= pd.Timestamp("2020-01-01")]
v8_ret = v8_ret[~v8_ret.index.duplicated()]
v8_nav = (1 + v8_ret).cumprod()
print(f"    V8 NAV: {len(v8_nav)}天, 期末 {v8_nav.iloc[-1]:.3f}")

# ---------- 3. 口径统一: reindex 到工作日 + ffill ----------
BD = pd.date_range(pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31"), freq="B")
S = stock_nv.reindex(BD).ffill().dropna()
V = v8_nav.reindex(BD).ffill().dropna()
idx = S.index.intersection(V.index)
S = S.reindex(idx) / S.reindex(idx).iloc[0]
V = V.reindex(idx) / V.reindex(idx).iloc[0]
print(f"[4] 对齐后 {len(idx)} 个工作日")

def stats(s):
    s = s.dropna()
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

def monthly_rebalance_nav(S, V, w):
    """月频再平衡: 每月初(新月份首日)重置权重 w/(1-w), 月内买入持有自然漂移
    实现: 日频迭代, 换月首日先重置目标权重再计算当日收益, 日内权重随两腿相对涨跌漂移
    """
    S = S.astype(float)
    V = V.astype(float)
    sr = S.pct_change().fillna(0.0).values
    vr = V.pct_change().fillna(0.0).values
    month = S.index.to_period('M').values
    nav = np.empty(len(S))
    nav[0] = 1.0
    wS, wV = w, 1.0 - w
    for i in range(1, len(S)):
        if month[i] != month[i-1]:
            wS, wV = w, 1.0 - w
        r = wS * sr[i] + wV * vr[i]
        nav[i] = nav[i-1] * (1.0 + r)
        wS = wS * (1.0 + sr[i]) / (1.0 + r)
        wV = 1.0 - wS
    return pd.Series(nav, index=S.index)

# ---------- 4. 月频再平衡混合扫描 ----------
print("\n" + "=" * 92)
print("【跨资产配置】股票Top3 + V8(债+黄金) 月频再平衡(月内买入持有)")
print("=" * 92)
print(f"{'股票权重':<8} {'期末':>7} {'年化':>8} {'回撤':>8} {'夏普':>8} {'卡玛':>8}")
print("-" * 52)

WEIGHTS = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
rows = []
nav_dict = {}
for w in WEIGHTS:
    nav = monthly_rebalance_nav(S, V, w)
    nav_dict[f"股票{w:.0%}"] = nav
    st = stats(nav)
    rows.append({**{"股票权重": f"{w:.0%}"}, **{k: st[k] for k in ["期末","年化","回撤","夏普","卡玛"]}})
    print(f"{w:>7.0%} {st['期末']:>7.3f} {st['年化']:>8.1%} {st['回撤']:>8.1%} {st['夏普']:>8.2f} {st['卡玛']:>8.2f}")

rdf = pd.DataFrame(rows)
rdf.to_csv(os.path.join(OUT, "v5_cross_asset.csv"), index=False, encoding="utf-8-sig")

# ---------- 5. 找 MDD ≤ -20% 的最优混合 ----------
print("\n" + "=" * 92)
feasible = [r for r in rows if r["回撤"] >= -0.20]
if feasible:
    best = max(feasible, key=lambda r: r["年化"])
    print(f"★ 回撤≤-20% 的最优混合(月频再平衡): 股票权重 {best['股票权重']}")
    print(f"   年化={best['年化']:.1%}  回撤={best['回撤']:.1%}  夏普={best['夏普']:.2f}  卡玛={best['卡玛']:.2f}")
    st0 = rows[0]
    print(f"   对比纯股票(Top3): 年化 {st0['年化']:.1%}→{best['年化']:.1%} (Δ{best['年化']-st0['年化']:+.1%}), "
          f"回撤 {st0['回撤']:.1%}→{best['回撤']:.1%} (Δ{best['回撤']-st0['回撤']:+.1%})")
else:
    print("★ 回撤≤-20% 无可行解，回撤下限见上表")

# ---------- 6. 买入持有(不调仓) 对照 ----------
print("\n" + "=" * 92)
print("【对照】买入持有(初始权重不调仓, 股票权重自然漂移)")
print("=" * 92)
print(f"{'初始股票权重':<10} {'期末':>7} {'年化':>8} {'回撤':>8} {'夏普':>8} {'卡玛':>8}")
print("-" * 52)
for w in [0.90, 0.80, 0.70, 0.60, 0.50]:
    combined = w * S + (1 - w) * V
    st = stats(combined)
    print(f"{w:>9.0%} {st['期末']:>7.3f} {st['年化']:>8.1%} {st['回撤']:>8.1%} {st['夏普']:>8.2f} {st['卡玛']:>8.2f}")

# ---------- 7. 逐年 ----------
print("\n" + "=" * 92)
print("【逐年】(月频再平衡)")
print("=" * 92)
years = [("2020","2020-01-01","2021-01-01"),("2021","2021-01-01","2022-01-01"),
         ("2022","2022-01-01","2023-01-01"),("2023","2023-01-01","2024-01-01"),
         ("2024","2024-01-01","2025-01-01"),("2025","2025-01-01","2026-01-01")]
cols = ["股票100%", "股票80%", "股票70%", "股票60%", "股票50%", "股票40%"]
print(f"{'年份':<6}", end="")
for c in cols:
    print(f" {c:<9}", end="")
print()
print("-" * 62)
for yr, s0, s1 in years:
    print(f"{yr:<6}", end="")
    for c in cols:
        s = nav_dict[c][(nav_dict[c].index >= pd.Timestamp(s0)) & (nav_dict[c].index < pd.Timestamp(s1))].dropna()
        if len(s) < 5:
            print(f" {'n/a':<9}", end="")
            continue
        s = s / s.iloc[0]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        ann = s.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else np.nan
        print(f" {ann:>8.1%}", end="")
    print()

# ---------- 8. 画图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
ax = axes[0]
for c in ["股票100%", "股票80%", "股票70%", "股票60%", "股票50%", "股票40%"]:
    s = nav_dict[c].dropna()
    ax.plot(s.index, s.values, lw=1.6, label=c, alpha=0.9)
ax.axhline(1.0, color="gray", lw=0.5, ls=":")
ax.set_title("跨资产配置(月频再平衡): 股票Top3 + V8(短债+信用债+黄金)", fontsize=13)
ax.set_ylabel("净值(基准=1)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

ax2 = axes[1]
xs, ys, labels = [], [], []
for r in rows:
    xs.append(abs(r["回撤"]) * 100)
    ys.append(r["年化"] * 100)
    labels.append(r["股票权重"])
ax2.scatter(xs, ys, s=80, c="#c0392b", zorder=3)
for i, lb in enumerate(labels):
    ax2.annotate(lb, (xs[i], ys[i]), textcoords="offset points", xytext=(8, 4), fontsize=9)
ax2.axvline(20, color="green", lw=1.2, ls="--", label="回撤 -20% 目标线")
ax2.set_xlabel("最大回撤 (%)")
ax2.set_ylabel("年化收益 (%)")
ax2.set_title("回撤-收益权衡 (越靠左上越好, 竖线为目标回撤)", fontsize=12)
ax2.legend(loc="best", fontsize=9)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "v5_cross_asset.png"), dpi=150)
plt.close(fig)

print(f"\n[完成] 耗时 {time.time()-t0:.0f}s")
print(f"  - v5_cross_asset.csv")
print(f"  - v5_cross_asset.png")
