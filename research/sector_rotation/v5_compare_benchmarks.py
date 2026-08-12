# -*- coding: utf-8 -*-
"""V5-BEST 与其他"正确"(无前视)策略 + 纯ETF基准 的同期对比

复用一个数据加载流程:
  1. exec backtest_stock_picking_v6.py 到 "# 8. v6 组合测试" 之前 (数据+run_strategy_v6)
  2. 提取 v4a3_fixed_compare.py 的 build_scores_v4 / run_v4a3, 复用同一份数据跑 v4-A3修复版
  3. V5-BEST: 直接读 results/v5_best_nav.csv
  4. 纯ETF基准: 512100(中证1000ETF) / 510300(沪深300ETF) 月频NAV (pct_chg 累乘)
  5. 全行业等权: industry_ret.csv 月度等权收益累乘
输出: results/v5_vs_benchmarks_monthly.csv + 控制台指标
"""
import os, sys, time
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
SR = os.path.join(ROOT, "research", "sector_rotation")
OUT = os.path.join(SR, "results")
IDX_DIR = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily")
t00 = time.time()

# ---------- 1. 复用 v6 引擎 (数据加载 + run_strategy_v6 + calc_metrics) ----------
print("[1] exec v6 数据加载+引擎...", flush=True)
v6_src = open(os.path.join(SR, "backtest_stock_picking_v6.py"), encoding="utf-8").read()
cut = v6_src.index("# 8. v6 组合测试")
ns = {}
exec(compile(v6_src[:cut], "backtest_stock_picking_v6.py", "exec"), ns)

# ---------- 2. 提取 v4a3 引擎 ----------
print("[2] 提取 v4a3 引擎 (build_scores_v4 / run_v4a3)...", flush=True)
v4_src = open(os.path.join(SR, "v4a3_fixed_compare.py"), encoding="utf-8").read()
s = v4_src.index("def build_scores_v4"); e = v4_src.index("# ---- 板块信号")
exec(compile(v4_src[s:e], "v4a3_build", "exec"), ns)
s = v4_src.index("def run_v4a3"); e = v4_src.index("def metrics")
exec(compile(v4_src[s:e], "v4a3_run", "exec"), ns)

# ---------- 3. 跑 V6-S2 (Top3+PE≤80+大盘100亿) ----------
print("[3] 回测 V6-S2...", flush=True)
nv_v6s2, trs6, _ = ns["run_strategy_v6"](
    global_top_k=3, max_same_sector=2, max_pe=80, min_turnover_pct=0.3,
    min_circ_mv_yi=100, max_circ_mv_yi=2000, chip_conc_pctl_threshold=0.70,
    peg_bonus_threshold=1.5, min_list_years=1.5, preferred_weight=1.20,
    pe_pct_thr=ns["PE_PCT_THR"])
nv_v6s2 = nv_v6s2[nv_v6s2.index >= pd.Timestamp("2020-01-01")]
print(f"    V6-S2 期末={nv_v6s2.iloc[-1]/1e4:.0f}万", flush=True)

# ---------- 4. 跑 v4-A3 修复版 (同数据, 换v4评分) ----------
print("[4] 回测 v4-A3 修复版...", flush=True)
ns["ml_scored"] = ns["build_scores_v4"](ns["ml2"])
nv_v4fix, trs4 = ns["run_v4a3"](shift=True)
nv_v4fix = nv_v4fix[nv_v4fix.index >= pd.Timestamp("2020-01-01")]
print(f"    v4-A3修复 期末={nv_v4fix.iloc[-1]/1e4:.0f}万", flush=True)

# ---------- 5. V5-BEST (读CSV, 日频) ----------
print("[5] 读 V5-BEST 净值...", flush=True)
v5df = pd.read_csv(os.path.join(OUT, "v5_best_nav.csv"), index_col=0)
nv_v5 = v5df.iloc[:, 0]
nv_v5.index = pd.to_datetime(nv_v5.index)

# ---------- 6. ETF 基准 (月频 NAV, pct_chg 累乘避免拆分跳变) ----------
print("[6] ETF 基准...", flush=True)
def etf_monthly(code, start_ym="202001", end_ym="202512"):
    fp = os.path.join(IDX_DIR, f"{code}.parquet")
    df = pd.read_parquet(fp)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date").reset_index(drop=True)
    df = df[(df["trade_date"] >= start_ym) & (df["trade_date"] <= end_ym + "31")]
    df = df[df["trade_date"] <= end_ym + "31"]
    daily = df["pct_chg"].fillna(0) / 100.0
    nav = (1 + daily).cumprod()
    df = df.copy(); df["nav"] = nav.values
    df["ym"] = df["trade_date"].str[:6]
    m = df.groupby("ym")["nav"].last()
    m.index = pd.to_datetime(m.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    return m / m.iloc[0]

etf_1000 = etf_monthly("512100.SH")   # 中证1000ETF
etf_300 = etf_monthly("510300.SH")    # 沪深300ETF
print(f"    512100 期末NAV={etf_1000.iloc[-1]:.3f}, 510300 期末NAV={etf_300.iloc[-1]:.3f}", flush=True)

# ---------- 7. 全行业等权 (industry_ret.csv 月度等权) ----------
print("[7] 全行业等权...", flush=True)
ret_df = pd.read_csv(os.path.join(OUT, "industry_ret.csv"), index_col=0)
ret_df.index = pd.to_datetime(ret_df.index, format="%Y%m%d")
ret_df = ret_df[(ret_df.index >= pd.Timestamp("2020-01-31")) & (ret_df.index <= pd.Timestamp("2025-12-31"))]
ew_ret = ret_df.mean(axis=1, skipna=True)
ew_nav = (1 + ew_ret).cumprod()
ew_nav = ew_nav.resample("M").last()  # 统一月末索引
print(f"    行业等权 期末NAV={ew_nav.iloc[-1]:.3f}", flush=True)

# ---------- 8. 对齐为月度, 归一化起点=1, 合并 ----------
print("[8] 对齐合并...", flush=True)
def monthly(s, name):
    m = s.resample("M").last()
    return m / m.iloc[0]

cols = {}
cols["V5-BEST"] = monthly(nv_v5, "v5")
cols["V6-S2"] = monthly(nv_v6s2, "v6")
cols["V4-A3修复"] = monthly(nv_v4fix, "v4")
cols["中证1000ETF"] = etf_1000
cols["沪深300ETF"] = etf_300
cols["全行业等权"] = ew_nav
comb = pd.DataFrame(cols).dropna(how="all")
comb.index.name = "ym"
comb.to_csv(os.path.join(OUT, "v5_vs_benchmarks_monthly.csv"), encoding="utf-8-sig")
print(f"    合并: {len(comb)}个月 ({comb.index.min()} ~ {comb.index.max()})", flush=True)

# ---------- 9. 指标 ----------
print("[9] 指标对比...", flush=True)
def stats(m):
    ret = m.pct_change().dropna()
    yrs = len(ret) / 12
    cagr = m.iloc[-1] ** (1 / yrs) - 1
    mdd = ((m - m.cummax()) / m.cummax()).min()
    shp = ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(12)
    return cagr, mdd, shp, m.iloc[-1]

print("\n" + "=" * 78)
print(f"{'系列':<14}{'年化':>9}{'累计':>9}{'MaxDD':>9}{'夏普':>7}{'期末NAV':>9}")
print("-" * 78)
res = {}
for c in comb.columns:
    cagr, mdd, shp, end = stats(comb[c])
    res[c] = (cagr, mdd, shp, end)
    print(f"{c:<14}{cagr:>9.1%}{end-1:>9.1%}{mdd:>9.1%}{shp:>7.2f}{end:>9.2f}")
print("=" * 78)

# ---------- 10. 对比曲线图 ----------
print("[10] 保存对比曲线 PNG...", flush=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False

STYLE = {
    "V5-BEST": dict(color="#c0392b", lw=2.2, ls="-"),
    "V6-S2": dict(color="#8e44ad", lw=1.8, ls="-"),
    "V4-A3修复": dict(color="#f39c12", lw=1.6, ls="-"),
    "中证1000ETF": dict(color="#7f8c8d", lw=1.4, ls="--"),
    "沪深300ETF": dict(color="#95a5a6", lw=1.4, ls="-."),
    "全行业等权": dict(color="#2980b9", lw=1.8, ls="-"),
}
fig, ax = plt.subplots(figsize=(14, 7))
for c in comb.columns:
    s = comb[c].dropna()
    ax.plot(s.index, s.values, lw=STYLE[c]["lw"], ls=STYLE[c]["ls"], color=STYLE[c]["color"], label=c)
ax.axhline(1.0, color="gray", lw=0.6, ls=":")
ax.set_title("V5-BEST vs 无前视策略 vs 纯ETF (2020-01 ~ 2025-12, 月度净值)", fontsize=13)
ax.set_ylabel("净值(基准=1)")
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v5_vs_benchmarks.png"), dpi=150)
plt.close(fig)

print(f"\n[完成] 总耗时 {time.time()-t00:.0f}s, 输出: v5_vs_benchmarks_monthly.csv + v5_vs_benchmarks.png")
