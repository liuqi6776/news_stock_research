# -*- coding: utf-8 -*-
"""
牛熊判断方法对比: s123估值择时 vs 趋势跟踪(均线/双均线)
固定股票端 = ROE12 好配置 (K3/PEG1.5/CHIP50/MV50/YR3)
只换择时信号, 隔离"牛熊判断"的贡献, 找更好的曲线

信号(月度状态机, T-1月信号→T月生效, 无前视):
  1. 无择时         (基线)
  2. s123估值择时   (当前: 沪深300 PE分位<20% + ERP>+1σ + 深跌<=-25%)
  3. MA200趋势      (收盘 > 200日均线 → 持有)
  4. MA60趋势       (收盘 > 60日均线 → 持有, 更灵敏)
  5. MA250年线趋势  (收盘 > 250日均线 → 持有)
  6. 双均线50/200   (MA50 > MA200 → 持有)
"""
import os, time
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
SR = os.path.join(ROOT, "research", "sector_rotation")
OUT = os.path.join(SR, "results")
IDX_DIR = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily")

# ============================================================
# 1. 加载引擎 + 数据
# ============================================================
v5_path = os.path.join(SR, 'backtest_stock_picking_v5.py')
v5_code = open(v5_path, encoding='utf-8').read()
marker = '# ============================================================\n# 8. v5'
ns = {}
exec(v5_code.split(marker)[0], ns)
run_strategy_v5 = ns["run_strategy_v5"]
build_s123_v8 = ns["build_s123_v8"]

sig_map_s123, v8_daily = build_s123_v8()
print(f"[1] 引擎+s123信号就绪, 耗时 {time.time()-t0:.0f}s")

# 沪深300 日收盘价 (用于趋势信号)
pe = ns.get("pe")  # 引擎里可能有; 没有则 fetch
if pe is None:
    from timing_dingtou import fetch_pe_csi300
    pe = fetch_pe_csi300()
close = pe["close"].copy()
close.index = pd.to_datetime(close.index)
close = close[~close.index.duplicated()].sort_index()
print(f"[2] 沪深300日收盘: {close.index[0].date()} ~ {close.index[-1].date()} ({len(close)}天)")

# ============================================================
# 2. 构建各种牛熊信号 → 月度 sig_map (值 0/3, 配合引擎 >=3进场 <=1离场)
# ============================================================
def close_to_monthly_sig(bull_mask_series):
    """把日频布尔(True=牛/持有)序列转成月度 sig_map {ym: 3或0}, 用月末信号"""
    df = pd.DataFrame({"bull": bull_mask_series.astype(int)})
    df["ym"] = [d.year * 100 + d.month for d in df.index]
    monthly = df.groupby("ym")["bull"].last()  # 每月最后一天信号
    return {int(ym): (3 if v == 1 else 0) for ym, v in monthly.items()}

signals = {}
signals["无择时"] = None  # 特殊: use_s123=False

# s123 (已有)
signals["s123估值"] = sig_map_s123

# 趋势跟踪
for win, name in [(200, "MA200趋势"), (60, "MA60趋势"), (250, "MA250年线")]:
    ma = close.rolling(win).mean()
    bull = (close > ma) & ma.notna()
    signals[name] = close_to_monthly_sig(bull)

# 双均线 50/200
ma50 = close.rolling(50).mean()
ma200 = close.rolling(200).mean()
bull = (ma50 > ma200) & ma200.notna()
signals["双均线50/200"] = close_to_monthly_sig(bull)

print(f"[3] 信号构建完成: {list(signals.keys())}")
for name, sm in signals.items():
    if sm is None:
        print(f"    {name}: 无择时")
    else:
        bulls = sum(1 for v in sm.values() if v == 3)
        print(f"    {name}: {len(sm)}个月, 牛市(持有){bulls}个月, 熊市(避险){len(sm)-bulls}个月")

# ============================================================
# 3. 固定股票端配置, 跑各信号
# ============================================================
FIXED = dict(global_top_k=3, max_same_sector=2, max_pe=60, min_turnover_pct=0.5,
             min_circ_mv_yi=50, max_circ_mv_yi=2000, max_peg=1.5, peg_preferred=1.5,
             min_roe_pct=12, chip_conc_pctl_threshold=0.50,
             min_list_years=3, preferred_weight=1.2)

print(f"\n[4] 跑 {len(signals)} 个信号 (固定股票端 ROE12/K3/PEG1.5/CHIP50/MV50/YR3)")
results = {}
for name, sm in signals.items():
    if sm is None:
        nv, trs, _ = run_strategy_v5(use_s123=False, verbose=False, **FIXED)
    else:
        nv, trs, _ = run_strategy_v5(use_s123=True, sig_map=sm, v8_daily=v8_daily,
                                     verbose=False, **FIXED)
    results[name] = nv
    print(f"    {name}: {len(nv)}天, 期末{nv.iloc[-1]:.3f}")

# ============================================================
# 4. 基准 ETF + 行业等权
# ============================================================
TRAIN_START = pd.Timestamp("2020-01-01")
def etf_daily_nav(code):
    pq = os.path.join(IDX_DIR, f"{code}.parquet")
    edf = pd.read_parquet(pq)
    edf["trade_date"] = edf["trade_date"].astype(str)
    edf = edf.sort_values("trade_date").reset_index(drop=True)
    edf["dt"] = pd.to_datetime(edf["trade_date"], format="%Y%m%d")
    edf = edf[(edf["dt"] >= TRAIN_START) & (edf["dt"] < pd.Timestamp("2026-01-01"))].set_index("dt")
    return (1 + edf["pct_chg"].fillna(0) / 100.0).cumprod()

results["中证1000ETF"] = etf_daily_nav("512100.SH")
results["沪深300ETF"] = etf_daily_nav("510300.SH")

ir_df = pd.read_csv(os.path.join(OUT, "industry_ret.csv"), index_col=0)
ir_df.index = pd.to_datetime(ir_df.index)
ew_daily = (1 + ir_df.mean(axis=1, skipna=True)).cumprod().resample("M").last().reindex(
    pd.date_range(TRAIN_START, pd.Timestamp("2025-12-31"), freq='B')).ffill()
results["行业等权"] = ew_daily

# ============================================================
# 5. 指标
# ============================================================
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
    return {"期末": s.iloc[-1], "年化": ann, "回撤": mdd, "夏普": shp}

def print_table(title, start, end):
    print(f"\n{'='*90}")
    print(f"【{title}】({start.date()} ~ {end.date()})")
    print(f"{'='*90}")
    print(f"{'策略':<16} {'期末':>7} {'年化':>8} {'回撤':>8} {'夏普':>8}")
    print("-" * 52)
    for col in comb.columns:
        r = stats_row(comb[col], start, end)
        if r is None:
            continue
        print(f"{col:<16} {r['期末']:>7.3f} {r['年化']:>7.1%} {r['回撤']:>7.1%} {r['夏普']:>8.2f}")

print_table("全周期 2020-2025", TRAIN_START, pd.Timestamp("2026-01-01"))
print_table("样本外 2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"))

# 逐年明细
print(f"\n{'='*90}")
print("【逐年年化】")
print(f"{'='*90}")
years = [("2020", "2020-01-01", "2021-01-01"), ("2021", "2021-01-01", "2022-01-01"),
         ("2022", "2022-01-01", "2023-01-01"), ("2023", "2023-01-01", "2024-01-01"),
         ("2024", "2024-01-01", "2025-01-01"), ("2025", "2025-01-01", "2026-01-01")]
print(f"{'年份':<6}", end="")
for col in comb.columns:
    print(f" {col:<12}", end="")
print()
print("-" * 110)
for yr, s0, s1 in years:
    print(f"{yr:<6}", end="")
    for col in comb.columns:
        s = comb[col][(comb[col].index >= pd.Timestamp(s0)) & (comb[col].index < pd.Timestamp(s1))].dropna()
        if len(s) < 5:
            print(f" {'  n/a':<12}", end="")
            continue
        s = s / s.iloc[0]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        ann = s.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else np.nan
        print(f" {ann:>8.1%}  ", end="")
    print()

# ============================================================
# 6. 保存
# ============================================================
comb.resample("M").last().to_csv(os.path.join(OUT, "v5_timing_compare_monthly.csv"),
                                  encoding='utf-8-sig')

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
ax.set_title("牛熊判断方法对比: s123估值 vs 趋势跟踪 (固定ROE12股票端)", fontsize=13)
ax.set_ylabel("净值(基准=1)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

ax2 = axes[1]
for c in comb.columns:
    s = comb[c].dropna()
    if len(s) == 0:
        continue
    s = s[s.index >= pd.Timestamp("2024-01-01")]
    if len(s) == 0:
        continue
    s = s / s.iloc[0]
    ax2.plot(s.index, s.values, lw=1.6, label=c, alpha=0.9)
ax2.axhline(1.0, color="gray", lw=0.5, ls=":")
ax2.set_title("2024-2025 特写", fontsize=12)
ax2.set_ylabel("净值(2024-01=1)")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "v5_timing_compare.png"), dpi=150)
plt.close(fig)

print(f"\n[完成] 耗时 {time.time()-t0:.0f}s")
print(f"  - v5_timing_compare_monthly.csv")
print(f"  - v5_timing_compare.png")
