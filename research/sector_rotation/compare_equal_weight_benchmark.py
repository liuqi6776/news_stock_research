# -*- coding: utf-8 -*-
"""
等权基准对比: 全市场 GBDT 选股策略(V2/V3) vs 全市场等权 vs 传统行业等权

等权基准口径:
  - 每个交易日, 全市场(或剔除科技后的传统行业)所有有行情股票的 pct_chg 等权平均
  - 累乘得到净值, 起点 2018-01 与 V2/V3 相同 (100万)
  - 无成本(理论等权市场组合), 作为"是否产生超额"的对照基准

输出: results/equal_weight_benchmark_compare.csv + 控制台指标
"""
import os, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
SR = os.path.join(ROOT, "research", "sector_rotation")
OUT = os.path.join(SR, "results")
INIT = 1_000_000

# ---------- 1. 读 V2/V3 净值 ----------
def read_nav(path):
    nav = pd.read_csv(path, index_col=0, parse_dates=True)
    nav = nav.iloc[:, 0].astype(float)
    nav.index = pd.to_datetime(nav.index)
    return nav.dropna()

nav_v2 = read_nav(os.path.join(OUT, "fullmarket_moneyflow_2015_nav.csv"))
nav_v3 = read_nav(os.path.join(OUT, "fullmarket_moneyflow_v3_2015_nav.csv"))
print(f"[1] V2/V3净值: {nav_v3.index[0].date()} ~ {nav_v3.index[-1].date()}, {len(nav_v3)}天, {time.time()-t0:.0f}s")

# ---------- 2. 全市场日频行情 (2018起) ----------
print("[2] 读全市场日频行情 2018起...")
px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
parts = []
for f in px_files:
    if os.path.getsize(f) <= 1024:
        continue
    d = os.path.basename(f)[:8]
    if d < "20180101":
        continue
    df = pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"])
    if len(df):
        parts.append(df)
px = pd.concat(parts, ignore_index=True)
px["trade_date"] = pd.to_datetime(px["trade_date"].astype(str), format="%Y%m%d")
px["r"] = px["pct_chg"] / 100.0
print(f"    行情: {len(px):,}行, {px['ts_code'].nunique()}只, "
      f"{px['trade_date'].min().date()}~{px['trade_date'].max().date()}, {time.time()-t0:.0f}s")

# ---------- 3. 全市场等权日度收益 ----------
ew_all = px.groupby("trade_date")["r"].mean().sort_index()
nav_all = (1 + ew_all).cumprod() * INIT
print(f"[3] 全市场等权: {len(nav_all)}天, {time.time()-t0:.0f}s")

# ---------- 4. 传统行业等权 (剔除科技行业) ----------
print("[4] 传统行业等权...")
ind = pd.read_parquet(os.path.join(DATA, "industry1", "industry.parquet"))
ind_map = dict(zip(ind["ts_code"], ind["industry"]))
TECH_KEYWORDS = [
    "半导体", "元器件", "IT设备", "计算机设备", "软件服务", "IT服务", "软件开发",
    "互联网", "通信设备", "通信服务", "游戏", "数字媒体", "广告营销", "影视院线",
    "出版业", "电视广播", "光学光电子", "消费电子", "其他电子", "电子化学品",
    "电池", "电机", "风电设备", "光伏设备", "电源设备", "电网设备",
    "航天装备", "航空装备", "地面兵装", "船舶装备", "军工电子", "航海装备",
]
px["industry"] = px["ts_code"].map(ind_map).fillna("其他")
px_trad = px[~px["industry"].isin(TECH_KEYWORDS)]
ew_trad = px_trad.groupby("trade_date")["r"].mean().sort_index()
nav_trad = (1 + ew_trad).cumprod() * INIT
print(f"    传统行业等权: {len(nav_trad)}天, 传统占比={px_trad['ts_code'].nunique()/px['ts_code'].nunique():.1%}, {time.time()-t0:.0f}s")

# ---------- 5. 对齐到 V3 净值日期 ----------
idx = nav_v3.index
nav_all = nav_all.reindex(idx).ffill().dropna()
nav_trad = nav_trad.reindex(idx).ffill().dropna()

# ---------- 6. 指标 ----------
def stats(nav):
    tr = nav.iloc[-1] / INIT - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = (1 + tr) ** (1 / years) - 1 if years > 0 else np.nan
    peak = nav.cummax()
    mdd = ((nav - peak) / peak).min()
    rets = nav.pct_change().dropna()
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
    return tr, ann, mdd, sharpe, nav.iloc[-1]

series = {"V3资金流(12因子)": nav_v3, "V2资金流(4因子)": nav_v2,
          "全市场等权": nav_all, "传统行业等权": nav_trad}

print("\n" + "=" * 92)
print(f"{'系列':<18}{'累计':>9}{'年化':>9}{'MaxDD':>9}{'夏普':>7}{'期末(万)':>10}")
print("-" * 92)
res = {}
for name, nav in series.items():
    tr, ann, mdd, shp, end = stats(nav)
    res[name] = (ann, mdd, shp)
    print(f"{name:<18}{tr:>9.1%}{ann:>9.1%}{mdd:>9.1%}{shp:>7.2f}{end/1e4:>10.0f}")
print("=" * 92)

# 超额 (相对全市场等权)
print("\n--- 相对全市场等权的超额 ---")
for name in ["V3资金流(12因子)", "V2资金流(4因子)"]:
    ann_s, mdd_s, shp_s = res[name]
    ann_b, mdd_b, shp_b = res["全市场等权"]
    print(f"  {name}: 年化超额 {ann_s-ann_b:+.1%} | 夏普差 {shp_s-shp_b:+.2f} | 回撤差 {mdd_s-mdd_b:+.1%}")

# 分年度对比
print("\n--- 分年度收益对比 ---")
def yearly(nav):
    out = {}
    yr = nav.resample("Y").last()
    prev = INIT
    for y, v in yr.items():
        out[y.year] = v / prev - 1
        prev = v
    return out

yy = {name: yearly(nav) for name, nav in series.items()}
years = sorted(yy["V3资金流(12因子)"].keys())
print(f"{'年份':<6}" + "".join(f"{name:>16}" for name in series))
for y in years:
    row = "".join(f"{yy[name].get(y, np.nan):>16.1%}" for name in series)
    print(f"{y:<6}{row}")

# 保存对比表
comb = pd.DataFrame({name: nav / INIT for name, nav in series.items()})
comb.to_csv(os.path.join(OUT, "equal_weight_benchmark_compare.csv"), encoding="utf-8-sig")
print(f"\n[保存] equal_weight_benchmark_compare.csv  总耗时 {time.time()-t0:.0f}s")
