# -*- coding: utf-8 -*-
"""
可交易宽基 ETF/指数 vs V3策略 vs 全市场等权 (2018-2026)

回答: 全市场等权(15.3%) 到底对应哪个可交易标的?
  - 中证1000ETF 512100 (可交易, 中小盘)
  - 沪深300ETF 510300 (可交易, 大盘)
  - 中证1000指数 000852 / 中证500 000905 / 沪深300 000300 / 上证50 000016 / 中证2000 932000

用 pct_chg 累乘(避免拆分跳变), 起点 2018-01 与 V3 对齐。
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
SR = os.path.join(ROOT, "research", "sector_rotation")
OUT = os.path.join(SR, "results")
IDX_DIR = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily")
INIT = 1_000_000

def read_nav(path):
    nav = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    nav.index = pd.to_datetime(nav.index)
    return nav.dropna()

nav_v3 = read_nav(os.path.join(OUT, "fullmarket_moneyflow_v3_2015_nav.csv"))
ew = pd.read_csv(os.path.join(OUT, "equal_weight_benchmark_compare.csv"), index_col=0, parse_dates=True)
ew_all = (ew["全市场等权"] * INIT).reindex(nav_v3.index).ffill()

# ---------- ETF/指数 ----------
ETFS = {
    "中证1000ETF 512100": "512100.SH",
    "沪深300ETF 510300": "510300.SH",
    "中证1000指数 000852": "000852.SH",
    "中证2000指数 932000": "932000.CSI",
    "中证500指数 000905": "000905.SH",
    "沪深300指数 000300": "000300.SH",
    "上证50指数 000016": "000016.SH",
}
# 注意: 中证2000(932000) 2023-08 才发布, 此前"历史"为指数公司回算, 引用其全区间 CAGR 需加脚注

def load_etf(code):
    df = pd.read_parquet(os.path.join(IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["r"] = df["pct_chg"].fillna(0) / 100.0
    nav = (1 + df["r"]).cumprod()
    s = pd.Series(nav.values, index=df["trade_date"])
    return s / s.iloc[0] * INIT

def stats(nav):
    nav = nav.dropna()  # 晚成立标的剔除上市前的 NaN, 年化按自身有效区间计算(否则全区间 8.9 年摊薄 CAGR)
    tr = nav.iloc[-1] / INIT - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = (1 + tr) ** (1 / years) - 1 if years > 0 else np.nan
    peak = nav.cummax()
    mdd = ((nav - peak) / peak).min()
    rets = nav.pct_change().dropna()
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
    return tr, ann, mdd, sharpe, nav.iloc[-1]

series = {"V3资金流策略": nav_v3, "全市场等权(不可交易)": ew_all}
for name, code in ETFS.items():
    try:
        nav = load_etf(code).reindex(nav_v3.index).ffill()
        if nav.notna().sum() > 200:
            series[name] = nav
    except Exception as e:
        print(f"  [跳过] {name}: {e}")

print(f"[1] 数据加载完成, {time.time()-t0:.0f}s")

print("\n" + "=" * 100)
print(f"{'系列':<24}{'累计':>9}{'年化':>9}{'MaxDD':>9}{'夏普':>7}{'期末(万)':>10}")
print("-" * 100)
res = {}
for name, nav in series.items():
    tr, ann, mdd, shp, end = stats(nav)
    res[name] = (ann, mdd, shp)
    print(f"{name:<24}{tr:>9.1%}{ann:>9.1%}{mdd:>9.1%}{shp:>7.2f}{end/1e4:>10.0f}")
print("=" * 100)

# 分年度
def yearly(nav):
    out = {}
    prev = INIT
    for y, v in nav.resample("Y").last().items():
        out[y.year] = v / prev - 1
        prev = v
    return out

print("\n--- 分年度收益 ---")
yy = {name: yearly(nav) for name, nav in series.items()}
years = sorted(yy["V3资金流策略"].keys())
print(f"{'年份':<6}" + "".join(f"{name:>18}" for name in series))
for y in years:
    row = "".join(f"{yy[name].get(y, np.nan):>18.1%}" for name in series)
    print(f"{y:<6}{row}")

comb = pd.DataFrame({name: nav / INIT for name, nav in series.items()})
comb.to_csv(os.path.join(OUT, "v3_vs_tradable_etf.csv"), encoding="utf-8-sig")
print(f"\n[保存] v3_vs_tradable_etf.csv  总耗时 {time.time()-t0:.0f}s")
