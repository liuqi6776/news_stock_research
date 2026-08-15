# -*- coding: utf-8 -*-
"""
验证"基准错配"假设: 用中证1000(512100) vs 沪深300 做趋势择时基准
固定股票端 = ROE12/K3/PEG1.5/CHIP50/MV50/YR3
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
build_s123_v8 = ns["build_s123_v8"]
_, v8_daily = build_s123_v8()

FIXED = dict(global_top_k=3, max_same_sector=2, max_pe=60, min_turnover_pct=0.5,
             min_circ_mv_yi=50, max_circ_mv_yi=2000, max_peg=1.5, peg_preferred=1.5,
             min_roe_pct=12, chip_conc_pctl_threshold=0.50,
             min_list_years=3, preferred_weight=1.2)

def load_index_close(code):
    pq = os.path.join(IDX_DIR, f"{code}.parquet")
    edf = pd.read_parquet(pq)
    edf["trade_date"] = edf["trade_date"].astype(str)
    edf = edf.sort_values("trade_date").reset_index(drop=True)
    if "close" in edf.columns:
        s = edf.set_index("trade_date")["close"].astype(float)
    else:
        # 用 pct_chg 累乘得到价格指数(等价于归一化close)
        nav = (1 + edf["pct_chg"].fillna(0) / 100.0).cumprod()
        s = nav
        s.index = edf["trade_date"]
    s.index = pd.to_datetime(s.index, format="%Y%m%d")
    return s[~s.index.duplicated()].sort_index()

def close_to_sig(bull_series):
    df = pd.DataFrame({"bull": bull_series.astype(int)})
    df["ym"] = [d.year * 100 + d.month for d in df.index]
    monthly = df.groupby("ym")["bull"].last()
    return {int(ym): (3 if v == 1 else 0) for ym, v in monthly.items()}

def trend_sig(close, win):
    ma = close.rolling(win).mean()
    bull = (close > ma) & ma.notna()
    return close_to_sig(bull)

def dual_sig(close, wfast, wslow):
    f = close.rolling(wfast).mean()
    sl = close.rolling(wslow).mean()
    bull = (f > sl) & sl.notna()
    return close_to_sig(bull)

# 沪深300 + 中证1000
close_300 = load_index_close("510300.SH")
close_1000 = load_index_close("512100.SH")
print(f"沪深300: {close_300.index[0].date()}~{close_300.index[-1].date()} ({len(close_300)}天)")
print(f"中证1000: {close_1000.index[0].date()}~{close_1000.index[-1].date()} ({len(close_1000)}天)")

signals = {
    "无择时": None,
    "沪深300_MA200": trend_sig(close_300, 200),
    "中证1000_MA200": trend_sig(close_1000, 200),
    "中证1000_MA60": trend_sig(close_1000, 60),
    "中证1000_双均线50/200": dual_sig(close_1000, 50, 200),
    "中证1000_MA120": trend_sig(close_1000, 120),
}

print(f"\n信号持仓统计:")
for name, sm in signals.items():
    if sm is None:
        print(f"  {name}: 无择时")
    else:
        bulls = sum(1 for v in sm.values() if v == 3)
        print(f"  {name}: {len(sm)}月, 牛市{sum(1 for v in sm.values() if v==3)}月, 熊市{sum(1 for v in sm.values() if v==0)}月")

results = {}
for name, sm in signals.items():
    if sm is None:
        nv, trs, _ = run_strategy_v5(use_s123=False, verbose=False, **FIXED)
    else:
        nv, trs, _ = run_strategy_v5(use_s123=True, sig_map=sm, v8_daily=v8_daily, verbose=False, **FIXED)
    results[name] = nv

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

print(f"\n{'='*80}")
print(f"【全周期 2020-2025】")
print(f"{'='*80}")
print(f"{'策略':<22} {'期末':>7} {'年化':>8} {'回撤':>8} {'夏普':>8}")
print("-" * 58)
for col in comb.columns:
    r = stats_row(comb[col], TRAIN_START, pd.Timestamp("2026-01-01"))
    if r:
        print(f"{col:<22} {r['期末']:>7.3f} {r['年化']:>7.1%} {r['回撤']:>7.1%} {r['夏普']:>8.2f}")

print(f"\n{'='*80}")
print(f"【逐年年化】")
print(f"{'='*80}")
years = [("2020","2020-01-01","2021-01-01"),("2021","2021-01-01","2022-01-01"),
         ("2022","2022-01-01","2023-01-01"),("2023","2023-01-01","2024-01-01"),
         ("2024","2024-01-01","2025-01-01"),("2025","2025-01-01","2026-01-01")]
print(f"{'年份':<6}", end="")
for col in comb.columns:
    print(f" {col:<14}", end="")
print()
print("-" * 120)
for yr, s0, s1 in years:
    print(f"{yr:<6}", end="")
    for col in comb.columns:
        s = comb[col][(comb[col].index >= pd.Timestamp(s0)) & (comb[col].index < pd.Timestamp(s1))].dropna()
        if len(s) < 5:
            print(f" {'n/a':<14}", end="")
            continue
        s = s / s.iloc[0]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        ann = s.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else np.nan
        print(f" {ann:>9.1%}   ", end="")
    print()

comb.resample("M").last().to_csv(os.path.join(OUT, "v5_timing_zhongzheng1000_monthly.csv"), encoding='utf-8-sig')
print(f"\n[完成] 耗时 {time.time()-t0:.0f}s")
