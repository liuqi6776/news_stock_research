# -*- coding: utf-8 -*-
"""
④ 组合层整合: 全市场个股端(V3资金流) + s123择时 + V8避险

对比:
  基准   V3 纯个股 (始终满仓, 无市场择时)  [22.6%/回撤-28.4%/夏普1.06]
  整合   V3 + s123择时(≥3进/≤1出) + V8避险(511990/511260/518880)

个股端规则保持一致: 月度 Top20/每行业≤3, 30%止盈, 180天时间止损, 买入10bps/卖出15bps
"""
import os, sys, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "fund_research", "studies", "rotation_dingtou"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore
from etf_optimize_backtest2 import load_hv_daily

OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
DATA = r"D:/iquant_data/data_v2"
t0 = time.time()

# ============ 1. 加载 V3 预测 ============
pred = pd.read_csv(os.path.join(OUT_DIR, "fullmarket_moneyflow_v3_oos_pred.csv"))
print(f"[1] V3预测: {len(pred):,}行, {pred['trade_date'].nunique()}月, {pred['ts_code'].nunique()}只")

# ============ 2. s123 信号 ============
print("[2] s123 信号...")
pe = fetch_pe_csi300()
bond = fetch_bond10y()
close_ix = pe["close"]
dd_ix = close_ix / close_ix.cummax() - 1.0
erp = 1.0 / pe["pe_ttm"] - bond["y10"].reindex(pe.index).ffill()

# 逐月 (月末) 计算 s123
pred_dates = sorted(pred["trade_date"].unique())
ym_list = sorted(set(d // 100 for d in pred_dates))
sig_map = {}
for ym in ym_list:
    d = pd.Timestamp(f"{ym}01") + pd.offsets.MonthEnd(0)
    s1 = 1 if _rolling_pct(pe["pe_ttm"], d) < 0.20 else 0
    s2 = 1 if _zscore(erp, d) > 1.0 else 0
    s3 = 1 if float(dd_ix.asof(d)) <= -0.25 else 0
    sig_map[ym] = s1 + s2 + s3
print("    s123 逐月 (2023-2025):")
for ym in ym_list:
    print(f"      {ym}: s123={sig_map[ym]}  {'【满仓≥3】' if sig_map[ym]>=3 else ('【清仓≤1】' if sig_map[ym]<=1 else '【持有/观望】')}")

# ============ 3. V8 避险日收益 ============
print("[3] V8 避险...")
v8 = load_hv_daily()
all_dates = sorted(set().union(*[set(s.index) for s in v8.values()]))
v8_df = pd.DataFrame(index=all_dates)
for code, s in v8.items():
    v8_df[code] = s.reindex(all_dates)
weights = pd.Series({"511990.SH": 1/3, "511260.SH": 1/3, "518880.SH": 1/3})
v8_daily = (v8_df * weights).sum(axis=1).fillna(0.0)
v8_daily.index = pd.to_datetime(v8_daily.index, format="%Y%m%d")

# ============ 4. 日频价格 ============
print("[4] 加载日频价格...")
codes_bt = set(pred["ts_code"].unique())
px_parts = []
for f in sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet"))):
    if os.path.getsize(f) <= 1024:
        continue
    d = os.path.basename(f)[:8]
    if d < "20221201":
        continue
    ddf = pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
    ddf = ddf[ddf["ts_code"].isin(codes_bt)]
    if len(ddf):
        px_parts.append(ddf)
px = pd.concat(px_parts, ignore_index=True)
px["trade_date"] = pd.to_datetime(px["trade_date"].astype(str), format="%Y%m%d")
px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
CL_MAP = {}
for code, gdf in px.groupby("ts_code"):
    CL_MAP[code] = (gdf["trade_date"].tolist(), gdf["close"].values)
ALL_DAYS = sorted(px["trade_date"].unique())
print(f"    价格: {len(px):,}行, {len(CL_MAP)}只, {len(ALL_DAYS)}个交易日, 耗时{time.time()-t0:.0f}s")

def get_close(code, dt):
    if code not in CL_MAP:
        return None
    dates, closes = CL_MAP[code]
    lo, hi = 0, len(dates) - 1
    while lo <= hi:
        m = (lo + hi) // 2
        if dates[m] == dt:
            return closes[m]
        if dates[m] < dt:
            lo = m + 1
        else:
            hi = m - 1
    return None

# ============ 5. 回测引擎 ============
TOP_GLOBAL, MAXK = 20, 3
BUY_FEE, SELL_FEE, INIT = 0.0010, 0.0015, 1_000_000
TP, MAX_HOLD = 0.30, 180
START = pd.Timestamp("2023-01-01")

# RB_PICKS: 每月预测 -> 第一个交易日
RB_PICKS = {}
for m_int in sorted(pred["trade_date"].unique()):
    m_dt = pd.to_datetime(str(m_int), format="%Y%m%d")
    sub = pred[pred["trade_date"] == m_int].sort_values("prob", ascending=False)
    picks = sub.groupby("industry", sort=False).head(MAXK).head(TOP_GLOBAL)
    plist = [(r["ts_code"], r["industry"], r["prob"]) for _, r in picks.iterrows()]
    for td in ALL_DAYS:
        if td >= m_dt:
            RB_PICKS[td] = plist
            break

# 月末信号 -> 下月状态: 用上个月月末的 s123 信号
# sig_map 的 key 是 ym=月末所在月份, 信号在 ym 月月末算, ym+1 月生效
def prev_month_ym(day):
    y, m = day.year, day.month
    return (y - 1) * 100 + 12 if m == 1 else y * 100 + (m - 1)

def run_backtest(timing):
    state_in = (not timing)  # 无择时: 始终满仓
    cash = INIT if not timing else 0.0
    reserve = 0.0 if not timing else INIT  # 有择时: 初始全在 V8, 等 s123>=3 进场
    holdings = {}
    nav_series, state_log = [], []
    start_idx = next(i for i, d in enumerate(ALL_DAYS) if d >= START)

    for di in range(start_idx, len(ALL_DAYS)):
        day = ALL_DAYS[di]
        # V8 增值
        reserve *= (1 + v8_daily.get(day, 0.0))

        # 月初判断择时状态 (用上月末 s123 信号)
        is_first_of_month = (di == start_idx) or (ALL_DAYS[di-1].month != day.month)
        if timing and is_first_of_month:
            s = sig_map.get(prev_month_ym(day), 0)
            if not state_in and s >= 3:
                cash += reserve
                reserve = 0.0
                state_in = True
            elif state_in and s <= 1:
                for code in list(holdings.keys()):
                    cnow = get_close(code, day)
                    if cnow is not None and cnow > 0:
                        cash += cnow * holdings[code]["qty"] * (1 - SELL_FEE)
                holdings = {}
                reserve += cash
                cash = 0.0
                state_in = False

        # 在仓: 调仓日买入新标的 (V3 规则)
        if state_in and day in RB_PICKS:
            picks = RB_PICKS[day]
            held_inds = set(h["industry"] for h in holdings.values())
            new_picks = [(c, ind, p) for c, ind, p in picks
                         if ind not in held_inds and c not in holdings]
            if new_picks and cash > 10000:
                per = cash / len(new_picks)
                for code, ind, prob in new_picks:
                    bp = get_close(code, day)
                    if bp is None or bp <= 0:
                        continue
                    qty = int((per * (1 - BUY_FEE)) / (bp * 100)) * 100
                    if qty <= 0:
                        continue
                    cost = qty * bp * (1 + BUY_FEE)
                    if cost > cash:
                        continue
                    cash -= cost
                    holdings[code] = {"buy_price": bp, "qty": qty,
                                      "buy_day_idx": di, "industry": ind}

        # 在仓: 每日止盈/时间止损 (V3 规则)
        if state_in:
            for code in list(holdings.keys()):
                h = holdings[code]
                cnow = get_close(code, day)
                if cnow is None:
                    continue
                ret = cnow / h["buy_price"] - 1
                held = di - h["buy_day_idx"]
                if ret >= TP or held >= MAX_HOLD:
                    cash += cnow * h["qty"] * (1 - SELL_FEE)
                    del holdings[code]

        # NAV
        total = cash + reserve
        for code, h in holdings.items():
            cnow = get_close(code, day) or h["buy_price"]
            total += cnow * h["qty"]
        nav_series.append((day, total))
        state_log.append((day, state_in))

    nav = pd.Series(dict(nav_series)).sort_index()
    return nav, state_log

# ============ 6. 跑两版 + 指标 ============
def metrics(nav, tag):
    tr = nav.iloc[-1] / INIT - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = (1 + tr) ** (1 / years) - 1
    peak = nav.cummax(); mdd = ((nav - peak) / peak).min()
    rets = nav.pct_change().dropna()
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if rets.std(ddof=1) > 0 else np.nan
    calmar = ann / (-mdd + 1e-9)
    print(f"\n{'='*60}\n[结果] {tag}\n{'='*60}")
    print(f"  累计 {tr:.1%} | 年化 {ann:.1%} | 回撤 {mdd:.1%} | 夏普 {sharpe:.2f} | Calmar {calmar:.2f}")
    print("  --- 分年度 ---")
    yr = nav.resample("Y").last()
    prev = INIT
    for y, v in yr.items():
        print(f"    {y.year}: {v/prev-1:+.1%}  (期末{v/1e4:.0f}万)")
        prev = v
    return {"tag": tag, "累计": tr, "年化": ann, "回撤": mdd, "夏普": sharpe, "Calmar": calmar}

nav_base, _ = run_backtest(timing=False)
nav_s123, log_s123 = run_backtest(timing=True)

m_base = metrics(nav_base, "V3 纯个股 (始终满仓)")
m_s123 = metrics(nav_s123, "V3 + s123择时 + V8避险")

# 仓位状态统计
log = pd.DataFrame(log_s123, columns=["day", "state_in"])
in_pct = log["state_in"].mean()
print(f"\ns123 整合: 持仓天数占比 {in_pct:.1%}")

# 保存
nav_base.to_csv(os.path.join(OUT_DIR, "fullmarket_v3_s123_base_nav.csv"), header=True)
nav_s123.to_csv(os.path.join(OUT_DIR, "fullmarket_v3_s123_nav.csv"), header=True)
pd.DataFrame([m_base, m_s123]).to_csv(os.path.join(OUT_DIR, "fullmarket_v3_s123_compare.csv"),
                                       index=False, encoding="utf-8-sig")
print(f"\n总耗时 {time.time()-t0:.0f}s")
