# -*- coding: utf-8 -*-
"""
回测v2: 板块之上的统一ML (用已保存的 universe_safe_hit30_oos_pred.csv)
修复:
  1. 回测窗口 = OOS期 (2023-01~2026-08), 避免前3年空仓稀释年化
  2. 卖出 = 30%止盈 OR 持仓180天时间止损 (匹配标签100天窗口, 防止套牢无限期)
  3. 每个调仓日重新审视全局Top20, 只买未持仓且未持仓行业的新票
"""
import os, glob, time
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
t0 = time.time()

PRED_CSV = os.path.join(OUT_DIR, "universe_safe_hit30_oos_pred.csv")
df_pred = pd.read_csv(PRED_CSV)
df_pred["trade_date_dt"] = pd.to_datetime(df_pred["trade_date"].astype(str), format="%Y%m%d")
codes_need = set(df_pred["ts_code"].unique())

# 读行情 (2022-12起, 覆盖2023开始的回测)
px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
parts = []
for f in px_files:
    if os.path.getsize(f) <= 1024: continue
    d = os.path.basename(f)[:8]
    if d < "20221201": continue
    df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
    df = df[df["ts_code"].isin(codes_need)]
    if len(df): parts.append(df)
px = pd.concat(parts, ignore_index=True)
px["trade_date"] = pd.to_datetime(px["trade_date"].astype(str), format="%Y%m%d")
px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
CL_MAP = {}
for code, gdf in px.groupby("ts_code"):
    CL_MAP[code] = (gdf["trade_date"].tolist(), gdf["close"].values)
ALL_DAYS = sorted(px["trade_date"].unique())
print(f"交易日: {len(ALL_DAYS)} ({ALL_DAYS[0].date()} ~ {ALL_DAYS[-1].date()})")

def get_close(code, dt):
    if code not in CL_MAP: return None
    dates, closes = CL_MAP[code]
    lo, hi = 0, len(dates)-1
    while lo <= hi:
        m = (lo + hi) // 2
        if dates[m] == dt: return closes[m]
        if dates[m] < dt: lo = m + 1
        else: hi = m - 1
    return None

# ---------- 每月调仓: 全市场打分 Top20 (每行业≤3) ----------
TOP_GLOBAL, MAXK = 20, 3
RB_PICKS = {}
for m_int in sorted(df_pred["trade_date"].unique()):
    m_dt = pd.to_datetime(str(m_int), format="%Y%m%d")
    subset = df_pred[df_pred["trade_date"] == m_int].sort_values("prob", ascending=False)
    picks = subset.groupby("industry", sort=False).head(MAXK).head(TOP_GLOBAL)
    pick_list = [(r["ts_code"], r["industry"], r["prob"]) for _, r in picks.iterrows()]
    for td in ALL_DAYS:
        if td >= m_dt:
            RB_PICKS[td] = pick_list
            break

# ---------- 参数 ----------
BUY_FEE = 0.0010; SELL_FEE = 0.0015; INIT = 1_000_000
TP = 0.30
MAX_HOLD_DAYS = 180   # 时间止损, 匹配标签100天窗口+缓冲
START = pd.Timestamp("2023-01-01")

cash = INIT
holdings = {}
nav_series = []; trades = []
start_idx = None
for i, d in enumerate(ALL_DAYS):
    if d >= START:
        start_idx = i; break

for di in range(start_idx, len(ALL_DAYS)):
    day = ALL_DAYS[di]
    # 1. 调仓买入: 只买未持仓行业的全局Top
    if day in RB_PICKS:
        picks = RB_PICKS[day]
        held_inds = set(h["industry"] for h in holdings.values())
        new_picks = [(c, ind, p) for c, ind, p in picks
                     if ind not in held_inds and c not in holdings]
        if new_picks and cash > 10000:
            per = cash / len(new_picks)
            for code, ind, prob in new_picks:
                bp = get_close(code, day)
                if bp is None or bp <= 0: continue
                qty = int((per * (1 - BUY_FEE)) / (bp * 100)) * 100
                if qty <= 0: continue
                cost = qty * bp * (1 + BUY_FEE)
                cash -= cost
                holdings[code] = {"buy_price": bp, "qty": qty,
                                  "buy_day_idx": di, "industry": ind}
                trades.append((day, "BUY", code, cost, np.nan, 0, ind))
    # 2. 卖出: 30%止盈 OR 180天时间止损
    for code in list(holdings.keys()):
        h = holdings[code]
        cnow = get_close(code, day)
        if cnow is None: continue
        ret = cnow / h["buy_price"] - 1
        held = di - h["buy_day_idx"]
        if ret >= TP or held >= MAX_HOLD_DAYS:
            proceeds = cnow * h["qty"] * (1 - SELL_FEE)
            cash += proceeds
            op = "TP" if ret >= TP else "T180"
            trades.append((day, op, code, proceeds, ret, held, h["industry"]))
            del holdings[code]
    # 3. 净值
    total = cash
    for code, h in holdings.items():
        cnow = get_close(code, day) or h["buy_price"]
        total += cnow * h["qty"]
    nav_series.append((day, total))

nav_s = pd.Series(dict(nav_series)).sort_index()
tr = nav_s.iloc[-1] / INIT - 1
years = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
ann = (1 + tr) ** (1/years) - 1 if years > 0 else np.nan
peak = nav_s.cummax(); mdd = ((nav_s - peak) / peak).min()
rets = nav_s.pct_change().dropna()
sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
buys = sum(1 for t in trades if t[1]=="BUY")
tps = sum(1 for t in trades if t[1]=="TP")
t180 = sum(1 for t in trades if t[1]=="T180")
held_days = [t[5] for t in trades if t[1] != "BUY"]

print(f"\n========== [回测v2] 板块之上统一ML OOS 2023-2026 ==========")
print(f"  时间: {nav_s.index[0].date()} → {nav_s.index[-1].date()} ({years:.1f}年)")
print(f"  期初: {INIT/1e4:.0f}万 → 期末: {nav_s.iloc[-1]/1e4:.1f}万")
print(f"  累计: {tr:.1%} | 年化: {ann:.1%}")
print(f"  回撤: {mdd:.1%} | 夏普: {sharpe:.2f}")
print(f"  买入{buys}笔 | 30%止盈{tps}笔({tps/(buys+1e-9):.0%}) | 180天止损{t180}笔 | 平均持仓{np.mean(held_days):.0f}天")
print(f"  参数: 全局Top{TOP_GLOBAL}(每行业≤{MAXK}), 30%止盈, 180天时间止损")

# 月度净值表
nav_m = nav_s.resample("M").last()
print("\n--- 年度净值 ---")
print(nav_m.groupby(nav_m.index.year).last().apply(lambda x: f"{(x/INIT-1):.1%}"))

nav_s.to_csv(os.path.join(OUT_DIR, "universe_safe_hit30_nav_v2.csv"), header=True)
pd.DataFrame(trades, columns=["date","op","code","amount","ret","days","industry"]
            ).to_csv(os.path.join(OUT_DIR, "universe_safe_hit30_trades_v2.csv"), index=False, encoding="utf-8-sig")
print(f"\n耗时 {time.time()-t0:.0f}s")
