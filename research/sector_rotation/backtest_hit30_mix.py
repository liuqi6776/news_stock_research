# -*- coding: utf-8 -*-
"""
STEP4: 正确的混合架构 = 低估板块分散 + 板块内ML选股Top6
  - 每月: 找PE分位<30%的低估行业集合 S = {ind1, ind2, ...}
  - 对每个 ind ∈ S: 该行业当月低估池内股票按pred排序取Top6
  - 等权分配: 每个行业分配 总现金/|S| → 行业内再分给Top6只
  - 日频: 单票30%止盈 / 120天强平
  → 保留原板块版分散优势, 用ML做行业内精选降低持仓数(省手续费)
"""
import os
import glob
import time
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")

t0 = time.time()
PRED_CSV = os.path.join(OUT_DIR, "hit30_oos_predictions.csv")
df_pred = pd.read_csv(PRED_CSV)
df_pred["trade_date_dt"] = pd.to_datetime(df_pred["trade_date"].astype(str), format="%Y%m%d")
codes_need = set(df_pred["ts_code"].unique())
rebalance_days = sorted(df_pred["trade_date_dt"].unique())

# 读行情
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
    dates = gdf["trade_date"].tolist()
    closes = gdf["close"].values
    CL_MAP[code] = (dates, closes)
ALL_DAYS = sorted(px["trade_date"].unique())
print(f"交易日: {len(ALL_DAYS)} | 候选: {len(codes_need)}只")

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

# ---------- 调仓日: 低估行业集合 S + S内各行业Top6股票清单 ----------
# 先取S (来自预测面板的低估行业, 因为df_pred本身就是低估池)
RB_PICKS = {}  # day -> list of (code, industry, pred)
for rd in rebalance_days:
    subset = df_pred[df_pred["trade_date_dt"] == rd].copy()
    # 低估行业集合 = subset里出现过的industry去重
    S_list = sorted(subset["industry"].dropna().unique())
    # 对每个行业取Top6 (如果不够6只, 就全取)
    picks = []
    for ind in S_list:
        ind_df = subset[subset["industry"] == ind].sort_values("pred", ascending=False).head(6)
        for _, row in ind_df.iterrows():
            picks.append((row["ts_code"], row["industry"], row["pred"]))
    RB_PICKS[rd] = picks
    if rd == rebalance_days[3]:
        print(f"  样例 {rd.date()}: 低估行业{len(S_list)}个, 选{len(picks)}只")

# ---------- 日频回测 ----------
BUY_FEE = 0.0010; SELL_FEE = 0.0015; INIT = 1_000_000; TP = 0.30; MAX_HOLD = 120
cash = INIT
holdings = {}  # code -> {buy_price, qty, buy_day_idx, buy_cost, industry}
nav_series = []; trades = []
for di, day in enumerate(ALL_DAYS):
    # 调仓日买入
    if day in RB_PICKS:
        picks = RB_PICKS[day]
        # 分行业等权: 先算行业集合, 每个行业一份
        inds = list(set(ind for _, ind, _ in picks))
        if inds and cash > 1000:
            per_ind = cash / len(inds)
            # 每个行业内Top6平分
            for ind in inds:
                ind_codes = [c for c, i, _ in picks if i == ind and c not in holdings]
                if not ind_codes: continue
                per_code = per_ind / len(ind_codes)
                for code in ind_codes:
                    bp = get_close(code, day)
                    if bp is None or bp <= 0: continue
                    qty = int((per_code * (1 - BUY_FEE)) / (bp * 100)) * 100
                    if qty <= 0: continue
                    cost = qty * bp * (1 + BUY_FEE)
                    cash -= cost
                    holdings[code] = {"buy_price": bp, "qty": qty, "buy_day_idx": di,
                                      "buy_cost": cost, "industry": ind}
                    trades.append((day, "BUY", code, cost, np.nan, 0, ind))
    # 止盈 / 强平
    for code in list(holdings.keys()):
        h = holdings[code]
        cnow = get_close(code, day)
        if cnow is None: continue
        ret = cnow / h["buy_price"] - 1
        held = di - h["buy_day_idx"]
        if ret >= TP or held >= MAX_HOLD:
            proceeds = cnow * h["qty"] * (1 - SELL_FEE)
            cash += proceeds
            op = "TP" if ret >= TP else "SL120"
            trades.append((day, op, code, proceeds, ret, held, h["industry"]))
            del holdings[code]
    # 净值
    total = cash
    for code, h in holdings.items():
        cnow = get_close(code, day) or h["buy_price"]
        total += cnow * h["qty"]
    nav_series.append(total)

nav_s = pd.Series(nav_series, index=ALL_DAYS, name="nav")
tr = nav_s.iloc[-1] / INIT - 1
years = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
ann = (1 + tr) ** (1/years) - 1 if years > 0 else np.nan
peak = nav_s.cummax()
mdd = ((nav_s - peak) / peak).min()
rets = nav_s.pct_change().dropna()
sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
buys = sum(1 for t in trades if t[1]=="BUY"); tps = sum(1 for t in trades if t[1]=="TP")
sls = sum(1 for t in trades if t[1] not in ("BUY","TP"))
held_days = [t[5] for t in trades if t[1] != "BUY"]

print(f"\n========== [混合架构 OOS 2023-2026] 低估行业分散 + 行业内ML选Top6 ==========")
print(f"  时间: {nav_s.index[0].date()} → {nav_s.index[-1].date()} ({years:.1f}年)")
print(f"  期初: {INIT/1e4:.0f}万  → 期末: {nav_s.iloc[-1]/1e4:.1f}万")
print(f"  累计收益: {tr:.1%} | 年化: {ann:.1%}")
print(f"  最大回撤: {mdd:.1%} | 夏普(日): {sharpe:.2f}")
print(f"  买入{buys}笔 | 止盈{tps}笔 | 强平{sls}笔 | 止盈胜率={tps/(tps+sls+1e-9):.1%} | 平均持仓{np.mean(held_days):.0f}天")
print(f"\n--- 同时期基准对比 ---")
print(f"  ·原板块版(行业等权无选股) 同期≈219%累计/年化37% (分散全部行业股票)")
print(f"  ·全市场低估Top6乱选版 同期≈-39% (集中高波动)")
print(f"  ·混合架构: 行业分散 + ML精选 = 你的选股程序未来接进来的正确位置")

# 保存
nav_s.to_csv(os.path.join(OUT_DIR, "hit30_mix_sector_ml_nav.csv"), header=True)
tcols = ["date","op","code","amount","ret","days","industry"]
pd.DataFrame(trades, columns=tcols).to_csv(os.path.join(OUT_DIR, "hit30_mix_sector_ml_trades.csv"),
                                            index=False, encoding="utf-8-sig")
print(f"\n耗时 {time.time()-t0:.0f}s")
