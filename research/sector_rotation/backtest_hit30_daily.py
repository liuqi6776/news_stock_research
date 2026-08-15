# -*- coding: utf-8 -*-
"""
STEP3: 日频逐交易日回测 (修正之前只在调仓日检查止盈的疏漏)

- 每月调仓日: 低估池内模型Top6 → 等权买入(如现金足够, 只买没仓位的)
- 每个交易日: 检查所有持仓是否 hit30% → 止盈; 是否到120交易日 → 强平
- 输出: 净值曲线 + 交易明细 + 与原板块版对比
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
# 读OOS预测
PRED_CSV = os.path.join(OUT_DIR, "hit30_oos_predictions.csv")
df_pred = pd.read_csv(PRED_CSV)
df_pred["trade_date_dt"] = pd.to_datetime(df_pred["trade_date"].astype(str), format="%Y%m%d")
codes_need = set(df_pred["ts_code"].unique())

# 调仓日 (预测里的trade_date) → 下一交易日开盘买入
rebalance_days = sorted(df_pred["trade_date_dt"].unique())
print(f"调仓日数: {len(rebalance_days)} | 候选股票池: {len(codes_need)}只")

# 读行情: 全量日频
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
# 构建 (ts_code, trade_date) -> close的dict/df
px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
# close lookup map: code -> 日期索引->价格
CL_MAP = {}
for code, gdf in px.groupby("ts_code"):
    dates = gdf["trade_date"].tolist()
    closes = gdf["close"].values
    CL_MAP[code] = (dates, closes)
ALL_DAYS = sorted(px["trade_date"].unique())
DAY2IDX = {d:i for i,d in enumerate(ALL_DAYS)}
print(f"日频行情: {len(ALL_DAYS)}个交易日, 从 {ALL_DAYS[0].date()} 到 {ALL_DAYS[-1].date()}")

# 回测参数
BUY_FEE = 0.0010
SELL_FEE = 0.0015
INIT = 1_000_000
TOPN = 6
MAX_HOLD_DAYS = 120  # 对应标签 T+100内hit30, 给120天宽限期
TP = 0.30

cash = INIT
holdings = {}  # code -> {buy_price, qty, buy_day_idx, buy_cost}
nav_series = []
trades = []

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

# 调仓日 -> TopN买入清单
RB_MAP = {}
for rd in rebalance_days:
    picks = df_pred[df_pred["trade_date_dt"] == rd].sort_values("pred", ascending=False).head(TOPN)["ts_code"].tolist()
    RB_MAP[rd] = picks

# ---------- 日频循环 ----------
print("开始日频回测...")
for di, day in enumerate(ALL_DAYS):
    # 1. 先处理调仓日的买入 (开盘用当日close近似)
    if day in RB_MAP:
        picks = RB_MAP[day]
        to_buy = [c for c in picks if c not in holdings]
        if to_buy and cash > 1000:
            per = cash / len(to_buy)
            for code in to_buy:
                bp = get_close(code, day)
                if bp is None or bp <= 0: continue
                qty = int((per * (1 - BUY_FEE)) / (bp * 100)) * 100
                if qty <= 0: continue
                cost = qty * bp * (1 + BUY_FEE)
                cash -= cost
                holdings[code] = {"buy_price": bp, "qty": qty, "buy_day_idx": di, "buy_cost": cost}
                trades.append((day, "BUY", code, cost, np.nan, 0))

    # 2. 检查所有持仓的止盈/强平 (用当日close)
    for code in list(holdings.keys()):
        h = holdings[code]
        cnow = get_close(code, day)
        if cnow is None: continue
        ret = cnow / h["buy_price"] - 1
        held = di - h["buy_day_idx"]
        if ret >= TP or held >= MAX_HOLD_DAYS:
            proceeds = cnow * h["qty"] * (1 - SELL_FEE)
            cash += proceeds
            op = "TP" if ret >= TP else ("SL60" if held >= MAX_HOLD_DAYS else "SL")
            trades.append((day, op, code, proceeds, ret, held))
            del holdings[code]

    # 3. 收盘算净值
    total = cash
    for code, h in holdings.items():
        cnow = get_close(code, day)
        if cnow is None:
            # 找最近有数据的前一天
            cnow = h["buy_price"]
        total += cnow * h["qty"]
    nav_series.append(total)

nav_s = pd.Series(nav_series, index=ALL_DAYS, name="nav")
tr = nav_s.iloc[-1] / INIT - 1
years = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
ann = (1 + tr) ** (1/years) - 1 if years > 0 else np.nan
peak = nav_s.cummax()
mdd = ((nav_s - peak) / peak).min()
rets = nav_s.pct_change().dropna()
sharpe = np.nan
if len(rets) > 1 and rets.std(ddof=1) > 0:
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252)
buys = sum(1 for t in trades if t[1]=="BUY")
tps = sum(1 for t in trades if t[1]=="TP")
sls = sum(1 for t in trades if t[1] not in ("BUY","TP"))
held_days = [t[5] for t in trades if t[1] != "BUY"]
avg_held = np.mean(held_days) if held_days else 0

print(f"\n========== [日频回测结果 OOS 2023-2026] 低估池Top{TOPN} 30%止盈/{MAX_HOLD_DAYS}天强平 ==========")
print(f"  时间: {nav_s.index[0].date()} → {nav_s.index[-1].date()} ({years:.1f}年)")
print(f"  期初: {INIT/1e4:.0f}万  → 期末: {nav_s.iloc[-1]/1e4:.1f}万")
print(f"  累计收益: {tr:.1%} | 年化: {ann:.1%}")
print(f"  最大回撤: {mdd:.1%} | 夏普(日): {sharpe:.2f}")
print(f"  买入{buys}笔 | 止盈{tps}笔 | 强平{sls}笔 | 止盈胜率={tps/(tps+sls+1e-9):.1%} | 平均持仓{avg_held:.0f}天")

# 对比: 原板块版 (行业个股等权+30%止盈) 给同时间段OOS 2023-2026
print("\n========== 对比 [原板块版OOS 2023-2026] (行业等权, 无选股) ==========")
# 从行业收益面板直接跑2023-2026
RET_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "industry_ret.csv")
PE_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "industry_pe.csv")
ret_df = pd.read_csv(RET_CSV, index_col=0); ret_df.index = ret_df.index.astype(str)
pe_df = pd.read_csv(PE_CSV, index_col=0); pe_df.index = pe_df.index.astype(str)
pe_pct = pe_df.rolling(60, min_periods=12).rank(pct=True)
# 只取OOS月份
oos_yymm = [d for d in ret_df.index if d >= '202301']
HIGH_WIN = [
    '新型电力','供气供热','煤炭开采','化工机械','铝','服饰','航空','船舶','水运',
    '焦炭加工','化纤','机械基件','铜','酒店餐饮','农业综合','中成药','铅锌',
    '环境保护','汽车服务','矿物制品','汽车整车','汽车配件','摩托车','专用机械',
    '钢加工','轻工机械','橡胶','建筑工程','化工原料','医药商业','小金属',
    '纺织机械','出版业','其他商业','农药化肥','IT设备','红黄酒','纺织','陶瓷',
    '火力发电','石油开采','商贸代理','家用电器','电气设备','工程机械','元器件',
    '半导体','软件服务','互联网','通信设备','石油加工',
]
cash2 = INIT
holdings2 = {}
for d in oos_yymm:
    if cash2 > 1000:
        candidates = []
        for ind in HIGH_WIN:
            if ind not in ret_df.columns or ind in holdings2: continue
            if d not in pe_pct.index: continue
            p = pe_pct.loc[d, ind]
            if pd.notna(p) and p < 0.30: candidates.append(ind)
        if candidates:
            per = cash2 / len(candidates)
            for ind in candidates:
                amt = per * 0.999  # 双边0.1%简化
                cash2 -= per
                holdings2[ind] = {"cost": per, "value": amt}
    total2 = cash2
    for ind in list(holdings2.keys()):
        r = ret_df.loc[d, ind] if (ind in ret_df.columns and d in ret_df.index) else 0
        if pd.isna(r): r = 0
        holdings2[ind]["value"] *= (1 + r)
        cum = holdings2[ind]["value"] / holdings2[ind]["cost"] - 1
        if cum >= 0.30:
            cash2 += holdings2[ind]["value"] * 0.9985
            del holdings2[ind]
        else:
            total2 += holdings2[ind]["value"]
    # 月度净值
    pass  # 只看期末
# 最后加剩余
for ind in holdings2: total2 += holdings2[ind]["value"]
tr2 = total2 / INIT - 1
print(f"  期末: {total2/1e4:.1f}万 | 累计: {tr2:.1%} | 年化: {(1+tr2)**(1/years)-1:.1%}")

# 保存
nav_s.to_csv(os.path.join(OUT_DIR, "hit30_stock_nav_daily.csv"), header=True)
pd.DataFrame(trades, columns=["date","op","code","amount","ret","days"]
            ).to_csv(os.path.join(OUT_DIR, "hit30_stock_trades_daily.csv"), index=False, encoding="utf-8-sig")
print(f"\n耗时 {time.time()-t0:.0f}s")
