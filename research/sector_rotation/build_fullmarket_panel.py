# -*- coding: utf-8 -*-
"""
构建全市场个股因子面板 (中证1000 → 全市场 5540 只)

与 stock_ml_feature_pipeline.py 的差异:
  1. 去除 index_weight 000852 成分过滤 → 全市场
  2. 行业映射改用 industry1/industry.parquet (110 个细分行业)
  3. 加流动性/停牌过滤 (60日均成交额<300万 剔除; 60日内有效交易日<20 剔除)
  4. 同时算 fwd_20 + fwd100_maxret + fwd100_minret (供 safe_hit30 标签)

输出: stock_ml_panel_fullmarket_72m.parquet
列: trade_date, ts_code, industry, is_traditional,
    ret_1m, ivol, momentum_5/10/20/60, volatility_5/10/20,
    alpha_006/009/012/023, roe, or_yoy, netprofit_yoy,
    vwap_20, float_pnl_20, prof_pct_20, chip_conc_20, chip_shift_5, pos_vol_20,
    fwd_20, fwd100_maxret, fwd100_minret
"""
import os, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
OUT = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fullmarket_2015.parquet")

# 流动性过滤阈值: 60日均成交额 < 300万(元) = 3000(千元) 剔除
MIN_AMOUNT_MA60 = 3000.0
MIN_VALID_60 = 20

# ---------- 1. 行业映射 (industry1, 110行业) ----------
ind = pd.read_parquet(os.path.join(DATA, "industry1", "industry.parquet"))
ind_map = dict(zip(ind["ts_code"], ind["industry"]))
print(f"[1] 行业映射: {len(ind_map)} 只, 行业数={ind['industry'].nunique()}")

# ---------- 2. 全市场日频行情 ----------
px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
parts = []
for f in px_files:
    if os.path.getsize(f) <= 1024:
        continue
    d = os.path.basename(f)[:8]
    if d < "20150101":  # 从2015年起, 留60日预热
        continue
    df = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low",
                                     "close", "pct_chg", "vol", "amount"])
    if len(df):
        parts.append(df)
px = pd.concat(parts, ignore_index=True)
px["trade_date"] = px["trade_date"].astype(int)
px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
print(f"[2] 全市场行情: {len(px):,} 行, {px['ts_code'].nunique()} 只, "
      f"{px['trade_date'].min()}~{px['trade_date'].max()}, {time.time()-t0:.0f}s")

# ---------- 3. 因子计算 ----------
px["r"] = px["pct_chg"] / 100.0
mkt = px.groupby("trade_date")["r"].mean().rename("mkt_ret")
px = px.merge(mkt, on="trade_date", how="left")
px["ex_ret"] = px["r"] - px["mkt_ret"]
g = px.groupby("ts_code", sort=False)

px["ret_1m"] = g["r"].transform(lambda s: (1 + s).rolling(20).apply(np.prod, raw=True) - 1)
px["ivol"] = g["ex_ret"].transform(lambda s: s.rolling(20).std(ddof=1))
px["cum20"] = g["r"].transform(lambda s: (1 + s).rolling(20).apply(np.prod, raw=True))
px["fwd_20"] = px.groupby("ts_code")["cum20"].shift(-20) - 1
for w in (5, 10, 20, 60):
    px[f"momentum_{w}"] = g["close"].pct_change(w)
for w in (5, 10, 20):
    px[f"volatility_{w}"] = g["r"].transform(lambda s, ww=w: s.rolling(ww).std(ddof=1))
print(f"  [3.1] 价量基础 完成, {time.time()-t0:.0f}s")

# Alpha101 核心4个
px["alpha_006"] = px.groupby("ts_code", group_keys=False).apply(
    lambda x: x["open"].rolling(10).corr(x["vol"])) * -1
dc = px.groupby("ts_code")["close"].diff(1)
min5 = dc.groupby(px["ts_code"]).transform(lambda s: s.rolling(5).min())
max5 = dc.groupby(px["ts_code"]).transform(lambda s: s.rolling(5).max())
px["alpha_009"] = np.where(min5 > 0, dc, np.where(max5 < 0, dc, -dc))
dv = px.groupby("ts_code")["vol"].diff(1)
px["alpha_012"] = np.sign(dv) * (-dc)
hi20 = px.groupby("ts_code")["high"].transform(lambda s: s.rolling(20).mean())
px["alpha_023"] = np.where(hi20 < px["high"], -px.groupby("ts_code")["high"].diff(2), 0.0)
print(f"  [3.2] Alpha101 完成, {time.time()-t0:.0f}s")

# 筹码因子 (chip_conc 用 std/mean 简化, 避免逐股 Python 循环)
px["amt3"] = (px["high"] + px["low"] + px["close"]) / 3.0 * px["vol"]
px["vwap_20"] = g["amt3"].transform(lambda s: s.rolling(20).sum()) / (
    1e-9 + g["vol"].transform(lambda s: s.rolling(20).sum()))
px["float_pnl_20"] = (px["close"] - px["vwap_20"]) / (px["vwap_20"] + 1e-9)
px["prof_pct_20"] = g["close"].transform(lambda s: s.rolling(20).rank(pct=True))
px["vwap_5"] = g["amt3"].transform(lambda s: s.rolling(5).sum()) / (
    1e-9 + g["vol"].transform(lambda s: s.rolling(5).sum()))
px["chip_shift_5"] = (px["vwap_5"] - px["vwap_20"]) / (px["vwap_20"] + 1e-9)
px["is_up"] = (px["pct_chg"] > 0).astype(float)
px["pos_vol_20"] = (px["vol"] * px["is_up"]).groupby(px["ts_code"]).transform(
    lambda s: s.rolling(20).sum()) / (px["vol"].groupby(px["ts_code"]).transform(
    lambda s: s.rolling(20).sum()) + 1e-9)
px["chip_conc_20"] = g["close"].transform(lambda s: s.rolling(20).std(ddof=1)) / (
    g["close"].transform(lambda s: s.rolling(20).mean()) + 1e-9)
px.drop(columns=["amt3", "vwap_5", "is_up"], inplace=True, errors="ignore")
print(f"  [3.3] 筹码因子 完成, {time.time()-t0:.0f}s")

# 流动性/停牌 (60日滚动)
px["amount_ma60"] = g["amount"].transform(lambda s: s.rolling(60).mean())
px["n_valid_60"] = g["close"].transform(lambda s: s.rolling(60).count())

# forward 100 天 max/min (safe_hit30 标签)
cnt100 = g["close"].transform(lambda s: s.rolling(100, min_periods=10).count().shift(-100))
fmax = g["close"].transform(lambda s: s.rolling(100, min_periods=10).max().shift(-100))
fmin = g["close"].transform(lambda s: s.rolling(100, min_periods=10).min().shift(-100))
px["fwd100_maxret"] = np.where(cnt100 >= 10, fmax / px["close"] - 1, np.nan)
px["fwd100_minret"] = np.where(cnt100 >= 10, fmin / px["close"] - 1, np.nan)
print(f"  [3.4] forward100 标签 完成, {time.time()-t0:.0f}s")

# ---------- 4. 月末快照 + 流动性/停牌过滤 ----------
cal = sorted(px["trade_date"].unique())
cal_s = pd.Series(cal)
month_last = cal_s.groupby(cal_s // 100).max().tolist()
month_last = [d for d in month_last if 20150401 <= d <= 20251231]
print(f"[4] 月末快照: {len(month_last)} 月 ({month_last[0]}~{month_last[-1]})")

panel = px[px["trade_date"].isin(month_last)].copy()
n_before = len(panel)
panel = panel[(panel["amount_ma60"] >= MIN_AMOUNT_MA60) & (panel["n_valid_60"] >= MIN_VALID_60)]
print(f"    流动性/停牌过滤: {n_before:,} → {len(panel):,} 股-月 "
      f"(剔除 {1-len(panel)/n_before:.1%})")

# ---------- 5. 财务 PIT (ann_date 对齐, 无前视) ----------
fin = pd.read_parquet(os.path.join(DATA, "fundamental1", "fina_indicator_cache.parquet"))
fin = fin[["ts_code", "ann_date", "roe", "or_yoy", "netprofit_yoy"]].dropna(subset=["ann_date"])
fin["ann_date"] = fin["ann_date"].astype(str).str.replace("-", "").str[:8].astype(int)
fin = fin.sort_values("ann_date")
panel = panel.sort_values("trade_date")
panel = pd.merge_asof(panel, fin, left_on="trade_date", right_on="ann_date",
                      by="ts_code", direction="backward")
print(f"    PIT财务: roe非空 {panel['roe'].notna().sum():,}/{len(panel)} "
      f"({panel['roe'].notna().mean():.1%}), {time.time()-t0:.0f}s")

# ---------- 6. 行业 + is_traditional ----------
panel["industry"] = panel["ts_code"].map(ind_map).fillna("其他")
TECH_KEYWORDS = [
    "半导体", "元器件", "IT设备", "计算机设备", "软件服务", "IT服务", "软件开发",
    "互联网", "通信设备", "通信服务", "游戏", "数字媒体", "广告营销", "影视院线",
    "出版业", "电视广播", "光学光电子", "消费电子", "其他电子", "电子化学品",
    "电池", "电机", "风电设备", "光伏设备", "电源设备", "电网设备",
    "航天装备", "航空装备", "地面兵装", "船舶装备", "军工电子", "航海装备",
]
panel["is_traditional"] = (~panel["industry"].isin(TECH_KEYWORDS)).astype(int)
print(f"    传统行业占比: {panel['is_traditional'].mean()*100:.1f}%")

# ---------- 7. 输出 ----------
feat_cols = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
             "volatility_5", "volatility_10", "volatility_20",
             "alpha_006", "alpha_009", "alpha_012", "alpha_023",
             "roe", "or_yoy", "netprofit_yoy",
             "vwap_20", "float_pnl_20", "prof_pct_20", "chip_conc_20", "chip_shift_5", "pos_vol_20"]
out_cols = ["trade_date", "ts_code", "industry", "is_traditional"] + feat_cols + \
           ["fwd_20", "fwd100_maxret", "fwd100_minret"]
panel = panel[out_cols].copy()
panel["fwd_20"] = panel["fwd_20"] * 100  # %
panel = panel.dropna(subset=["fwd100_maxret"])  # 去掉未来数据不足的尾部
panel.to_parquet(OUT, index=False)
print(f"[7] 保存: {OUT}")
print(f"    面板: {len(panel):,} 股-月, 月份={panel['trade_date'].nunique()}, "
      f"股票={panel['ts_code'].nunique()}, 传统占比={panel['is_traditional'].mean():.1%}")
print(f"    safe_hit30候选(涨幅>=30%且回撤>=-20%)命中率: "
      f"{((panel['fwd100_maxret']>=0.30)&(panel['fwd100_minret']>=-0.20)).mean():.1%}")
print(f"总耗时 {time.time()-t0:.0f}s")
