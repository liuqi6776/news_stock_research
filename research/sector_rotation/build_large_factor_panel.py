# -*- coding: utf-8 -*-
"""构建大因子面板: 从 daily 行情计算 80+ 因子 → 月末快照 → parquet

因子分 8 大类:
  1. 价量基础 (10): ret_1m, ivol, momentum_3/5/10/20/60/120, volatility_5/10/20
  2. Alpha101 扩展 (14): alpha_001/002/004/005/006/009/010/012/017/023/033/041/054/060
  3. 技术指标 (18): RSI_6/14, KDJ_K/D/J, MACD_dif/dea/hist, BOLL_width/pctb,
                    CCI_14, Williams_R, ROC_5/10, ATR_14, price_ma_ratio_5/20/60, ma_cross_5_20
  4. 滚动统计 (16): return_mean_5/10/20, return_std_5/10/20, return_skew_20, return_kurt_20,
                    high_max_5/10/20, low_min_5/10/20, volume_mean_5/10/20, volume_std_10, volume_ratio
  5. 筹码因子 (6): vwap_20, float_pnl_20, prof_pct_20, chip_conc_20, chip_shift_5, pos_vol_20
  6. 资金流近似 (5): net_mf_approx, mf_5d_sum, lg_buy_ratio, mf_signal, mf_reversal
  7. 滞后收益 (4): lag_return_1/3/5/10
  8. 截面排名 (5): rank_return, rank_volatility, rank_volume_ratio, rank_momentum, rank_rsi

输出: stock_ml_panel_large_72m.parquet
"""
import os, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
OUT = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_large_72m.parquet")

# ---------- 1. 中证1000 成分 ----------
iw_files = sorted(glob.glob(os.path.join(DATA, "index_weight", "*.parquet")))
iw = pd.concat([pd.read_parquet(f) for f in iw_files], ignore_index=True)
iw = iw[iw["index_code"] == "000852.SH"].copy()
iw["iw_date"] = iw["trade_date"].astype(int)
iw = iw[["iw_date", "con_code"]].drop_duplicates()
iw_dates = sorted(iw["iw_date"].unique())
member_codes = set(iw["con_code"].unique())
print(f"[1] 成分快照: {len(iw_dates)} 期, {len(member_codes)} 只历史成分")

# ---------- 2. 行情 ----------
px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
parts = []
for f in px_files:
    if os.path.getsize(f) <= 1024: continue
    d = os.path.basename(f)[:8]
    if d < "20190601": continue  # 120日预热
    df = pd.read_parquet(f, columns=["ts_code","trade_date","open","high","low","close","pct_chg","vol"])
    df = df[df["ts_code"].isin(member_codes)]
    if len(df): parts.append(df)
px = pd.concat(parts, ignore_index=True)
px["trade_date"] = px["trade_date"].astype(int)
px = px.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
print(f"[2] 行情: {len(px):,} 行, {px['ts_code'].nunique()} 只, {px['trade_date'].min()}~{px['trade_date'].max()}, {time.time()-t0:.0f}s")

# ---------- 3. 因子计算 ----------
px["r"] = px["pct_chg"] / 100.0
mkt = px.groupby("trade_date")["r"].mean().rename("mkt_ret")
px = px.merge(mkt, on="trade_date", how="left")
px["ex_ret"] = px["r"] - px["mkt_ret"]
g = px.groupby("ts_code", sort=False)

# === 3.1 价量基础 (10) ===
px["ret_1m"] = g["r"].transform(lambda s: (1+s).rolling(20).apply(np.prod, raw=True)-1)
px["ivol"]   = g["ex_ret"].transform(lambda s: s.rolling(20).std(ddof=1))
px["cum20"]  = g["r"].transform(lambda s: (1+s).rolling(20).apply(np.prod, raw=True))
px["fwd_20"] = px.groupby("ts_code")["cum20"].shift(-20) - 1
for w in (3,5,10,20,60,120):
    px[f"momentum_{w}"] = g["close"].pct_change(w)
for w in (5,10,20):
    px[f"volatility_{w}"] = g["r"].transform(lambda s,w=w: s.rolling(w).std(ddof=1))
print(f"  [3.1] 价量基础 完成, {time.time()-t0:.0f}s")

# === 3.2 Alpha101 扩展 (14) ===
# alpha_006 = -corr(open, vol, 10)
px["alpha_006"] = px.groupby("ts_code", group_keys=False).apply(
    lambda x: x["open"].rolling(10).corr(x["vol"])) * -1

dc = px.groupby("ts_code")["close"].diff(1)
dv = px.groupby("ts_code")["vol"].diff(1)
# alpha_009
min5 = dc.groupby(px["ts_code"]).transform(lambda s: s.rolling(5).min())
max5 = dc.groupby(px["ts_code"]).transform(lambda s: s.rolling(5).max())
px["alpha_009"] = np.where(min5 > 0, dc, np.where(max5 < 0, dc, -dc))
# alpha_012 = sign(delta(vol,1)) * -delta(close,1)
px["alpha_012"] = np.sign(dv) * (-dc)
# alpha_023
hi20 = px.groupby("ts_code")["high"].transform(lambda s: s.rolling(20).mean())
px["alpha_023"] = np.where(hi20 < px["high"], -px.groupby("ts_code")["high"].diff(2), 0.0)

# alpha_001 = (rank(ts_argmax(SignedPower((returns < 0 ? stddev(returns,20) : close), 2.), 5)) - 0.5) * -1
# 简化版: -rank(ts_argmax(close^2, 5)) + 0.5 → 近似 -argmax(close,5)/5 + 0.5
px["alpha_001"] = px.groupby("ts_code")["close"].transform(
    lambda s: s.rolling(5).apply(lambda x: -np.argmax(x)/5.0 + 0.5, raw=True))
# alpha_002 = -1 * delta(rank(log(volume)), 2)
px["_log_vol"] = np.log(px["vol"] + 1)
px["_rank_lv"] = px.groupby("trade_date")["_log_vol"].rank(pct=True)
px["alpha_002"] = px.groupby("ts_code")["_rank_lv"].diff(2) * -1
# alpha_004 = -1 * ts_rank(rank(low), 10)
px["_rank_low"] = px.groupby("trade_date")["low"].rank(pct=True)
px["alpha_004"] = px.groupby("ts_code")["_rank_low"].transform(
    lambda s: s.rolling(10).rank(pct=True)) * -1
# alpha_005 = -1 * (rank(open - sum(vwap,10)/10) * -1 + rank(abs(open - vwap)))
px["_vwap_d"] = (px["high"]+px["low"]+px["close"])/3
px["_sum_vwap10"] = px.groupby("ts_code")["_vwap_d"].transform(lambda s: s.rolling(10).sum())
px["alpha_005"] = -(px.groupby("trade_date")["open"].rank(pct=True) -
                     px.groupby("trade_date")["_sum_vwap10"].rank(pct=True))
# alpha_010 = rank(max(((high<low)?0: 1)*((high-close)/ (close-low)), 6))
px["_hl_ratio"] = np.where(px["high"] <= px["low"], 0,
    ((px["high"]-px["close"])/(px["close"]-px["low"]+1e-9)))
px["alpha_010"] = px.groupby("ts_code")["_hl_ratio"].transform(
    lambda s: s.rolling(6).max())
px["alpha_010"] = px.groupby("trade_date")["alpha_010"].rank(pct=True)
# alpha_017 = rank(vwap - close) / rank(vwap + close)
px["_vc_diff"] = px["_vwap_d"] - px["close"]
px["_vc_sum"]  = px["_vwap_d"] + px["close"]
px["alpha_017"] = px.groupby("trade_date")["_vc_diff"].rank(pct=True) / (
    px.groupby("trade_date")["_vc_sum"].rank(pct=True) + 1e-9)
# alpha_033 = -1 * (1 - open/close)  → 简化
px["alpha_033"] = -1 * (1 - px["open"] / px["close"])
# alpha_041 = power(max(high-low,5), 2) / (sum(high,5)/5 - sum(low,5)/5)
px["_hl"] = (px["high"] - px["low"]).clip(lower=5)
px["_h5"] = px.groupby("ts_code")["high"].transform(lambda s: s.rolling(5).sum())/5
px["_l5"] = px.groupby("ts_code")["low"].transform(lambda s: s.rolling(5).sum())/5
px["alpha_041"] = px["_hl"]**2 / (px["_h5"] - px["_l5"] + 1e-9)
# alpha_054 = -(low - close) * open^5 / ((high - close+1e-9) * low^5)
px["alpha_054"] = -((px["low"]-px["close"]) * px["open"]**5) / (
    (px["high"]-px["close"]+1e-9) * px["low"]**5 + 1e-9)
# alpha_060 = -1 * (2*scale(rank((close-low)/(high-low+1e-9))) - scale(rank((high-close)/(high-low+1e-9))))
px["_cl_hl"] = (px["close"]-px["low"])/(px["high"]-px["low"]+1e-9)
px["_hc_hl"] = (px["high"]-px["close"])/(px["high"]-px["low"]+1e-9)
px["alpha_060"] = -(2*px.groupby("trade_date")["_cl_hl"].rank(pct=True) -
                     px.groupby("trade_date")["_hc_hl"].rank(pct=True))
px.drop(columns=["_log_vol","_rank_lv","_rank_low","_vwap_d","_sum_vwap10",
                  "_hl_ratio","_hl","_h5","_l5","_cl_hl","_hc_hl","_vc_diff","_vc_sum"],
        inplace=True, errors="ignore")
print(f"  [3.2] Alpha101扩展 完成, {time.time()-t0:.0f}s")

# === 3.3 技术指标 (18) ===
# RSI
for w in (6, 14):
    delta = px.groupby("ts_code")["close"].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = px.groupby("ts_code")["gain" if w==6 else "gain14"] if False else None
    px[f"_gain_{w}"] = gain.groupby(px["ts_code"]).transform(lambda s,w=w: s.rolling(w).mean())
    px[f"_loss_{w}"] = loss.groupby(px["ts_code"]).transform(lambda s,w=w: s.rolling(w).mean())
    px[f"rsi_{w}"] = 100 - 100/(1 + px[f"_gain_{w}"]/(px[f"_loss_{w}"]+1e-9))

# KDJ
low_9  = px.groupby("ts_code")["low"].transform(lambda s: s.rolling(9).min())
high_9 = px.groupby("ts_code")["high"].transform(lambda s: s.rolling(9).max())
rsv = (px["close"] - low_9) / (high_9 - low_9 + 1e-9) * 100
rsv = rsv.fillna(50)
px["kdj_k"] = rsv.groupby(px["ts_code"]).transform(lambda s: s.ewm(alpha=1/3, adjust=False).mean())
px["kdj_d"] = px["kdj_k"].groupby(px["ts_code"]).transform(lambda s: s.ewm(alpha=1/3, adjust=False).mean())
px["kdj_j"] = 3 * px["kdj_k"] - 2 * px["kdj_d"]

# MACD
ema12 = px.groupby("ts_code")["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
ema26 = px.groupby("ts_code")["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
px["macd_dif"] = ema12 - ema26
px["macd_dea"] = px["macd_dif"].groupby(px["ts_code"]).transform(lambda s: s.ewm(span=9, adjust=False).mean())
px["macd_hist"] = 2 * (px["macd_dif"] - px["macd_dea"])

# BOLL
ma20 = px.groupby("ts_code")["close"].transform(lambda s: s.rolling(20).mean())
std20 = px.groupby("ts_code")["close"].transform(lambda s: s.rolling(20).std(ddof=1))
px["boll_width"] = 4 * std20 / (ma20 + 1e-9)
px["boll_pctb"]  = (px["close"] - (ma20 - 2*std20)) / (4*std20 + 1e-9)

# CCI
tp = (px["high"] + px["low"] + px["close"]) / 3
tp_ma = tp.groupby(px["ts_code"]).transform(lambda s: s.rolling(14).mean())
tp_md = tp.groupby(px["ts_code"]).transform(lambda s: s.rolling(14).apply(lambda x: np.abs(x-x.mean()).mean(), raw=True))
px["cci_14"] = (tp - tp_ma) / (0.015 * tp_md + 1e-9)

# Williams %R
px["williams_r"] = -100 * (high_9 - px["close"]) / (high_9 - low_9 + 1e-9)

# ROC
for w in (5, 10):
    px[f"roc_{w}"] = px.groupby("ts_code")["close"].pct_change(w) * 100

# ATR
tr = pd.concat([
    px["high"] - px["low"],
    (px["high"] - px.groupby("ts_code")["close"].shift(1)).abs(),
    (px["low"]  - px.groupby("ts_code")["close"].shift(1)).abs(),
], axis=1).max(axis=1)
px["atr_14"] = tr.groupby(px["ts_code"]).transform(lambda s: s.rolling(14).mean())

# 均线位置
for w in (5, 20, 60):
    ma = px.groupby("ts_code")["close"].transform(lambda s,w=w: s.rolling(w).mean())
    px[f"price_ma_ratio_{w}"] = px["close"] / (ma + 1e-9) - 1
ma5  = px.groupby("ts_code")["close"].transform(lambda s: s.rolling(5).mean())
ma20x = px.groupby("ts_code")["close"].transform(lambda s: s.rolling(20).mean())
px["ma_cross_5_20"] = (ma5 - ma20x) / (ma20x + 1e-9)

px.drop(columns=[c for c in px.columns if c.startswith("_gain_") or c.startswith("_loss_")],
        inplace=True, errors="ignore")
print(f"  [3.3] 技术指标 完成, {time.time()-t0:.0f}s")

# === 3.4 滚动统计 (16) ===
for w in (5, 10, 20):
    px[f"return_mean_{w}"] = px.groupby("ts_code")["r"].transform(lambda s,w=w: s.rolling(w).mean())
    px[f"return_std_{w}"]  = px.groupby("ts_code")["r"].transform(lambda s,w=w: s.rolling(w).std(ddof=1))
    px[f"high_max_{w}"]    = px.groupby("ts_code")["high"].transform(lambda s,w=w: s.rolling(w).max())
    px[f"low_min_{w}"]     = px.groupby("ts_code")["low"].transform(lambda s,w=w: s.rolling(w).min())
    px[f"volume_mean_{w}"] = px.groupby("ts_code")["vol"].transform(lambda s,w=w: s.rolling(w).mean())
px["return_skew_20"] = px.groupby("ts_code")["r"].transform(lambda s: s.rolling(20).skew())
px["return_kurt_20"] = px.groupby("ts_code")["r"].transform(lambda s: s.rolling(20).kurt())
px["volume_std_10"]  = px.groupby("ts_code")["vol"].transform(lambda s: s.rolling(10).std(ddof=1))
px["volume_ratio"]   = px["vol"] / (px.groupby("ts_code")["vol"].transform(lambda s: s.rolling(20).mean()) + 1)
# close_rank_20: 当日 close 在过去 20 日的分位
px["close_rank_20"]  = px.groupby("ts_code")["close"].transform(
    lambda s: s.rolling(20).rank(pct=True))
print(f"  [3.4] 滚动统计 完成, {time.time()-t0:.0f}s")

# === 3.5 筹码因子 (6) — 复用 pipeline 逻辑 ===
px["amt3"] = (px["high"]+px["low"]+px["close"])/3.0*px["vol"]
px["vwap_20"] = g["amt3"].transform(lambda s: s.rolling(20).sum()) / (
    1e-9 + g["vol"].transform(lambda s: s.rolling(20).sum()))
px["float_pnl_20"] = (px["close"]-px["vwap_20"])/(px["vwap_20"]+1e-9)
def _prof_pct(x):
    if len(x) < 20: return np.nan
    return float((x < x[-1]).sum())/len(x)
px["prof_pct_20"] = g["close"].transform(lambda s: s.rolling(20).apply(_prof_pct, raw=True))
px["vwap_5"] = g["amt3"].transform(lambda s: s.rolling(5).sum()) / (
    1e-9 + g["vol"].transform(lambda s: s.rolling(5).sum()))
px["chip_shift_5"] = (px["vwap_5"]-px["vwap_20"])/(px["vwap_20"]+1e-9)
px["is_up"] = (px["pct_chg"]>0).astype(float)
px["pos_vol_20"] = (px["vol"]*px["is_up"]).groupby(px["ts_code"]).transform(
    lambda s: s.rolling(20).sum()) / (px["vol"].groupby(px["ts_code"]).transform(
    lambda s: s.rolling(20).sum()) + 1e-9)
# chip_conc_20 (简化版: 用 close std / close mean 近似)
px["chip_conc_20"] = g["close"].transform(lambda s: s.rolling(20).std(ddof=1)) / (
    g["close"].transform(lambda s: s.rolling(20).mean()) + 1e-9)
px.drop(columns=["amt3","vwap_5","is_up"], inplace=True, errors="ignore")
print(f"  [3.5] 筹码因子 完成, {time.time()-t0:.0f}s")

# === 3.6 资金流近似 (5) — 无逐笔数据, 用价量近似 ===
px["mf_approx"] = px["vol"] * (2*(px["close"]>px["open"]).astype(float)-1) * px["close"]
px["net_mf_approx"] = px.groupby("ts_code")["mf_approx"].transform(lambda s: s.rolling(5).sum())
px["mf_5d_sum"]     = px.groupby("ts_code")["mf_approx"].transform(lambda s: s.rolling(5).sum())
px["lg_buy_ratio"]  = px.groupby("ts_code")["vol"].transform(
    lambda s: s.rolling(5).max()) / (px.groupby("ts_code")["vol"].transform(lambda s: s.rolling(5).sum())+1)
px["mf_signal"]     = px.groupby("ts_code")["mf_approx"].transform(
    lambda s: s.rolling(10).mean()) / (px.groupby("ts_code")["mf_approx"].transform(lambda s: s.rolling(10).std(ddof=1))+1e-9)
px["mf_reversal"]   = px.groupby("ts_code")["mf_approx"].transform(
    lambda s: s.rolling(5).sum()) * -1 / (px.groupby("ts_code")["vol"].transform(lambda s: s.rolling(5).sum())*px["close"]+1)
print(f"  [3.6] 资金流近似 完成, {time.time()-t0:.0f}s")

# === 3.7 滞后收益 (4) ===
for lag in (1, 3, 5, 10):
    px[f"lag_return_{lag}"] = px.groupby("ts_code")["r"].shift(lag)
print(f"  [3.7] 滞后收益 完成, {time.time()-t0:.0f}s")

# ---------- 4. 月末快照 ----------
cal = sorted(px["trade_date"].unique())
cal_s = pd.Series(cal)
month_last = cal_s.groupby(cal_s // 100).max().tolist()
month_last = [d for d in month_last if 20191201 <= d <= 20251231]
print(f"[4] 月末快照: {len(month_last)} 月 ({month_last[0]}~{month_last[-1]})")

iw_by_date = {d: set(g2["con_code"]) for d, g2 in iw.groupby("iw_date")}
iw_sorted = iw_dates
def latest_members(rebal_d):
    for d in reversed(iw_sorted):
        if d <= rebal_d: return iw_by_date[d]
    return set()

panel = pd.concat(
    [px[px["trade_date"]==d].assign(is_member=True) for d in month_last],
    ignore_index=True)
panel = panel[panel["is_member"]].copy()
panel["is_member"] = panel.apply(
    lambda r: r["ts_code"] in latest_members(r["trade_date"]), axis=1)
panel = panel[panel["is_member"]].drop(columns=["is_member"])

# 行业映射
ind = pd.read_parquet(os.path.join(ROOT, "research/studies/study_008_enhancements/data/industry_map.parquet"))
panel = panel.merge(ind[["ts_code","industry"]].drop_duplicates(), on="ts_code", how="left")
panel["is_traditional"] = panel["industry"].notna().astype(int)

# 清理辅助列
panel = panel.drop(columns=[c for c in panel.columns if c.startswith("_") or c in
    ["r","mkt_ret","ex_ret","cum20","mf_approx","mkt_ret"]], errors="ignore")

# 列出所有因子列
exclude = {"trade_date","ts_code","open","high","low","close","pct_chg","vol",
           "industry","is_traditional","fwd_20"}
feat_cols = [c for c in panel.columns if c not in exclude]
print(f"[5] 大因子面板: {len(panel):,} 行, {len(feat_cols)} 个因子")
print(f"     因子列表: {feat_cols}")

panel.to_parquet(OUT, index=False)
print(f"[保存] {OUT}")
print(f"[总耗时] {time.time()-t0:.0f}s")
